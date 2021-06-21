import mmcv
import numpy as np
import torch
from copy import deepcopy
from mmcv.parallel import collate, scatter
from os import path as osp

from mmdet3d.core import (Box3DMode, DepthInstance3DBoxes,
                          LiDARInstance3DBoxes, show_multi_modality_result)
from mmdet3d.core.bbox import get_box_type
from mmdet3d.datasets.pipelines import Compose
from mmdet3d.core.bbox.structures.cam_box3d import CameraInstance3DBoxes


def inference_mono_3d_detector(model, image):
    """Inference image with the monocular 3D detector.

    Args:
        model (nn.Module): The loaded detector.
        image (str): Image files.
        ann_file (str): Annotation files.

    Returns:
        tuple: Predicted results and data from pipeline.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = deepcopy(cfg.data.test.pipeline)
    test_pipeline = Compose(test_pipeline)
    box_type_3d, box_mode_3d = get_box_type(cfg.data.test.box_type_3d)
    # TODO: filename is the address the results will be saved, it must be change in each call to this function
    # otherwise its just replacing the prev image
    data = dict(
        img=image,
        img_prefix=None,
        img_info=dict(
            filename="/home/hossein/3ddet_ws/demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__CAM_BACK__1532402927637525.jpg"),
        box_type_3d=box_type_3d,
        box_mode_3d=box_mode_3d,
        img_fields=[],
        bbox3d_fields=[],
        pts_mask_fields=[],
        pts_seg_fields=[],
        bbox_fields=[],
        mask_fields=[],
        seg_fields=[])

    # TODO: cam intrinsic must be defined as a ROS2 Param and pass to this function as an argument
    # camera points to image conversion
    if box_mode_3d == Box3DMode.CAM:
        data['img_info'].update(dict(
            cam_intrinsic=[[809.2209905677063, 0.0, 829.2196003259838], [0.0, 809.2209905677063, 481.77842384512485],
                           [0.0, 0.0, 1.0]]))
        # data['img_info'].update(dict(
        #     cam_intrinsic=[[1696.8, 0.0, 960.5], [0.0, 1696.8, 540.5],
        #                    [0.0, 0.0, 1.0]]))

    data = test_pipeline(data)

    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device.index])[0]
    else:
        # this is a workaround to avoid the bug of MMDataParallel
        data['img_metas'] = data['img_metas'][0].data
        data['img'] = data['img'][0].data

    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result, data


def show_result_meshlab(img, data,
                        result,
                        out_dir,
                        score_thr=0.0,
                        show=False,
                        snapshot=False,
                        task='det',
                        palette=None):
    """Show result by meshlab.

    Args:
        img (np.ndarray): image wants to show
        data (dict): Contain data from pipeline.
        result (dict): Predicted result from model.
        out_dir (str): Directory to save visualized result.
        score_thr (float): Minimum score of bboxes to be shown. Default: 0.0
        show (bool): Visualize the results online. Defaults to False.
        snapshot (bool): Whether to save the online results. Defaults to False.
        task (str): Distinguish which task result to visualize. Currently we
            support 3D detection, multi-modality detection and 3D segmentation.
            Defaults to 'det'.
        palette (list[list[int]]] | np.ndarray | None): The palette of
                segmentation map. If None is given, random palette will be
                generated. Defaults to None.
    """
    assert task in ['det', 'multi_modality-det', 'seg', 'mono-det'], \
        f'unsupported visualization task {task}'
    assert out_dir is not None, 'Expect out_dir, got none.'

    if task in ['multi_modality-det', 'mono-det']:
        file_name = show_proj_det_result_meshlab(img, data, result, out_dir,
                                                 score_thr, show, snapshot)

    return out_dir, file_name


def show_proj_det_result_meshlab(img, data,
                                 result,
                                 out_dir,
                                 score_thr=0.0,
                                 show=False,
                                 snapshot=False):
    """Show result of projecting 3D bbox to 2D image by meshlab."""
    assert 'img' in data.keys(), 'image data is not provided for visualization'

    img_filename = data['img_metas'][0][0]['filename']
    file_name = osp.split(img_filename)[-1].split('.')[0]

    # read from file because img in data_dict has undergone pipeline transform
    # img = mmcv.imread(img_filename)


    if 'pts_bbox' in result[0].keys():
        result[0] = result[0]['pts_bbox']
    elif 'img_bbox' in result[0].keys():
        result[0] = result[0]['img_bbox']
    pred_bboxes = result[0]['boxes_3d'].tensor.numpy()
    pred_scores = result[0]['scores_3d'].numpy()

    # filter out low score bboxes for visualization
    if score_thr > 0:
        inds = pred_scores > score_thr
        pred_bboxes = pred_bboxes[inds]

    box_mode = data['img_metas'][0][0]['box_mode_3d']
    if box_mode == Box3DMode.LIDAR:
        if 'lidar2img' not in data['img_metas'][0][0]:
            raise NotImplementedError(
                'LiDAR to image transformation matrix is not provided')

        show_bboxes = LiDARInstance3DBoxes(pred_bboxes, origin=(0.5, 0.5, 0))

        show_multi_modality_result(
            img,
            None,
            show_bboxes,
            data['img_metas'][0][0]['lidar2img'],
            out_dir,
            file_name,
            box_mode='lidar',
            show=show)
    elif box_mode == Box3DMode.DEPTH:
        if 'calib' not in data.keys():
            raise NotImplementedError(
                'camera calibration information is not provided')

        show_bboxes = DepthInstance3DBoxes(pred_bboxes, origin=(0.5, 0.5, 0))

        show_multi_modality_result(
            img,
            None,
            show_bboxes,
            data['calib'][0],
            out_dir,
            file_name,
            box_mode='depth',
            img_metas=data['img_metas'][0][0],
            show=show)
    elif box_mode == Box3DMode.CAM:
        if 'cam_intrinsic' not in data['img_metas'][0][0]:
            raise NotImplementedError(
                'camera intrinsic matrix is not provided')

        from mmdet3d.core.bbox import mono_cam_box2vis
        show_bboxes = CameraInstance3DBoxes(
            pred_bboxes, box_dim=pred_bboxes.shape[-1], origin=(0.5, 1.0, 0.5))
        # TODO: remove the hack of box from NuScenesMonoDataset
        show_bboxes = mono_cam_box2vis(show_bboxes)

        show_multi_modality_result(
            img,
            None,
            show_bboxes,
            data['img_metas'][0][0]['cam_intrinsic'],
            out_dir,
            file_name,
            box_mode='camera',
            show=show)
    else:
        raise NotImplementedError(
            f'visualization of {box_mode} bbox is not supported')

    return file_name
