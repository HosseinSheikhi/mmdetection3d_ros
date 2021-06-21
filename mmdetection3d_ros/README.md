# Instruction:
This package is tested on:

+ Ubuntu 20.04
+ mmdetection-2.11.0
+ mmdet3d v0.14.0
+ ROS2 rolling (main branch) 

NOTE: This package must be compatible with ROS2 Foxy and Galactic cause no extra ROS2 package is used 
but may need some changes in param definitions because there are differences between how Foxy and Galactic define parameters

## Change in mmdet3d
mmdet dataset pipeline supports [LoadImageFromWebcam](https://github.com/open-mmlab/mmdetection/blob/0e09f00a5bf76169386cadbf4c41bd0ff7f6208d/mmdet/datasets/pipelines/loading.py#L81)
and [LoadImageFromFile](https://github.com/open-mmlab/mmdetection/blob/0e09f00a5bf76169386cadbf4c41bd0ff7f6208d/mmdet/datasets/pipelines/loading.py#L12),
but mmdet3d just supports [LoadImageFromFileMono3D](https://github.com/open-mmlab/mmdetection3d/blob/d61476a26c20b2e874647d096c115214635a0bae/mmdet3d/datasets/pipelines/loading.py#L69).
So add the following to `/mmdet3d/datasets/piplines/loading.py`

```angular2html
@PIPELINES.register_module()
class LoadImageFromWebcam3D(LoadImageFromWebcam):
    """Load an image from webcam in monocular 3D object detection. Compared to 2D
    detection, additional camera parameters need to be loaded.

    Args:
        kwargs (dict): Arguments are the same as those in \
            :class:`LoadImageFromWebcam`.
    """

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        super().__call__(results)
        results['cam_intrinsic'] = results['img_info']['cam_intrinsic']
        return results
```

## Build and RUN
simple build and run like other ROS2 packages:
`colcon build`
`ros2 run mmdetection3d_ros mmdetection3d_inference`
