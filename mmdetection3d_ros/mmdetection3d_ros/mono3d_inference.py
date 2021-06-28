import mmcv
from mmdet3d.apis import init_model
from mmdetection3d_ros.inference import inference_mono_3d_detector, show_result_meshlab, inference_init_data_dict
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_services_default
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import copy
import numpy as np


class MMDET3DInference(Node):
    def __init__(self):
        super().__init__('mono3d_inference')
        # define parameters,
        self.cfg = None
        self.checkpoint = None
        self.device = ""
        self.cam_intrinsic = None
        self.out_dir = None
        self.score_thr = None
        self.show = None
        self.snapshot = None

        self.img_sub_topic = ""
        self.image_ctr = 0

        self.parameters()

        self.img_subscriber = self.create_subscription(Image, self.img_sub_topic, self.img_cb,
                                                       qos_profile=qos_profile_services_default)
        # build the model from a config file and a checkpoint file
        self.model = init_model(self.cfg, self.checkpoint, device=self.device)
        # initialize the data
        self.data = inference_init_data_dict(self.model, cam_intrinsic=self.cam_intrinsic)

    def parameters(self):
        # define parameters
        self.declare_parameter("mmdet_3d.detector_cfg", "")
        self.declare_parameter('mmdet_3d.checkpoint', "")
        self.declare_parameter('mmdet_3d.device', 'cuda:0')
        self.declare_parameter('mmdet_3d.cam_intrinsic', "")
        self.declare_parameter('mmdet_3d.out_dir', "")
        self.declare_parameter('mmdet_3d.score_thr', 0.0)
        self.declare_parameter('mmdet_3d.show', True)
        self.declare_parameter('mmdet_3d.snapshot', True)
        self.declare_parameter('mmdet3d_ros.img_sub_topic', "")

        # get parameters
        self.cfg = self.get_parameter("mmdet_3d.detector_cfg").get_parameter_value().string_value
        self.checkpoint = self.get_parameter('mmdet_3d.checkpoint').get_parameter_value().string_value
        self.device = self.get_parameter('mmdet_3d.device').get_parameter_value().string_value
        # parse cam_intrinsic string
        temp_cam_int = self.get_parameter('mmdet_3d.cam_intrinsic').get_parameter_value().string_value
        temp_cam_int = temp_cam_int.replace("[", "").replace("]", "").split(',')
        for i in range(0, len(temp_cam_int)):
            temp_cam_int[i] = float(temp_cam_int[i])
        self.cam_intrinsic = np.array(temp_cam_int).reshape((3, 3))

        self.out_dir = self.get_parameter('mmdet_3d.out_dir').get_parameter_value().string_value
        self.score_thr = self.get_parameter('mmdet_3d.score_thr').get_parameter_value().double_value
        self.show = self.get_parameter('mmdet_3d.show').get_parameter_value().bool_value
        self.snapshot = self.get_parameter('mmdet_3d.snapshot').get_parameter_value().bool_value
        self.img_sub_topic = self.get_parameter('mmdet3d_ros.img_sub_topic').get_parameter_value().string_value

    def inference(self, image):
        """
        :param image: image that must be fed to model for 3d detection
        :return:
        """
        # test a single image
        modified_data = copy.deepcopy(self.data)
        modified_data['img'] = image
        modified_data['img_info']['filename'] = self.data['img_info']['filename'] + "inference_" + str(
            self.image_ctr) + ".jpg"
        print(modified_data['img_info']['filename'])

        result, data = inference_mono_3d_detector(self.model, image, modified_data)
        return result, data

    def show_inferennce(self, image, data, result):
        """
        :param image: image that must be shown
        :param result: results of inference including predicted bboxes
        :param data: data pipline, including image meta info
        :return:
        """
        show_result_meshlab(image,
                            data,
                            result,
                            self.out_dir,
                            self.score_thr,
                            show=self.show,
                            snapshot=self.snapshot,
                            task='mono-det')

    def img_cb(self, image):
        """
        ROS2 callbacks for the subscribed image, converts the input to a np.ndarray and calls the inference and show results
        :param image(sensor_msg.msg.Image): ROS2 topic
        :return:
        """
        self.image_ctr += 1
        cv_image = CvBridge().imgmsg_to_cv2(image, desired_encoding='bgr8')
        res, data = self.inference(cv_image)
        self.show_inferennce(cv_image, data, res)


def main(args=None):
    rclpy.init(args=args)

    mmdet3d_inference = MMDET3DInference()

    rclpy.spin(mmdet3d_inference)
    mmdet3d_inference.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
