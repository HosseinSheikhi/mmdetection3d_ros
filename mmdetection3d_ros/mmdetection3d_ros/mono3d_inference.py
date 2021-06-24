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


class MMDET3DInference(Node):
    def __init__(self):
        super().__init__('mmdetection3d_inference')
        # define parameters, TODO: must be defined as a ROS2 param and read from the config file
        self.cfg = None
        self.checkpoint = None
        self.device = None
        self.cam_intrinsic = None
        self.img_sub_topic = None
        self.image_ctr = 1

        self.parameters()

        #self.img_subscriber = self.create_subscription(Image, self.img_sub_topic, self.img_cb,
                                                       #qos_profile=qos_profile_services_default)
        # build the model from a config file and a checkpoint file
        #self.model = init_model(self.cfg, self.checkpoint, device=self.device)
        #self.data = inference_init_data_dict(self.model, cam_intrinsic=self.cam_intrinsic)

    def parameters(self):
        param_cfg_descriptor = ParameterDescriptor(type=ParameterType.PARAMETER_STRING,
                                                    description='address to the config file of the desired detector')
        self.declare_parameter('detector_cfg', 'naaamoosan', param_cfg_descriptor)

        self.cfg = self.get_parameter('detector_cfg').get_parameter_value().string_value
        print(self.cfg)
    def inference(self, image):
        """
        :param image: image that must be fed to model for 3d detection
        :return:
        """
        # test a single image
        modified_data = copy.deepcopy(self.data)
        modified_data['img'] = image
        modified_data['img_info']['filename'] = self.data['img_info']['filename'] + "yoyo_" + str(
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
        # show the results
        # TODO: extra params must be read as ROS2 parameters from config file instead of being hardcoded
        show_result_meshlab(image,
                            data,
                            result,
                            'demo',
                            0.01,
                            show=True,
                            snapshot=True,
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
