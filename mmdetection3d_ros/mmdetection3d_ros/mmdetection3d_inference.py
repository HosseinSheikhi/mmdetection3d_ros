import mmcv
from mmdet3d.apis import init_model
from mmdetection3d_ros.inference import inference_mono_3d_detector, show_result_meshlab
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_services_default
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class MMDET3DInference(Node):
    def __init__(self):
        super().__init__('mmdetection3d_inference')
        # define parameters, TODO: must be defined as a ROS2 param and read from the config file
        self.cfg = 'configs/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d.py'
        self.checkpoint = 'checkpoints/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_20210425_181341-8d5a21fe.pth'
        self.device = 'cuda:0'

        self.img_subscriber = self.create_subscription(Image, "/camera/image_raw", self.img_cb,
                                                       qos_profile=qos_profile_services_default)
        self.image = None  # TODO: will be removed, just for test

        # build the model from a config file and a checkpoint file
        self.model = init_model(self.cfg, self.checkpoint, device=self.device)

    def load_imag(self):
        """
        Loads image from disk, TODO: will be removed, is defined just for test
        :return:
        """
        file_client = mmcv.FileClient(**dict(backend='disk'))
        img_bytes = file_client.get(
            'demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__CAM_BACK__1532402927637525.jpg')
        self.image = mmcv.imfrombytes(img_bytes, flag='color')

    def inference(self, image):
        """
        :param image(np.ndarray): image that must be fed to model for 3d detection
        :return:
        """
        # test a single image
        result, data = inference_mono_3d_detector(self.model, image)
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
