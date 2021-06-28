from launch import LaunchDescription
from launch_ros.actions import Node
import pathlib

parameters_file_name = '/home/hossein/3ddet_ws/src/mmdetection3d_ros/config/mono3d_inference_params.yaml'


def generate_launch_description():

    parameters_file_path = parameters_file_name
    return LaunchDescription([
        Node(
            package='mmdetection3d_ros',
            namespace='mmdetection3d_ros',
            # Unique namespaces allow the system to start two simulators without node name nor topic name conflicts
            executable='mono3d_inference_node',
            output='screen',
            emulate_tty=True,
            parameters=[
                parameters_file_path
                #{'detector_cfg': 'src/mmdetection3d_ros/configs/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d.py'}
            ]
        )
    ])
