from setuptools import setup
import os
from glob import glob
package_name = 'mmdetection3d_ros'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*_launch.py')),
        (os.path.join('share', package_name), glob('config/*.yaml')),
        (os.path.join('share', package_name), glob('*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hossein',
    maintainer_email='hsa150@sfu.ca',
    description='Wrap the mmdetection3d inference as a ROS2 node',
    license='Apache License 2.0',
    tests_require=['pytest'],

    entry_points={
        'console_scripts': [
            'mono3d_inference_node = mmdetection3d_ros.mono3d_inference:main'
        ],
    },
)
