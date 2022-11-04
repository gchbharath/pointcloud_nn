import os
from glob import glob
from setuptools import setup

package_name = 'pointcloud_nn'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*launch.[pxy][yma]*')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='bharath',
    maintainer_email='bharath.garigipati@tuni.fi',
    description='Node for applying machine learning methods on Pointclouds',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'predictor_node = pointcloud_nn.predictor_node:main'
        ],
    },
)
