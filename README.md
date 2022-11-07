# Pointcloud_nn
ROS2 node for applying machine learning based methods on PointCloud2 messages and republishing processed clouds.
 - current implementation is running [4DMOS](https://github.com/PRBonn/4DMOS) motion segmentation netowork which can be easily swapped.
 - [config/params.yaml](https://github.com/tau-alma/pointcloud_nn/blob/main/config/params.yaml) contains the parameters to be adjusted.
## [predictor_node.py](https://github.com/tau-alma/pointcloud_nn/blob/main/pointcloud_nn/predictor_node.py)
 - This file contains the Subscriber class where your main prediction operation should take place.
 - It converts the incoming pointcloud into X, Y, Z, time np array and saves a window of recent pointclouds into the local variable [cloud_list](https://github.com/tau-alma/pointcloud_nn/blob/9c1b2f9532fa4ea1415c22fc9da76f3fc6c78b06/pointcloud_nn/predictor_node.py#L76). 
 - Load and initialize your model during subscriber class intialization and perform your predictions in the [cloud_callback](https://github.com/tau-alma/pointcloud_nn/blob/9c1b2f9532fa4ea1415c22fc9da76f3fc6c78b06/pointcloud_nn/predictor_node.py#L91) function,


