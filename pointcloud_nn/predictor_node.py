import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import String
import pcl
import numpy as np
from .pcd2_conversions import pointcloud2_to_xyz_array, array_to_pointcloud2
import torch
import mos4d.models.models as models
import mos4d.datasets.datasets as datasets
from pytorch_lightning import Trainer
from .get_lables import lables_ros
from std_msgs.msg import Header

def point_cloud(points, parent_frame):
    """ Creates a point cloud message.
    Args:
        points: Nx7 array of xyz positions (m) and rgba colors (0..1)
        parent_frame: frame in which the point cloud is defined
    Returns:
        sensor_msgs/PointCloud2 message
    """
    ros_dtype = PointField.FLOAT32
    dtype = np.float32
    itemsize = np.dtype(dtype).itemsize

    data = points.astype(dtype).tobytes()

    fields = [PointField(
        name=n, offset=i*itemsize, datatype=ros_dtype, count=1)
        for i, n in enumerate('xyzrgba')]

    header = Header()
    header.frame_id = parent_frame

    return PointCloud2(
        header=header,
        height=1,
        width=points.shape[0],
        is_dense=False,
        is_bigendian=False,
        fields=fields,
        point_step=(itemsize * 7),
        row_step=(itemsize * 7 * points.shape[0]),
        data=data
    )
def timestamp_tensor(tensor, time):
        """Add time as additional column to tensor"""
        n_points = tensor.shape[0]
        time = time * torch.ones((n_points, 1))
        timestamped_tensor = torch.hstack([tensor, time])
        return timestamped_tensor


class CloudSubscriber(Node):
    """ROS2 node for using Poitncloud2 message with learning based preditions"""
    def __init__(self):
        super().__init__('predictor_node')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('model_path', 'None'),
                ('semantic_config_path', 'None'),
                ('input_cloud_topic', 'None'),
                ('window_size', 5)
            ])

        self.window_size = self.get_parameter('model_path').get_parameter_value().integer_value
        self.subscription = self.create_subscription(
            PointCloud2,
            self.get_parameter('input_cloud_topic').get_parameter_value().string_value,
            self.cloud_callback,
            10)
        self.subscription  
        self.cloud_list = []
        self.publish_seg = self.create_publisher(
            PointCloud2,
            '/velodyne_points_motion_seg',10)
        self.time = 0.0
        #setup your model for predictions here
        weights = self.get_parameter('model_path').get_parameter_value().string_value
        self.cfg = torch.load(weights)["hyper_parameters"]
        self.cfg["DATA"]["SPLIT"]["TEST"] = [3]
        self.cfg["DATA"]["TRANSFORM"] = False
        self.cfg["DATA"]["SEMANTIC_CONFIG_FILE"] = self.get_parameter('semantic_config_path').get_parameter_value().string_value
        self.model = models.MOSNet.load_from_checkpoint(weights, hparams=self.cfg)
        self.trainer = Trainer(gpus=1, logger=False, enable_progress_bar=False)
        self.conf_list = []

    def cloud_callback(self, msg):
        """ Function for passing pointcloud through the network and generating labels. 
        Args:
            sensor_msgs/PointCloud2 : input point cloud
        """
        #Converting Pointcloud into (X,Y,Z, Time) numpy array and adding to the sliding window. 
        ros_time = float(str(msg.header.stamp.sec)+"."+str(msg.header.stamp.nanosec))
        #self.time = ros_time             #uncomment this if ROS time is needed
        xyz_cloud = pointcloud2_to_xyz_array(msg)
        xyz_cloud = torch.Tensor(xyz_cloud)
        xyz_cloud_t = timestamp_tensor(xyz_cloud, self.time)
        self.cloud_list.append(xyz_cloud_t)

        #Updating the sliding window.
        if len(self.cloud_list) > self.window_size:
            self.cloud_list = self.cloud_list[1:]

            #Predict using your model here
            data = datasets.KittiSequentialModule(self.cfg, self.cloud_list)
            data.setup()
            confideneces = self.trainer.predict(self.model, data.test_dataloader())
            self.conf_list.append(confideneces)

            if len(self.conf_list) > self.window_size:
                self.conf_list = self.conf_list[1:]
                self.publish_seg_cloud(self.conf_list, self.cloud_list[0])
        
        self.time += 0.1 #for Lidars running at 10Hz

    def publish_seg_cloud(self, confideneces, xyz_cloud):
        """ Publishes output poitncloud by using the labels
        Args:
            xyz_cloud: the poitncloud array corresponding to the lables
            confideneces: confidence values for generating labels
        Returns:
            sensor_msgs/PointCloud2 message
        """
        labels = lables_ros(confideneces) 
        if labels.shape[0] > 10:
            #print("Got Lables",labels)
            out_cloud = np.array(xyz_cloud)
            out_cloud[:,-1] = labels
            out_cloud = out_cloud[out_cloud[:,-1] == 251]
            out_cloud = out_cloud[:,:-1]
            #print(out_cloud.shape, labels.shape)
            rgba  = np.zeros((out_cloud.shape[0],4))
            rgba[:,-1] += 1
            rgba[:, 0] += 1
            out_cloud = np.concatenate((out_cloud,rgba),axis=1)
            seg_msg = point_cloud(out_cloud, 'velodyne')
            self.publish_seg.publish(seg_msg)

def main(args=None):
    rclpy.init(args=args)

    predictor_node = CloudSubscriber()

    rclpy.spin(predictor_node)
    predictor_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()