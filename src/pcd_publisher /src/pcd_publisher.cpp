#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <string>
#include <pcl_ros/point_cloud.h>//ros need
#include <sensor_msgs/PointCloud2.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/common/transforms.h>

std_msgs::Header _velodyne_header;

void publishCloud(const ros::Publisher* in_publisher, const pcl::PointCloud<pcl::PointXYZI>::Ptr in_cloud_to_publish_ptr)
{
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(*in_cloud_to_publish_ptr, cloud_msg);
  cloud_msg.header=_velodyne_header;
  in_publisher->publish(cloud_msg);
}

int
main (int argc, char** argv)
{
  ros::init (argc, argv, "pcd");
  ros::NodeHandle nh;


  ros::Publisher pub;
  pub = nh.advertise<sensor_msgs::PointCloud2> ("/pcd_topic", 1);
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZI>);
  
  if (pcl::io::loadPCDFile<pcl::PointXYZI> ("/home/ee904-i5-old-pc-1/Desktop/ground_ws/src/itri_pcd/first3.pcd", *cloud) == -1) 
  {
    PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
    return (-1);
  }

   Eigen::Matrix4f transform_1 = Eigen::Matrix4f::Identity();
   transform_1 (0,3) = 0;
   transform_1 (1,3) = 0;

   pcl::transformPointCloud(*cloud, *transformed_cloud, transform_1);

  


  _velodyne_header.frame_id = "velodyne";

  
  ros::Rate loop_rate(1);
  while(ros::ok())
  {
     
     publishCloud(&pub, transformed_cloud);
     ros::spinOnce();
     loop_rate.sleep();
  }
  

  return (0);
}
