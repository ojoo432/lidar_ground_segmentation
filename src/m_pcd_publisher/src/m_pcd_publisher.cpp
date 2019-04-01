#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <string>
#include <pcl_ros/point_cloud.h>//ros need
#include <sensor_msgs/PointCloud2.h>

std_msgs::Header _velodyne_header;
ros::Publisher pub_gct;
ros::Publisher pub_gcf;
ros::Publisher pub_ngct;
ros::Publisher pub_ngcf;

void publishCloud(const ros::Publisher* in_publisher, const pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud_to_publish_ptr)
{
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(*in_cloud_to_publish_ptr, cloud_msg);
  cloud_msg.header=_velodyne_header;
  in_publisher->publish(cloud_msg);
}

void _callback(const sensor_msgs::PointCloud2ConstPtr& in_sensor_cloud)
{
  _velodyne_header.frame_id = "velodyne";

  pcl::PointCloud<pcl::PointXYZ>::Ptr ground_cloud_t (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr ground_cloud_f (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr nonground_cloud_t (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr nonground_cloud_f (new pcl::PointCloud<pcl::PointXYZ>);
  if (pcl::io::loadPCDFile<pcl::PointXYZ> ("/home/ee904-i5-old-pc-1/Desktop/ground_ws/src/itri_pcd/horizontal.pcd", *ground_cloud_t) == -1) 
  {
    PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
  }

  if (pcl::io::loadPCDFile<pcl::PointXYZ> ("/home/ee904-i5-old-pc-1/Desktop/ground_ws/src/itri_pcd/select1.pcd", *ground_cloud_f) == -1) //* load the file
  {
    PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
  }

  // if (pcl::io::loadPCDFile<pcl::PointXYZ> ("/home/ee904-i5-old-pc-1/Desktop/ground_ws/plane_nonground_3_true.pcd", *nonground_cloud_t) == -1) //* load the file
  // {
  //   PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
  // }

  // if (pcl::io::loadPCDFile<pcl::PointXYZ> ("/home/ee904-i5-old-pc-1/Desktop/ground_ws/plane_nonground_3_false.pcd", *nonground_cloud_f) == -1) //* load the file
  // {
  //   PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
  // }

  std::cout<<"ground_t:"<<ground_cloud_t->points.size()<<std::endl;
  std::cout<<"ground_f:"<<ground_cloud_f->points.size()<<std::endl;
  // std::cout<<"nonground_t:"<<nonground_cloud_t->points.size()<<std::endl;
  // std::cout<<"nonground_f:"<<nonground_cloud_f->points.size()<<std::endl;

  publishCloud(&pub_gct,ground_cloud_t);
  publishCloud(&pub_gcf,ground_cloud_f);
  // publishCloud(&pub_ngct,nonground_cloud_t);
  // publishCloud(&pub_ngcf,nonground_cloud_f);

}


int main(int argc, char **argv)
{
  
  ros::init(argc, argv, "m_pcd_publisher");
  ros::NodeHandle nh;

  ros::Subscriber sub = nh.subscribe ("/pcd_topic", 1, _callback);

  pub_gct = nh.advertise<sensor_msgs::PointCloud2> ("ground_true", 1);
  pub_gcf = nh.advertise<sensor_msgs::PointCloud2> ("ground_false", 1);
  pub_ngct = nh.advertise<sensor_msgs::PointCloud2> ("nonground_true", 1);
  pub_ngcf = nh.advertise<sensor_msgs::PointCloud2> ("nonground_false", 1);

  ros::spin ();
}
