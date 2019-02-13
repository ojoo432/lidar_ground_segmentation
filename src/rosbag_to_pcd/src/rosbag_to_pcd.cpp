#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <string>
#include <pcl_ros/point_cloud.h>//ros need
#include <sensor_msgs/PointCloud2.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/extract_indices.h>


std_msgs::Header _velodyne_header;

void pcd_callback(const sensor_msgs::PointCloud2ConstPtr& in_sensor_cloud)
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr current_sensor_cloud_ptr (new pcl::PointCloud<pcl::PointXYZI>);
   
    _velodyne_header = in_sensor_cloud->header;
    
    _velodyne_header.frame_id = "velodyne";

    std::string pcd_name = boost::to_string(_velodyne_header.stamp);

    pcl::fromROSMsg(*in_sensor_cloud, *current_sensor_cloud_ptr);

    current_sensor_cloud_ptr->width = 1;
    current_sensor_cloud_ptr->height = current_sensor_cloud_ptr->points.size();

    pcl::PCDWriter writer;
    writer.write<pcl::PointXYZI> ( pcd_name + ".pcd", *current_sensor_cloud_ptr, false);

    
  // outlier_cloud_ptr->width = 1;
  // outlier_cloud_ptr->height = outlier_cloud_ptr->points.size();

  // cloud->width = 1;
  // cloud->height = cloud->points.size();

  // pcl::PCDWriter writer_1;
  // writer_1.write<pcl::PointXYZ> ( "plane_nonground_3_false.pcd", *outlier_cloud_ptr, false);

  // pcl::PCDWriter writer_2;
  // writer_2.write<pcl::PointXYZ> ( "plane_nonground_3_true.pcd", *cloud, false);


}

int main(int argc, char **argv)
{
  
  ros::init(argc, argv, "rosbag_to_pcd");
  ros::NodeHandle nh;


  ros::Subscriber sub = nh.subscribe ("/velodyne_points", 1, pcd_callback);

  ros::spin ();
}
