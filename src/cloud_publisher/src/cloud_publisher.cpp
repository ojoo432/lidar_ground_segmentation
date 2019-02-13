#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <string>
#include <pcl_ros/point_cloud.h>//ros need
#include <sensor_msgs/PointCloud2.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/extract_indices.h>


ros::Publisher _pub_out_cloud;
std_msgs::Header _velodyne_header;
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr out_cloud (new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr outlier_cloud_ptr (new pcl::PointCloud<pcl::PointXYZ>);
// pcl::io::loadPCDFile<pcl::PointXYZI> ("~/Desktop/ground_ws/src/pcd/" + file_name + ".pcd", *cloud) //* load the file


void publishCloud(const ros::Publisher* in_publisher, const pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud_to_publish_ptr)
{
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(*in_cloud_to_publish_ptr, cloud_msg);
  cloud_msg.header=_velodyne_header;
  in_publisher->publish(cloud_msg);
}

void select_callback(const sensor_msgs::PointCloud2ConstPtr& in_sensor_cloud)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr current_sensor_cloud_ptr (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr remain_cloud_ptr (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud_ptr (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());
    
    _velodyne_header.frame_id = "velodyne";

    pcl::fromROSMsg(*in_sensor_cloud, *current_sensor_cloud_ptr);

    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud (cloud);

    int K = 1;

    std::vector<int> pointIdxNKNSearch(K);
    std::vector<float> pointNKNSquaredDistance(K);

    for (size_t i = 0; i < current_sensor_cloud_ptr->points.size (); ++i)
  {
      if ( kdtree.nearestKSearch (current_sensor_cloud_ptr->points[i], K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0 )
    {
      inliers->indices.push_back(pointIdxNKNSearch[0]);
    }
  }

  pcl::ExtractIndices<pcl::PointXYZ> extract;
  extract.setInputCloud (cloud);
  extract.setIndices (inliers);
  extract.setNegative (false);
  extract.filter (*remain_cloud_ptr);
  extract.setNegative (true);
  extract.filter (*filtered_cloud_ptr);

  pcl::copyPointCloud(*filtered_cloud_ptr, *cloud);

  for (size_t i = 0; i < remain_cloud_ptr->points.size (); ++i)
  {
    pcl::PointXYZ current_point;
    current_point.x = remain_cloud_ptr->points[i].x;
    current_point.y = remain_cloud_ptr->points[i].y;
    current_point.z = remain_cloud_ptr->points[i].z;
 
    outlier_cloud_ptr->points.push_back(current_point);
  }

  publishCloud(&_pub_out_cloud, cloud);

  std::cout<<"false_size:"<<outlier_cloud_ptr->points.size()<<std::endl;

  outlier_cloud_ptr->width = 1;
  outlier_cloud_ptr->height = outlier_cloud_ptr->points.size();

  cloud->width = 1;
  cloud->height = cloud->points.size();

  pcl::PCDWriter writer_1;
  writer_1.write<pcl::PointXYZ> ( "plane_nonground_3_false.pcd", *outlier_cloud_ptr, false);

  pcl::PCDWriter writer_2;
  writer_2.write<pcl::PointXYZ> ( "plane_nonground_3_true.pcd", *cloud, false);


}

int main(int argc, char **argv)
{
  
  ros::init(argc, argv, "selected_points_publisher");
  ros::NodeHandle nh;

  
  if (pcl::io::loadPCDFile<pcl::PointXYZ> ("/home/ee904-i5-old-pc-1/Desktop/ground_ws/1.pcd", *cloud)== -1)
  {
    return (-1);
  }

  ros::Subscriber sub = nh.subscribe ("/selected_pointcloud", 1, select_callback);
  _pub_out_cloud = nh.advertise<sensor_msgs::PointCloud2> ("/point_frame", 1);

  ros::spin ();
}
