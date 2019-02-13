#include <ros/ros.h>
#include <ros/package.h>
#include <std_msgs/String.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <std_msgs/Bool.h>
#include <iostream>
#include <ros/ros.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>
#include <pcl_ros/point_cloud.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>

#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/project_inliers.h>

#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/don.h>
#include <pcl/features/fpfh_omp.h>

#include <pcl/kdtree/kdtree.h>

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>

#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>

#include <pcl/common/common.h>

#include <pcl/search/organized.h>
#include <pcl/search/kdtree.h>


std_msgs::Header _velodyne_header;
ros::Publisher _pub_ground_cloud;
ros::Publisher _pub_nonground_cloud;


using namespace std;



void segGround(const pcl::PointCloud<pcl::PointXYZI>::Ptr in_cloud_ptr, 
                     pcl::PointCloud<pcl::PointXYZI>::Ptr out_ground_cloud_ptr,
                     pcl::PointCloud<pcl::PointXYZI>::Ptr out_nonground_cloud_ptr)
{
  pcl::SACSegmentation<pcl::PointXYZI> seg;
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);


  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  
  seg.setDistanceThreshold(0.3);  // floor distance
  seg.setOptimizeCoefficients(true);
  seg.setInputCloud(in_cloud_ptr);
  seg.segment(*inliers, *coefficients);
  if (inliers->indices.size() == 0)
  {
    std::cout << "Could not estimate a LINE model for the given dataset." << std::endl;
  }


  pcl::ExtractIndices<pcl::PointXYZI> extract;
  extract.setInputCloud (in_cloud_ptr);
  extract.setIndices(inliers);
  extract.setNegative(false);//true removes the indices, false leaves only the indices
  extract.filter(*out_ground_cloud_ptr);
  extract.setNegative(true);//true removes the indices, false leaves only the indices
  extract.filter(*out_nonground_cloud_ptr);
 

}


void publishCloud(const ros::Publisher* in_publisher, const pcl::PointCloud<pcl::PointXYZI>::Ptr in_cloud_to_publish_ptr)
{

  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(*in_cloud_to_publish_ptr, cloud_msg);
  cloud_msg.header=_velodyne_header;
  in_publisher->publish(cloud_msg);
}



void velodyne_callback(const sensor_msgs::PointCloud2ConstPtr& in_sensor_cloud)
{
  
  pcl::PointCloud<pcl::PointXYZI>::Ptr current_sensor_cloud_ptr (new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr ground_cloud_ptr (new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr nonground_cloud_ptr (new pcl::PointCloud<pcl::PointXYZI>);

  const clock_t begin_time = clock();

  _velodyne_header = in_sensor_cloud->header;

  double t = double(_velodyne_header.stamp.sec) + double(_velodyne_header.stamp.nsec)*1e-9;

  pcl::fromROSMsg(*in_sensor_cloud, *current_sensor_cloud_ptr);


  segGround(current_sensor_cloud_ptr, ground_cloud_ptr, nonground_cloud_ptr);

  publishCloud(&_pub_ground_cloud, ground_cloud_ptr);
  publishCloud(&_pub_nonground_cloud, nonground_cloud_ptr);

  std::cout << double( clock () - begin_time)/CLOCKS_PER_SEC<<std::endl;

  // ground_cloud_ptr->width = 1;
  // ground_cloud_ptr->height = ground_cloud_ptr->points.size();

  // nonground_cloud_ptr->width = 1;
  // nonground_cloud_ptr->height = nonground_cloud_ptr->points.size();

  // pcl::PCDWriter writer_1;
  // writer_1.write<pcl::PointXYZI> ("plane_ground_3.pcd", *ground_cloud_ptr, false);

  // pcl::PCDWriter writer_2;
  // writer_2.write<pcl::PointXYZI> ("plane_nonground_3.pcd", *nonground_cloud_ptr, false);

  // int threshold = Otsu(ground_cloud_ptr,255);

  // std::cout<<threshold<<std::endl;

}


int main(int argc, char **argv)
{

  ros::init(argc, argv, "pcl_plane_model");
  ros::NodeHandle nh;

  ros::Subscriber sub = nh.subscribe ("/points_raw", 1, velodyne_callback);

  _pub_ground_cloud = nh.advertise<sensor_msgs::PointCloud2>("/plane_ground_points",1);
  _pub_nonground_cloud = nh.advertise<sensor_msgs::PointCloud2>("/plane_nonground_points",1);

  ros::spin ();
}