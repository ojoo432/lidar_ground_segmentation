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

void publishCloud(const ros::Publisher* in_publisher, const pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud_to_publish_ptr)
{
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(*in_cloud_to_publish_ptr, cloud_msg);
  cloud_msg.header=_velodyne_header;
  in_publisher->publish(cloud_msg);
}

void printOutAngle(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
  float dx,dy,dz;
  std::vector<float> angle;

  std::cout<<"i: "<<cloud->points.size()<<std::endl;

  for(unsigned int i=0; i< cloud->points.size(); i++)
  {
    // std::cout<<"i: "<<i<<std::endl;
    // std::cout<<cloud->points[i]<<std::endl;
    if(i == cloud->points.size()-1)
      {
        std::cout<<"last row"<<std::endl;
        angle.push_back(0.0);
      }
    else
    {
      dx = fabs(cloud->points[i].x - cloud->points[i+1].x);
      dy = fabs(cloud->points[i].y - cloud->points[i+1].y);
      dz = fabs(cloud->points[i].z - cloud->points[i+1].z);

      std::cout<<"row: "<<i+1<<" angle: "<<atan2(dz, sqrt(dx*dx + dy*dy))<<std::endl;
      angle.push_back(atan2(dz, sqrt(dx*dx + dy*dy)));
    }
  }

  for(int i =0; i < angle.size(); i++)
  {
    if(i == cloud->points.size()-1)
      {
        std::cout<<"row: "<<i+1<<"delta_angle: "<<angle[i]<<std::endl;
      }
    else
    {
      std::cout<<fabs(angle[i]-angle[i+1])<<std::endl;
    }
  }
}

int
main (int argc, char** argv)
{
  ros::init (argc, argv, "pcd");
  ros::NodeHandle nh;


  ros::Publisher pub;
  ros::Publisher _pub;
  pub = nh.advertise<sensor_msgs::PointCloud2> ("/slope", 1);
  _pub = nh.advertise<sensor_msgs::PointCloud2> ("/horizontal", 1);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZ>);
  
  if (pcl::io::loadPCDFile<pcl::PointXYZ> ("/home/ee904-i5-old-pc-1/Desktop/ground_ws/src/itri_pcd/row3_project_sample.pcd", *cloud) == -1) 
  {
    PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
    return (-1);
  }

  //  if (pcl::io::loadPCDFile<pcl::PointXYZI> ("/home/ee904-i5-old-pc-1/Desktop/ground_ws/src/itri_pcd/first4.pcd", *transformed_cloud) == -1) 
  // {
  //   PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
  //   return (-1);
  // }

  

  Eigen::Affine3f transform_2 = Eigen::Affine3f::Identity();

  float theta = -1*M_PI/18; 

  // Define a translation of 0.0 meters on the x axis.
  // transform_2.translation() << 0.0, 0.0, 1.7;
  transform_2.translation() << 0.0, 0.0, 0.0;

  // The same rotation matrix as before; theta radians around Z axis
  transform_2.rotate (Eigen::AngleAxisf (theta, Eigen::Vector3f::UnitY()));

  // Print the transformation
  printf ("\nMethod #2: using an Affine3f\n");
  std::cout << transform_2.matrix() << std::endl;

  // You can either apply transform_1 or transform_2; they are the same
  // pcl::transformPointCloud (*cloud, *transformed_cloud, transform_2);

   // Eigen::Matrix4f transform_1 = Eigen::Matrix4f::Identity();
   // transform_1 (0,3) = 0;
   // transform_1 (1,3) = 0;

   // pcl::transformPointCloud(*cloud, *transformed_cloud, transform_1);

  int number = 18;

  float rotate_angle = 0;

  std::cout<<"----cloud----"<<std::endl;
  printOutAngle(cloud);
  for(int n = 0; n < number;n++)
  {
    if(n ==0)
    {
      rotate_angle = theta*180/M_PI + rotate_angle;
      pcl::transformPointCloud (*cloud, *transformed_cloud, transform_2);
      std::cout<<"----transform_"<<n+1<<"_cloud----"<<std::endl;
      std::cout<<"rotate_angle: "<<rotate_angle<<std::endl;
      printOutAngle(transformed_cloud);
    }
    else
    {
      rotate_angle = theta*180/M_PI + rotate_angle;
      pcl::transformPointCloud (*transformed_cloud, *transformed_cloud, transform_2);
      std::cout<<"----transform_"<<n+1<<"_cloud----"<<std::endl;
      std::cout<<"rotate_angle: "<<rotate_angle<<std::endl;
      printOutAngle(transformed_cloud);
    }
  }



  _velodyne_header.frame_id = "velodyne";

  
  ros::Rate loop_rate(1);
  while(ros::ok())
  {
     
     publishCloud(&pub, cloud);
     publishCloud(&_pub, transformed_cloud);
     ros::spinOnce();
     loop_rate.sleep();
  }
  

  return (0);
}
