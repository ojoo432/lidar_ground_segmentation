#ifndef GROUND_SEGMENTATION_H
#define GROUND_SEGMENTATION_H

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
#include <string>

#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <utils/rich_point.h>
#include "utils/cloud.h"
#include "utils/radians.h"
#include "image_labelers/linear_image_labeler.h"
#include "diff_helpers/angle_diff.h"
#include "diff_helpers/simple_diff.h"
#include "communication/abstract_sender.h"
#include "time.h"

#include <pcl/common/common.h>

void velodyne_callback();


#endif