#include "ground_segmentation/ground_segmentation.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include "utils/timer.h"


std_msgs::Header _velodyne_header;
ros::Publisher _pub_nonground_cloud;
ros::Publisher _pub_image_cloud;
ros::Publisher _pub_ground_cloud;


using namespace std;
using namespace depth_clustering;
using time_utils::Timer;
using cv::Mat;
using cv::DataType;

const int _window_size =5;
const cv::Point ANCHOR_CENTER = cv::Point(-1, -1);
const int SAME_OUTPUT_TYPE = -1;
const  Radians _ground_remove_angle = 4_deg;
const bool seg_no_ground = true; 
const bool seg_ground = false;

void show_angle_data(const Mat& image)
{
   for (int c = 0; c < image.cols; ++c) 
  {
    for (int r = 0; r < image.rows; ++r) 
    {
      if(c > 302 && c < 307 && r < 30 && r > 27)
      {
        cout<<"["<<c<<"]"<<"["<<r<<"] "<<"angle"<<":"<<image.at<float>(r, c)<<endl;
      }
    }
  }

}

void show_depth_data(const Mat& image)
{
   for (int c = 0; c < image.cols; ++c) 
  {
    for (int r = 0; r < image.rows; ++r) 
    {
      if(c > 302 && c < 307 && r < 30 && r > 27)
      {
        cout<<"["<<c<<"]"<<"["<<r<<"] "<<"depth"<<":"<<image.at<float>(r, c)<<endl;
      }
    }
  }

}

Mat Colormap(const Mat& image)
{

  double min;
  double max;
  cv::minMaxIdx(image, &min, &max);
  cv::Mat adjMap;
  // expand your range to 0..255. Similar to histEq();
  image.convertTo(adjMap,CV_8UC1, 1080/ (max-min), -1080*min/(max-min)); 
  cv::Mat falseColorsMap;
  applyColorMap(adjMap, falseColorsMap, cv::COLORMAP_PARULA);


  return falseColorsMap;
}

Mat RepairDepth(const Mat& no_ground_image, int step,
                                    float depth_threshold) 
{
  Mat inpainted_depth = no_ground_image.clone();
  for (int c = 0; c < inpainted_depth.cols; ++c) 
  {
    for (int r = 0; r < inpainted_depth.rows; ++r) 
    {
      float& curr_depth = inpainted_depth.at<float>(r, c);
      if (curr_depth < 0.001f) 
      {
        int counter = 0;
        float sum = 0.0f;
        for (int i = 1; i < step; ++i) 
        {
          if (r - i < 0) 
          {
            continue;
          }
          for (int j = 1; j < step; ++j) 
          {
            if (r + j > inpainted_depth.rows - 1) 
            {
              continue;
            }
            const float& prev = inpainted_depth.at<float>(r - i, c);
            const float& next = inpainted_depth.at<float>(r + j, c);
            if (prev > 0.001f && next > 0.001f &&
                fabs(prev - next) < depth_threshold) 
            {
              sum += prev + next;
              counter += 2;
            }
          }
        }
        if (counter > 0) {
          curr_depth = sum / counter;
        }
      }
    }
  }
  return inpainted_depth;
}

Mat CreateAngleImage(const Mat& depth_image, const ProjectionParams& _params, const Mat& row_angle_image) 
{
  Mat angle_image = Mat::zeros(depth_image.size(), DataType<float>::type);
  Mat x_mat = Mat::zeros(depth_image.size(), DataType<float>::type);
  Mat y_mat = Mat::zeros(depth_image.size(), DataType<float>::type);
  const auto& sines_vec = _params.RowAngleSines();
  const auto& cosines_vec = _params.RowAngleCosines();

  float dx, dy;

  for (int r = 0; r < angle_image.rows; ++r) 
  {
    for (int c = 0; c < angle_image.cols; ++c) 
    {
      x_mat.at<float>(r, c) = depth_image.at<float>(r, c) * cos(row_angle_image.at<float>(r, c));
      y_mat.at<float>(r, c) = depth_image.at<float>(r, c) * sin(row_angle_image.at<float>(r, c));
    }
  }

  for (int r = 1; r < angle_image.rows; ++r) 
  {  
    for (int c = 0; c < angle_image.cols; ++c) 
    {
      dx = fabs(x_mat.at<float>(r, c) - x_mat.at<float>(r - 1, c));
      dy = fabs(y_mat.at<float>(r, c) - y_mat.at<float>(r - 1, c));
      if(depth_image.at<float>(r, c) < 0.001f| depth_image.at<float>(r-1, c) <0.001f) //| depth_image.at<float>(r-1, c) <0.001f
        angle_image.at<float>(r, c) = 0;
      else
      angle_image.at<float>(r, c) = atan2(dy, dx);
    }
  }
  // float dx, dy;
  // x_mat.row(0) = depth_image.row(0) * cosines_vec[0];
  // y_mat.row(0) = depth_image.row(0) * sines_vec[0];
  // for (int r = 1; r < angle_image.rows; ++r) 
  // {
  //   x_mat.row(r) = depth_image.row(r) * cosines_vec[r];
  //   y_mat.row(r) = depth_image.row(r) * sines_vec[r];

  //   for (int c = 0; c < angle_image.cols; ++c) 
  //   {
  //     dx = fabs(x_mat.at<float>(r, c) - x_mat.at<float>(r - 1, c));
  //     dy = fabs(y_mat.at<float>(r, c) - y_mat.at<float>(r - 1, c));
  //     if(depth_image.at<float>(r, c) < 0.001f| depth_image.at<float>(r-1, c) <0.001f) //| depth_image.at<float>(r-1, c) <0.001f
  //       angle_image.at<float>(r, c) = 0;
  //     else
  //     angle_image.at<float>(r, c) = atan2(dy, dx);
  //   }
  // }
  return angle_image;
}

void project_depth_cloud(const Mat& depth_image,
                         const ProjectionParams& _params, 
                         pcl::PointCloud<pcl::PointXYZ>::Ptr out_cloud_ptr, 
                         const Mat& row_angle_image,
                         const Mat& col_angle_image)
{
 
  Mat x_mat = Mat::zeros(depth_image.size(), DataType<float>::type);
  Mat y_mat = Mat::zeros(depth_image.size(), DataType<float>::type);
  Mat z_mat = Mat::zeros(depth_image.size(), DataType<float>::type);
  const auto& sines_vec = _params.RowAngleSines();
  const auto& cosines_vec = _params.RowAngleCosines();
  const auto& col_sines_vec = _params.ColAngleSines();
  const auto& col_cosines_vec = _params.ColAngleCosines();


  for (int r = 0; r < depth_image.rows; ++r) 
  {

    for (int c = 0; c < depth_image.cols; ++c) 
    {
      if(depth_image.at<float>(r, c) > 0.001f )
      {

        x_mat.at<float>(r,c) = depth_image.at<float>(r,c) * cos(row_angle_image.at<float>(r,c)) * cos(col_angle_image.at<float>(r,c));
        y_mat.at<float>(r,c) = depth_image.at<float>(r,c) * cos(row_angle_image.at<float>(r,c)) * sin(col_angle_image.at<float>(r,c));
        z_mat.at<float>(r,c) = depth_image.at<float>(r,c) * sin(row_angle_image.at<float>(r,c));

        pcl::PointXYZ current_point;

        current_point.x = x_mat.at<float>(r, c);
        current_point.y = y_mat.at<float>(r, c);
        current_point.z = z_mat.at<float>(r, c);
      
        out_cloud_ptr->points.push_back(current_point);
        
      }
    }
  }
}

void CalRowsAngle(const Mat& smoothed_image, float* angle_thresholds)
{

  angle_thresholds = new float[smoothed_image.cols];

  for(int c= 0; c < smoothed_image.cols; ++c)
  {
    float tmp = 3.14;
    for(int r=0; r < smoothed_image.rows; ++r)
    {
      if(tmp > fabs(smoothed_image.at<float>(r,c)) && fabs(smoothed_image.at<float>(r,c)) > 0.01)
        tmp = fabs(smoothed_image.at<float>(r,c));
    }
    // std::cout<<tmp<<std::endl;

    angle_thresholds[c] = tmp;
  }
}

Mat GetSavitskyGolayKernel(int window_size)
{
  if (window_size % 2 == 0) {
    throw std::logic_error("only odd window size allowed");
  }
  bool window_size_ok = window_size == 5 || window_size == 7 ||
                        window_size == 9 || window_size == 11;
  if (!window_size_ok) {
    throw std::logic_error("bad window size");
  }
  // below are no magic constants. See Savitsky-golay filter.
  Mat kernel;
  switch (window_size) {
    case 5:
      kernel = Mat::zeros(window_size, 1, CV_32F);
      kernel.at<float>(0, 0) = -3.0f;
      kernel.at<float>(0, 1) = 12.0f;
      kernel.at<float>(0, 2) = 17.0f;
      kernel.at<float>(0, 3) = 12.0f;
      kernel.at<float>(0, 4) = -3.0f;
      kernel /= 35.0f;
      return kernel;
    case 7:
      kernel = Mat::zeros(window_size, 1, CV_32F);
      kernel.at<float>(0, 0) = -2.0f;
      kernel.at<float>(0, 1) = 3.0f;
      kernel.at<float>(0, 2) = 6.0f;
      kernel.at<float>(0, 3) = 7.0f;
      kernel.at<float>(0, 4) = 6.0f;
      kernel.at<float>(0, 5) = 3.0f;
      kernel.at<float>(0, 6) = -2.0f;
      kernel /= 21.0f;
      return kernel;
    case 9:
      kernel = Mat::zeros(window_size, 1, CV_32F);
      kernel.at<float>(0, 0) = -21.0f;
      kernel.at<float>(0, 1) = 14.0f;
      kernel.at<float>(0, 2) = 39.0f;
      kernel.at<float>(0, 3) = 54.0f;
      kernel.at<float>(0, 4) = 59.0f;
      kernel.at<float>(0, 5) = 54.0f;
      kernel.at<float>(0, 6) = 39.0f;
      kernel.at<float>(0, 7) = 14.0f;
      kernel.at<float>(0, 8) = -21.0f;
      kernel /= 231.0f;
      return kernel;
    case 11:
      kernel = Mat::zeros(window_size, 1, CV_32F);
      kernel.at<float>(0, 0) = -36.0f;
      kernel.at<float>(0, 1) = 9.0f;
      kernel.at<float>(0, 2) = 44.0f;
      kernel.at<float>(0, 3) = 69.0f;
      kernel.at<float>(0, 4) = 84.0f;
      kernel.at<float>(0, 5) = 89.0f;
      kernel.at<float>(0, 6) = 84.0f;
      kernel.at<float>(0, 7) = 69.0f;
      kernel.at<float>(0, 8) = 44.0f;
      kernel.at<float>(0, 9) = 9.0f;
      kernel.at<float>(0, 10) = -36.0f;
      kernel /= 429.0f;
      return kernel;
  }
  return kernel;
}

Mat ApplySavitskyGolaySmoothing(const Mat& image,
                                int window_size) 
{
  Mat kernel = GetSavitskyGolayKernel(window_size);

  Mat smoothed_image;  // init an empty smoothed image
  cv::filter2D(image, smoothed_image, SAME_OUTPUT_TYPE, kernel, ANCHOR_CENTER,
               0, cv::BORDER_REFLECT101);
  return smoothed_image;
}

Mat GetUniformKernel(int window_size, int type)  {
  if (window_size % 2 == 0) {
    throw std::logic_error("only odd window size allowed");
  }
  Mat kernel = Mat::zeros(window_size, 1, type);
  kernel.at<float>(0, 0) = 1;
  kernel.at<float>(window_size - 1, 0) = 1;
  kernel /= 2.0;
  return kernel;
}

Mat ZeroOutGroundBFS(const cv::Mat& image,
                     const cv::Mat& angle_image,
                     const Radians& threshold,
                     int kernel_size,
                     const ProjectionParams& _params,
                     const bool _ground) 
{
  Mat res = cv::Mat::zeros(image.size(), CV_32F);
  LinearImageLabeler<> image_labeler(image, _params, threshold);
  SimpleDiff simple_diff_helper(&angle_image);
  Radians start_thresh = 30_deg;
  const auto& sines_vec = _params.RowAngleSines();
  int counter = 0;
  string name;
  for (int c = 0; c < image.cols; ++c) 
  {
    //start at bottom pixels and do bfs
    int r = image.rows - 1;
    while (r > 0 && image.at<float>(r, c) < 0.001f) 
    {
      --r;
    }

    auto current_coord = PixelCoord(r, c);
    uint16_t current_label = image_labeler.LabelAt(current_coord);
    if (current_label > 0)
    {
      // this coord was already labeled, skip
      continue;
    }
    // TODO(igor): this is a test. Maybe switch it on, maybe off.
    // std::cout<<"z:"<<image.at<float>(r, c)*sines_vec[r]<<std::endl;
    if (angle_image.at<float>(r, c) > start_thresh.val()) 
    {
      continue;
    }
    image_labeler.LabelOneComponent(1, current_coord, &simple_diff_helper);
  }

  auto label_image_ptr = image_labeler.GetLabelImage();
  if (label_image_ptr->rows != res.rows || label_image_ptr->cols != res.cols) 
  {
    fprintf(stderr, "ERROR: label image and res do not correspond.\n");
    return res;
  }
  kernel_size = std::max(kernel_size - 2, 3);
  Mat kernel = GetUniformKernel(kernel_size, CV_8U);
  Mat dilated = Mat::zeros(label_image_ptr->size(), label_image_ptr->type());
  cv::dilate(*label_image_ptr, dilated, kernel);
  Mat dilatedMap = Colormap(dilated);
  // cv::imwrite( "dilated.jpg", dilated );
  if(_ground == true)
  {
  for (int r = 0; r < dilated.rows; ++r) 
    {
    for (int c = 0; c < dilated.cols; ++c) 
      {
      if (dilated.at<uint16_t>(r, c) == 0) 
        {
        // all unlabeled points are non-ground
        res.at<float>(r, c) = image.at<float>(r, c);
        }
      }
    }
  }
  else
  {
    for (int r = 0; r < dilated.rows; ++r) 
    {
    for (int c = 0; c < dilated.cols; ++c) 
      {
      if (dilated.at<uint16_t>(r, c) != 0) 
        {
        // all unlabeled points are non-ground
        res.at<float>(r, c) = image.at<float>(r, c);
        }
      }
    }
  }
  return res;
}


Mat DepthIntCompare(const cv::Mat& depthimage, const cv::Mat& intimage)
{
  Mat res = cv::Mat::zeros(depthimage.size(), CV_32S);

  for (int r = 0; r < depthimage.rows; ++r) 
  {
    for (int c = 0; c < depthimage.cols; ++c) 
    {
      if (depthimage.at<int>(r, c) != 0) 
      {
        // all unlabeled points are non-ground
        res.at<int>(r, c) = intimage.at<int>(r, c);
      }
    }
  }

  return res;
}




template <class T>
T BytesTo(const vector<uint8_t>& data, uint32_t start_idx) 
{
  const size_t kNumberOfBytes = sizeof(T);
  uint8_t byte_array[kNumberOfBytes];
  // forward bit order (it is a HACK. We do not account for bigendianes)
  for (size_t i = 0; i < kNumberOfBytes; ++i) {
    byte_array[i] = data[start_idx + i];
  }
  T result;
  std::copy(reinterpret_cast<const uint8_t*>(&byte_array[0]),
            reinterpret_cast<const uint8_t*>(&byte_array[kNumberOfBytes]),
            reinterpret_cast<uint8_t*>(&result));
  return result;
}

void points_from_image(const Cloud& cloud,const Mat ground_image, pcl::PointCloud<pcl::PointXYZ>::Ptr out_cloud_ptr)
{
  for(int row = 0; row < ground_image.rows; ++row)
  {
    for(int col = 0; col < ground_image.cols; ++col)
    {
      if(ground_image.at<float>(row, col) < 0.001f)
      {
        continue;
      }
      
      const auto& point_container = cloud.projection_ptr()->at(row,col);
      if(point_container.IsEmpty())
      {
        continue;
      }
      for(const auto& point_idx : point_container.points())
      {
        const auto& point = cloud.points()[point_idx];
        pcl::PointXYZ current_point;

        current_point.x = point.x();
        current_point.y = point.y();
        current_point.z = point.z();
        // current_point.intensity = point.intensity();

        out_cloud_ptr->points.push_back(current_point);

      }
    }
  }
}


void publishCloud(const ros::Publisher* in_publisher, const pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud_to_publish_ptr)
{

  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(*in_cloud_to_publish_ptr, cloud_msg);
  cloud_msg.header=_velodyne_header;
  in_publisher->publish(cloud_msg);
}

int getMaxInt(const pcl::PointCloud<pcl::PointXYZI>::Ptr in_cloud_ptr)
{
  int maxInt = 0; 
  for(unsigned int i=0; i< in_cloud_ptr->points.size(); i++)
  {
    if(in_cloud_ptr->points[i].intensity > maxInt)
    maxInt = in_cloud_ptr->points[i].intensity;
  }

  return maxInt;
}

int Otsu(const pcl::PointCloud<pcl::PointXYZI>::Ptr in_cloud_ptr, const int maxInt)
{

  // pcl::PointXYZI min;//用于存放三个轴的最小值
  // pcl::PointXYZI max;//用于存放三个轴的最大值
  // pcl::getMinMax3D(*in_cloud_ptr,min,max);

  float wB = 0,wF = 0,mB = 0,mF = 0,varMax = 0,sum =0;
  int num = in_cloud_ptr->points.size();
  int num_B = 0,num_F = 0, threshold = 0;
  float sum_B = 0,sum_F = 0;
  int num_int[maxInt+1] = {0};
  int index =0;

  for(unsigned int i=0; i< in_cloud_ptr->points.size(); i++)
  {
    index = (int)in_cloud_ptr->points[i].intensity;
    num_int[index] = num_int[index] +1;
    // std::cout<<num_int[i]<<std::endl;
  }

  // std::cout<<num_int[10]<<std::endl;

  // for(int k=0; k<=maxInt;k++)
  // {
  //   std::cout<<"intensity:"<<k<<":"<<num_int[k]<<std::endl;
  // }


  for(int t=0; t<=maxInt;t++)
  {
    sum += t*num_int[t];
  }



  for(int i=0; i<=maxInt;i++)
  {
    num_B += num_int[i];
    sum_B += i*num_int[i];
  
    num_F = num - num_B;
    sum_F = sum - sum_F;

   float mB = sum_B / num_B;    // Mean Background
   float mF = sum_F / num_F;    // Mean Foreground

   wB = (float)num_B / num;
   wF = (float)num_F / num;

   // std::cout<<num<<std::endl;
   // std::cout<<wB<<std::endl;

   float varBetween = (float)wB * (float)wF * (mB - mF) * (mB - mF);
   // std::cout<<varBetween<<std::endl;

   if (varBetween > varMax) 
   {
      varMax = varBetween;
      threshold = i;
    }
  }

  // std::cout<<varMax<<std::endl;

  return threshold;
}



void velodyne_callback(const sensor_msgs::PointCloud2ConstPtr& msg)

{
  pcl::PointCloud<pcl::PointXYZ>::Ptr ground_cloud_ptr (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr non_ground_cloud_ptr (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr project_depth_cloud_ptr (new pcl::PointCloud<pcl::PointXYZ>);


  _velodyne_header = msg->header;
  const clock_t begin_time = clock();

  double t = double(_velodyne_header.stamp.sec) + double(_velodyne_header.stamp.nsec)*1e-9;
  // std::cout<<_velodyne_header.stamp<<std::endl;
    // std::cout<<"hello world"<<std::endl;

  auto params = ProjectionParams();
  params.SetSpan(SpanParams(-180_deg, 180_deg, 870),
                 SpanParams::Direction::HORIZONTAL);
  params.SetSpan(SpanParams(10.0_deg, -30.0_deg, 32),
                 SpanParams::Direction::VERTICAL);
  params.FillCosSin();

  if (!params.valid()) 
    fprintf(stderr, "ERROR: params are not valid!\n");

  Cloud cloud;
  
  uint32_t x_offset = msg->fields[0].offset;
  uint32_t y_offset = msg->fields[1].offset;
  uint32_t z_offset = msg->fields[2].offset;
  // uint32_t intensity_offset = msg->fields[3].offset;
  // uint32_t ring_offset = msg->fields[4].offset;


   for (uint32_t point_start_byte = 0, counter = 0;
        point_start_byte < msg->data.size();
        point_start_byte += msg->point_step, ++counter) 
   {
    RichPoint point;
    point.x() = BytesTo<float>(msg->data, point_start_byte + x_offset);
    point.y() = BytesTo<float>(msg->data, point_start_byte + y_offset);
    point.z() = BytesTo<float>(msg->data, point_start_byte + z_offset);
    // point.intensity() = BytesTo<float>(msg->data, point_start_byte + intensity_offset);
    // point.ring() = BytesTo<uint16_t>(msg->data, point_start_byte + ring_offset);

    // point.z *= -1;  // hack
    if((point.x()*point.x() + point.y()*point.y() + point.z()*point.z()) >4)
    cloud.push_back(point);

   }

   cloud.InitProjection(params);

  int min_cluster_size = 20;
  int max_cluster_size = 100000;


  cv::Mat depth_image_unfixed = cloud.projection_ptr()->depth_image();
  const cv::Mat& depth_image = RepairDepth(cloud.projection_ptr()->depth_image(), 5, 1.0f);
  cv::Mat row_angle_image = cloud.projection_ptr()->row_angle_image();
  cv::Mat col_angle_image = cloud.projection_ptr()->col_angle_image();
  

  cv::Mat unfixed,depthMap,intMap,nogrdMap,grdintMap,grdMap,angleMap,smoothangleMap;
 

  auto angle_image = CreateAngleImage(depth_image, params, row_angle_image);
  // show_angle_data(angle_image);
  // show_depth_data(depth_image);
  project_depth_cloud(depth_image, params, project_depth_cloud_ptr, row_angle_image, col_angle_image);
 
  auto no_ground_image = ZeroOutGroundBFS(depth_image, angle_image,
_ground_remove_angle, _window_size, params, seg_no_ground);

  auto ground_image = ZeroOutGroundBFS(depth_image, angle_image,
_ground_remove_angle, _window_size, params, seg_ground);


  //create 3d points from image
  points_from_image(cloud, ground_image, ground_cloud_ptr);
  points_from_image(cloud, no_ground_image, non_ground_cloud_ptr);

  
  publishCloud(&_pub_ground_cloud, ground_cloud_ptr);
  publishCloud(&_pub_nonground_cloud, non_ground_cloud_ptr);
  publishCloud(&_pub_image_cloud, project_depth_cloud_ptr);

}
int main(int argc, char **argv)
{

  ros::init(argc, argv, "ground_segmentation");
  ros::NodeHandle nh;

  ros::Subscriber sub = nh.subscribe ("/velodyne_points", 1, velodyne_callback);

  
  _pub_nonground_cloud = nh.advertise<sensor_msgs::PointCloud2>("/nonground_points",1);
  _pub_ground_cloud = nh.advertise<sensor_msgs::PointCloud2>("/ground_points",1);
  _pub_image_cloud = nh.advertise<sensor_msgs::PointCloud2>("/image_points",1);

  ros::spin();
}