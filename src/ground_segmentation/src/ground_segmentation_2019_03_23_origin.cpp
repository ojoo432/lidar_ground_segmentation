#include "ground_segmentation/ground_segmentation.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include "utils/timer.h"
#include <Eigen/Dense>
#include <Eigen/Geometry>


std_msgs::Header _velodyne_header;
ros::Publisher _pub_nonground_cloud;
ros::Publisher _pub_image_cloud;
ros::Publisher _pub_row_cloud;
ros::Publisher _pub_ground_cloud;
ros::Publisher _pub_project_sample_points;

using namespace Eigen;
using namespace std;
using namespace depth_clustering;
using time_utils::Timer;
using cv::Mat;
using cv::DataType;

const int _window_size =5;
const cv::Point ANCHOR_CENTER = cv::Point(-1, -1);
const int SAME_OUTPUT_TYPE = -1;
const  Radians _ground_remove_angle = 3_deg;
const bool seg_no_ground = true; 
const bool seg_ground = false;

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

Mat CreateAngleImage(const Mat& depth_image, const ProjectionParams& _params) 
{
  Mat angle_image = Mat::zeros(depth_image.size(), DataType<float>::type);
  Mat x_mat = Mat::zeros(depth_image.size(), DataType<float>::type);
  Mat y_mat = Mat::zeros(depth_image.size(), DataType<float>::type);
  const auto& sines_vec = _params.RowAngleSines();
  const auto& cosines_vec = _params.RowAngleCosines();
  float dx, dy;
  x_mat.row(0) = depth_image.row(0) * cosines_vec[0];
  y_mat.row(0) = depth_image.row(0) * sines_vec[0];
  for (int r = 1; r < angle_image.rows; ++r) 
  {
    x_mat.row(r) = depth_image.row(r) * cosines_vec[r];
    y_mat.row(r) = depth_image.row(r) * sines_vec[r];

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
  return angle_image;
}


Vector3f calnormal(Vector3f p1, Vector3f p2, Vector3f p3)
{
  Vector3f v1 = p2-p1;
  Vector3f v2 = p3-p1;

  Vector3f normal = v1.cross(v2);

  return normal;
}


void project_sample_point_cloud(const Mat& depth_image,const ProjectionParams& _params, pcl::PointCloud<pcl::PointXYZI>::Ptr out_cloud_ptr)
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

        x_mat.at<float>(r,c) = depth_image.at<float>(r,c) * cosines_vec[r] * col_cosines_vec[c];
        y_mat.at<float>(r,c) = depth_image.at<float>(r,c) * cosines_vec[r] * col_sines_vec[c];
        z_mat.at<float>(r,c) = depth_image.at<float>(r,c) * sines_vec[r];

        pcl::PointXYZI current_point;

        current_point.x = x_mat.at<float>(r, c);
        current_point.y = y_mat.at<float>(r, c);
        current_point.z = z_mat.at<float>(r, c);
        
    
        out_cloud_ptr->points.push_back(current_point);

      }
    }
  }
}

bool check_start_ground_point(const Mat& depth_image,
                              const Mat& row_angle_image,
                              const Mat& col_angle_image,
                              const int r,
                              const int c)
{
  if(c == 869)
    return true;
  else
  {
  Vector3f p1(depth_image.at<float>(r,c) * cos(row_angle_image.at<float>(r,c)) * cos(col_angle_image.at<float>(r,c)),
              depth_image.at<float>(r,c) * cos(row_angle_image.at<float>(r,c)) * sin(col_angle_image.at<float>(r,c)),
              depth_image.at<float>(r,c) * sin(row_angle_image.at<float>(r,c)));

  Vector3f p2(depth_image.at<float>(r-1,c) * cos(row_angle_image.at<float>(r-1,c)) * cos(col_angle_image.at<float>(r-1,c)),
              depth_image.at<float>(r-1,c) * cos(row_angle_image.at<float>(r-1,c)) * sin(col_angle_image.at<float>(r-1,c)),
              depth_image.at<float>(r-1,c) * sin(row_angle_image.at<float>(r-1,c)));

  Vector3f p3(depth_image.at<float>(r,c+1) * cos(row_angle_image.at<float>(r,c+1)) * cos(col_angle_image.at<float>(r,c+1)),
              depth_image.at<float>(r,c+1) * cos(row_angle_image.at<float>(r,c+1)) * sin(col_angle_image.at<float>(r,c+1)),
              depth_image.at<float>(r,c+1) * sin(row_angle_image.at<float>(r,c+1)));

  Vector3f v = calnormal(p1, p2, p3);
  Vector3f v_opposite(-v(0), -v(1), -v(2));
  Vector3f normal(0,0,1);

  float angle;
  float angle_opposite;

  angle = acos(v.dot(normal)/(v.norm()*normal.norm()));
  angle = angle/M_PI*180;

  angle_opposite = acos(v_opposite.dot(normal)/(v_opposite.norm()*normal.norm()));
  angle_opposite = angle_opposite/M_PI*180;
  // cout<<"row:"<<r<<"column:"<<c<<"angle:"<<angle<<endl;

  if(r == 30 && c == 414)
    cout<<"normal angle:"<<angle<<endl;
  if(angle < 35 )
    return false;
  else
    return true;

  }
}

bool check_continuous_point(const Mat& angle_image,
                            const int r,
                            const int c,
                            const int number,
                            const float angle_threshold)
{

  float angle;

  for(int i=0; i<number; i++)
  {
    angle =0;
    angle = fabs(angle_image.at<float>(r-i, c) - angle_image.at<float>(r-i-1, c));


    // cout<<"row: "<<r<<"col: "<<c<<"angle_diff: "<<angle<<endl;

    if(angle > angle_threshold || angle_image.at<float>(r-i, c) == 0 ||  angle_image.at<float>(r-i, c) >0.8)
      return false;
  }

  return true;
}

void show_row_cloud(const Cloud& cloud, pcl::PointCloud<pcl::PointXYZI>::Ptr out_cloud_ptr)
{

  int col = 420;
  
  for(int row = 0; row < 31; ++row)
    {
      
      const auto& point_container = cloud.projection_ptr()->at(row,col);
      if(point_container.IsEmpty())
      {
       
        continue;
      }
      for(const auto& point_idx : point_container.points())
      {
        const auto& point = cloud.points()[point_idx];
        pcl::PointXYZI current_point;

        current_point.x = point.x();
        current_point.y = point.y();
        current_point.z = point.z();
        current_point.intensity = point.intensity();
     
        out_cloud_ptr->points.push_back(current_point);

      }
    }
}

void show_row_angle(const cv::Mat& angle_image)
{

  int col = 420;

  for(int row = 0; row < 31; ++row)
    {
      cout<<"row:"<<row<<"   angle:"<<angle_image.at<float>(row, col)<<endl;
    }
}


vector<int> cal_ground_index(const cv::Mat& angle_image, const int col)
{
  float angle_threshold = 0.0523;
  int n = 3;

  int index = 0;
  int row = 0;
  bool search_ground_index = true;
  vector<int> _ground_index;

  for(int row = angle_image.rows-1; row > 2; --row)
  {
    index = 0;

    if(search_ground_index && check_continuous_point(angle_image, row, col, n, angle_threshold))
    {
      index = row;
      _ground_index.push_back(index);
      if(col == 528)
      // cout<<"row: "<<row<<"push_back_success"<<endl;
      search_ground_index = false;
    }
    else if(!check_continuous_point(angle_image, row, col, n, angle_threshold))
      search_ground_index = true;
          // cout<<"row: "<<row<<"push_back_fail"<<endl;
  }

      return _ground_index;
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
      kernel = Mat::zeros(1, window_size, CV_32F);
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
                     const cv::Mat& cal_image,
                     const Radians& threshold,
                     int kernel_size,
                     const ProjectionParams& _params,
                     const bool _ground,
                     const cv::Mat& row_angle_image,
                     const cv::Mat& col_angle_image
                     ) 
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
    
    int r = image.rows - 1;
    while ((r > 0 && image.at<float>(r, c) < 0.001f)) 
    {
      --r;
    }

    auto current_coord = PixelCoord(r, c);
    uint16_t current_label = image_labeler.LabelAt(current_coord);
    // if (current_label > 0)
    // {
    //   cout<<"check_label_skip"<<endl;
    //   // this coord was already labeled, skip
    //   continue;
    // }
    // TODO(igor): this is a test. Maybe switch it on, maybe off.
    // std::cout<<"z:"<<image.at<float>(r, c)*sines_vec[r]<<std::endl;
    // if (angle_image.at<float>(r, c) > start_thresh.val() || image.at<float>(r, c)*sin(row_angle_image.at<float>(r, c)) > -2.1) 
    if (angle_image.at<float>(r, c) > start_thresh.val())
    {
      if(c == 397)
      cout<<"check_angle_skip"<<endl;
      continue;
    }

    if((r-1) < 0)
    {
      if(c == 397)
      cout<<"check_row_skip"<<endl;
      continue;
    }
    else
    {
      if(image.at<float>(r-1,c) < 0.001f)
      {
         if(c == 397)
        cout<<"check_depth_skip"<<endl;
        continue;
      }
    }

    image_labeler.LabelOneComponent(1, current_coord, &simple_diff_helper);

  }

  for (int c = 0; c < image.cols; ++c) 
  {

  vector<int> ground_index;

    ground_index = cal_ground_index(cal_image, c);

    for(int i =0; i < ground_index.size(); i++)
    {
      auto ground_coord = PixelCoord(ground_index[i], c);
      uint16_t ground_label = image_labeler.LabelAt(ground_coord);

      if (ground_label == 0 && 
          angle_image.at<float>(ground_index[i], c) < start_thresh.val() &&
          ground_index[i]-1 >= 0 &&
          image.at<float>(ground_index[i]-1,c) >= 0.001f
          )
      {
        //&&  !(check_start_ground_point(image, row_angle_image, col_angle_image, ground_index[i], c))
        image_labeler.LabelOneComponent(1, ground_coord, &simple_diff_helper);
      }
    }
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

void points_from_image(const Cloud& cloud,const Mat ground_image, pcl::PointCloud<pcl::PointXYZI>::Ptr out_cloud_ptr)
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
        pcl::PointXYZI current_point;

        current_point.x = point.x();
        current_point.y = point.y();
        current_point.z = point.z();
        current_point.intensity = point.intensity();
     
        out_cloud_ptr->points.push_back(current_point);

      }
    }
  }
}



void publishCloud(const ros::Publisher* in_publisher, const pcl::PointCloud<pcl::PointXYZI>::Ptr in_cloud_to_publish_ptr)
{

  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(*in_cloud_to_publish_ptr, cloud_msg);
  cloud_msg.header=_velodyne_header;
  in_publisher->publish(cloud_msg);
}


void velodyne_callback(const sensor_msgs::PointCloud2ConstPtr& msg)

{
  pcl::PointCloud<pcl::PointXYZI>::Ptr ground_cloud_ptr (new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr non_ground_cloud_ptr (new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr sample_cloud_ptr (new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr project_sample_cloud_ptr (new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr row_cloud_ptr (new pcl::PointCloud<pcl::PointXYZI>);

  int start_index[870] = {0};
  int g_index[870] = {0};


  _velodyne_header = msg->header;
  const clock_t begin_time = clock();

  double t = double(_velodyne_header.stamp.sec) + double(_velodyne_header.stamp.nsec)*1e-9;
  std::cout<<_velodyne_header.stamp<<std::endl;
    // std::cout<<"hello world"<<std::endl;

  auto params = ProjectionParams();
  params.SetSpan(SpanParams(-180_deg, 180_deg, 870),
                 SpanParams::Direction::HORIZONTAL);
  params.SetSpan(SpanParams(10.0_deg, -30.0_deg, 31),
                 SpanParams::Direction::VERTICAL);
  params.FillCosSin();

  if (!params.valid()) 
    fprintf(stderr, "ERROR: params are not valid!\n");

  Cloud cloud;
  
  uint32_t x_offset = msg->fields[0].offset;
  uint32_t y_offset = msg->fields[1].offset;
  uint32_t z_offset = msg->fields[2].offset;
  uint32_t intensity_offset = msg->fields[3].offset;
  uint32_t ring_offset = msg->fields[4].offset;


   for (uint32_t point_start_byte = 0, counter = 0;
        point_start_byte < msg->data.size();
        point_start_byte += msg->point_step, ++counter) 
   {
    RichPoint point;
    point.x() = BytesTo<float>(msg->data, point_start_byte + x_offset);
    point.y() = BytesTo<float>(msg->data, point_start_byte + y_offset);
    point.z() = BytesTo<float>(msg->data, point_start_byte + z_offset);
    point.intensity() = BytesTo<float>(msg->data, point_start_byte + intensity_offset);
    point.ring() = BytesTo<uint16_t>(msg->data, point_start_byte + ring_offset);

    if((point.x()*point.x() + point.y()*point.y() + point.z()*point.z()) >4)
    cloud.push_back(point);

   }

   cloud.InitProjection(params);

 
  const cv::Mat& depth_image = RepairDepth(cloud.projection_ptr()->depth_image(), 5, 1.0f);
  cv::Mat intensity_image = cloud.projection_ptr()->intensity_image();
  cv::Mat row_angle_image = cloud.projection_ptr()->row_angle_image();
  cv::Mat col_angle_image = cloud.projection_ptr()->col_angle_image();

 


  cv::Mat angleMap,smoothangleMap,anglePointMap;


  auto angle_image = CreateAngleImage(depth_image, params);

  auto smoothed_image = ApplySavitskyGolaySmoothing(angle_image, _window_size);
  project_sample_point_cloud(depth_image, params, project_sample_cloud_ptr);


  angleMap = Colormap(angle_image);
  smoothangleMap = Colormap(smoothed_image);

  // cv::imwrite( "project_angle.jpg", angleMap );

  auto no_ground_image = ZeroOutGroundBFS(depth_image, smoothed_image, angle_image,
_ground_remove_angle, _window_size, params, seg_no_ground, row_angle_image, col_angle_image);


  auto ground_image = ZeroOutGroundBFS(depth_image, smoothed_image, angle_image,
_ground_remove_angle, _window_size, params, seg_ground, row_angle_image, col_angle_image);


  auto ground_int_image = DepthIntCompare(ground_image, intensity_image);


  //create 3d points from image
  points_from_image(cloud, ground_image, ground_cloud_ptr);
  points_from_image(cloud, no_ground_image, non_ground_cloud_ptr);
  show_row_cloud(cloud, row_cloud_ptr);
  cout<<"-----angle_image-----"<<endl;
  show_row_angle(angle_image);
  cout<<"-----angle_image_point-----"<<endl;
  show_row_angle(smoothed_image);
  // cout<<"-----depth_image-----"<<endl;
  // show_row_angle(depth_image);

  // vector<int> ground_index;

  // ground_index = cal_ground_index(smoothed_image, 429);

  // cout <<"ground_index: "<< ground_index.size() << ' ';



  // cv::namedWindow("Image_Angle", CV_WINDOW_NORMAL);
  // cv::imshow("Image_Angle",angleMap);

  // cv::waitKey(0);


  publishCloud(&_pub_ground_cloud, ground_cloud_ptr);
  publishCloud(&_pub_nonground_cloud, non_ground_cloud_ptr);
  publishCloud(&_pub_project_sample_points, project_sample_cloud_ptr);
  publishCloud(&_pub_row_cloud, row_cloud_ptr);

  // std::cout << double( clock () - begin_time)/CLOCKS_PER_SEC<<std::endl;

  // std::cout<<threshold<<std::endl;
}
int main(int argc, char **argv)
{

  ros::init(argc, argv, "ground_segmentation");
  ros::NodeHandle nh;

  ros::Subscriber sub = nh.subscribe ("/slope", 1, velodyne_callback);

  
  _pub_nonground_cloud = nh.advertise<sensor_msgs::PointCloud2>("/nonground_points",1);
  _pub_ground_cloud = nh.advertise<sensor_msgs::PointCloud2>("/ground_points",1);
  _pub_project_sample_points = nh.advertise<sensor_msgs::PointCloud2>("/project_sample_points",1);
  _pub_row_cloud = nh.advertise<sensor_msgs::PointCloud2>("/row_points",1);

  ros::spin();
}