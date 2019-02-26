/*
The MIT License
Copyright (c) 2015 Avi Singh
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <ros/ros.h>
#include <ros/package.h>
#include <std_msgs/String.h>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <cv_bridge/cv_bridge.h>

#include <iostream>
#include <ctype.h>
#include <algorithm> // for copy
#include <iterator> // for ostream_iterator
#include <vector>
#include <ctime>
#include <sstream>
#include <fstream>
#include <string>


int start_flag = 0;


using namespace cv;
using namespace std;

Mat prevImage,currImage;
Mat traj = Mat::zeros(600, 600, CV_8UC3);
Mat R_f = (Mat_<double>(3,3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
Mat t_f = Mat::zeros(3, 1, CV_64F);

void featureTracking(Mat img_1, Mat img_2, vector<Point2f>& points1, vector<Point2f>& points2, vector<uchar>& status) 
{ 

//this function automatically gets rid of points for which tracking fails

  vector<float> err;          
  Size winSize=Size(21,21);                                               
  TermCriteria termcrit=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01);

  calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);

  //getting rid of points for which the KLT tracking failed or those who have gone outside the frame
  int indexCorrection = 0;
  for( int i=0; i<status.size(); i++)
     {  Point2f pt = points2.at(i- indexCorrection);
      if ((status.at(i) == 0)||(pt.x<0)||(pt.y<0))  {
          if((pt.x<0)||(pt.y<0))  {
            status.at(i) = 0;
          }
          points1.erase (points1.begin() + (i - indexCorrection));
          points2.erase (points2.begin() + (i - indexCorrection));
          indexCorrection++;
          }
      }

}


void featureDetection(Mat img_1, vector<Point2f>& points1)  
{  
 //uses FAST as of now, modify parameters as necessary
  vector<KeyPoint> keypoints_1;
  int fast_threshold = 20;
  bool nonmaxSuppression = true;
  FAST(img_1, keypoints_1, fast_threshold, nonmaxSuppression);
  KeyPoint::convert(keypoints_1, points1, vector<int>());
}


void image_callback(const sensor_msgs::ImageConstPtr& msg)
{
  if(start_flag == 0)
  {
    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);

    currImage = cv_ptr->image;
    start_flag = 1;
  }
  else
  {
    prevImage = currImage.clone();

    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    currImage = cv_ptr->image;

   //the final rotation and tranlation vectors containing the 
  
    char text[100];
    int fontFace = FONT_HERSHEY_PLAIN;
    double fontScale = 1;
    int thickness = 1;  
    cv::Point textOrg(10, 50);
  
 
    if ( !prevImage.data || !currImage.data ) 
    { 
      std::cout<< " --(!) Error reading images " << std::endl;

    }

    Mat prevGray,currGray;
  
    // we work with grayscale images
    cvtColor(prevImage, prevGray, COLOR_BGR2GRAY);
    cvtColor(currImage, currGray, COLOR_BGR2GRAY);
  
    // feature detection, tracking
    vector<Point2f> prevFP, currFP;        //vectors to store the coordinates of the feature points
    featureDetection(prevGray, prevFP);        //detect features in img_1
    vector<uchar> status;
    featureTracking(prevGray,currGray,prevFP,currFP, status); //track those features to img_2
  
    //TODO: add a fucntion to load these values directly from KITTI's calib files
    // WARNING: different sequences in the KITTI VO dataset have different intrinsic/extrinsic parameters
    double focal = 690.6878510284205;
    cv::Point2d pp(633.4759793568744, 379.0474946457819);
    //recovering the pose and the essential matrix
    Mat E, R, t, mask;
    E = findEssentialMat(currFP, prevFP, focal, pp, RANSAC, 0.999, 1.0, mask);
    recoverPose(E, currFP, prevFP, R, t, focal, pp, mask);
  
    
  
    t_f = t_f + 0.5*(R_f*t);
    R_f = R*R_f;
  
    namedWindow( "Road facing camera", WINDOW_AUTOSIZE );// Create a window for display.
    // namedWindow( "Trajectory", WINDOW_AUTOSIZE );// Create a window for display.
  

  
    
    cout<<"Rotation: "<<R_f<<endl<<endl;
    cout<<"translation: "<<t_f<<endl<<endl;
    int x = int(t_f.at<double>(0)) + 300;
    int y = int(t_f.at<double>(2)) + 100;
    circle(traj, Point(x, y) ,1, CV_RGB(255,0,0), 2);
  
    rectangle( traj, Point(10, 30), Point(550, 50), CV_RGB(0,0,0), CV_FILLED);
    sprintf(text, "Coordinates: x = %02fm y = %02fm z = %02fm", t_f.at<double>(0), t_f.at<double>(1), t_f.at<double>(2));
    putText(traj, text, textOrg, fontFace, fontScale, Scalar::all(255), thickness, 8);
  
    imshow( "Road facing camera", currImage);
    imshow( "Trajectory", traj );
  
    waitKey(30);
  }

}


int main( int argc, char** argv ) 
{

  ros::init(argc, argv, "visual_odometry");
  ros::NodeHandle nh;

  ros::Subscriber sub = nh.subscribe ("/zed/left/image_raw_color", 1, image_callback);

  ros::spin();
}