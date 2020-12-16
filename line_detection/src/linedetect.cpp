// Class linedetect's function definitions
#include "line_detection/linedetect.hpp"
#include <cstdlib>
#include <string>
#include "ros/console.h"

void LineDetect::imageCallback(const sensor_msgs::ImageConstPtr& msg) {
  cv_bridge::CvImagePtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    img = cv_ptr->image;
    cv::waitKey(30);
  }
  catch (cv_bridge::Exception& e) {
    ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
  }
}


cv::Mat LineDetect::Gauss(cv::Mat input) {
  cv::Mat output;
  // Applying Gaussian Filter
  cv::GaussianBlur(input, output, cv::Size(3, 3), 0.1, 0.1);
  return output;
}


int LineDetect::colorthresh(cv::Mat input) {
  // Initializaing variables
  cv::Size s = input.size();
  std::vector<std::vector<cv::Point> > v;
  int w = s.width;
  int h = s.height;
  float c_x = 0.0;

  // Detect all objects within the HSV range
  cv::cvtColor(input, LineDetect::img_hsv, CV_BGR2HSV);
  // red line boundary
  LineDetect::LowerYellow = {155, 60, 60};
  LineDetect::UpperYellow = {179, 255, 255};
  cv::inRange(LineDetect::img_hsv, LowerYellow, UpperYellow, LineDetect::img_mask);
  img_mask(cv::Rect(0, 0, w, 0.8*h)) = 0;  // black out the (x:0-w, y:0-0.8h) area. (0,0) is the top-left corner.

  // Find contours for better visualization
  // Each contour is stored as a vector of points (e.g. std::vector<std::vector<cv::Point> >).
  // CV_RETR_LIST: retrieves all of the contours without establishing any hierarchical relationships.
  // CV_CHAIN_APPROX_NONE: stores absolutely all the contour points. That is, any 2 subsequent points (x1,y1) and (x2,y2) 
  // of the contour will be either horizontal, vertical or diagonal neighbors, that is, max(abs(x1-x2),abs(y2-y1))==1.
  cv::findContours(LineDetect::img_mask, v, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

  // If contours exist add a bounding
  // Choosing contours with maximum area
  if (v.size() != 0) {
	int area = 0;
	int idx = 0;
	int count = 0;
	while (count < v.size()) {
	  if (area < v[count].size()) {
	     idx = count;
	     area = v[count].size();
	  }
	  count++;
  	}
  	
  	// boundingRect(): Calculates the up-right bounding rectangle of a point set or non-zero pixels of gray-scale image.
    // The function calculates and returns the minimal up-right bounding rectangle for the specified point 
    // set or non-zero pixels of gray-scale image.
  	cv::Rect rect = boundingRect(v[idx]);
  	cv::Point pt1, pt2, pt3;
  	pt1.x = rect.x;   
  	pt1.y = rect.y;
  	pt2.x = rect.x + rect.width;
  	pt2.y = rect.y + rect.height;
  	pt3.x = pt1.x+5;   // pt3 is where we put the text
  	pt3.y = pt1.y-5;
  	// Drawing the rectangle using points obtained
  	rectangle(input, pt1, pt2, CV_RGB(255, 0, 0), 2);
  	// Inserting text box
  	cv::putText(input, "Line Detected", pt3,
    CV_FONT_HERSHEY_COMPLEX, 1, CV_RGB(255, 0, 0));
  }
  
  // Mask image to limit the future turns affecting the output
  img_mask(cv::Rect(0.7*w, 0, 0.3*w, h)) = 0;
  img_mask(cv::Rect(0, 0, 0.3*w, h)) = 0;

  // Perform centroid detection of line
  // https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#moments
  cv::Moments M = cv::moments(LineDetect::img_mask);
  if (M.m00 > 0) {   // avoid dividing by zero
    cv::Point p1(M.m10/M.m00, M.m01/M.m00);  // p1(x= m10/m00, y= m01/m00) is the mass center.
    cv::circle(LineDetect::img_mask, p1, 5, cv::Scalar(155, 200, 0), -1);   //5, (155, 200, 0), -1
  }
  c_x = M.m10/M.m00;
  
  // Tolerance to chooise directions
  int tol = 15;
  int count = cv::countNonZero(img_mask);
  // Turn left if centroid is to the left of the image center minus tolerance
  // Turn right if centroid is to the right of the image center plus tolerance
  // Go straight if centroid is near image center
  if (c_x < w/2-tol) {
    LineDetect::dir = 0;   // turning left
  } else if (c_x > w/2+tol) {
    LineDetect::dir = 2;   // turnin right
  } else {
    LineDetect::dir = 1;   // going straight
  }

  // Search if no line detected, rotate 360 degree
  if (count == 0) {
    LineDetect::dir = 3;
  }
  

  // Output images viewed by the turtlebot
  cv::namedWindow("Rikirobot View");
  imshow("Rikirobot View", input);
  return LineDetect::dir;
}

