#include "line_detection/linedetect.hpp"

int main(int argc, char **argv) {
    // Initializing node and object
    ros::init(argc, argv, "detection");
    ros::NodeHandle nh;
    LineDetect det;

    // Creating Publisher and subscriber
    /*
    &det is a tracked_object: A shared pointer to an object to track for these callbacks.  
    If set, a weak_ptr will be created to this object, and if the reference count goes to 0,
    the subscriber callbacks will not get called.
 	
 	Note that setting this will cause a new reference to be added to the object before the
 	callback, and for it to go out of scope (and potentially be deleted) in the code path (and therefore
 	thread) that the callback is invoked from.
    */
    ros::Subscriber sub = nh.subscribe("/camera/rgb/image_raw", 1, &LineDetect::imageCallback, &det);

    // ros::Publisher dirPub = n.advertise<riki_line_follower::pos>("direction", 1);
    
    // riki_line_follower::pos msg;


    /*
    By default roscpp will install a SIGINT handler which provides Ctrl-C handling which will 
    cause ros::ok() to return false if that happens.

	ros::ok() will return false if:
		- a SIGINT is received (Ctrl-C)
		- we have been kicked off the network by another node with the same name.
		- ros::shutdown() has been called by another part of the application.
		- all ros::NodeHandles have been destroyed.
    */

    int direction;
    while (ros::ok()) {
        if (!det.img.empty()) {
            // Perform image processing
            det.img_filt = det.Gauss(det.img);
            direction = det.colorthresh(det.img_filt);
            // Publish direction message
            // dirPub.publish(msg);
            }
        ros::spinOnce();
    }
    // Closing image viewer
    cv::destroyWindow("Turtlebot View");
}
