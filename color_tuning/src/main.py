#!/usr/bin/env python
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2
import numpy as np
import rospy # Python library for ROS
from sensor_msgs.msg import Image as ImageMsg # Using ImageMsg to avoid name conflict with PIL's Image
from cv_bridge import CvBridge, CvBridgeError # Package to convert between ROS and OpenCV Images

class CameraSubscriber(object):
    def __init__(self):
        self.receive_message()
        
    def receive_message(self):
        # Tells rospy the name of the node.
        rospy.init_node('camera_subscriber', anonymous=True)
        
        # Subscribing to the video_frames topic
        rospy.Subscriber('/camera/rgb/image_raw', ImageMsg, self.callback)
        
        # spin() simply keeps python from exiting until this node is stopped
        rospy.spin()

        # Close down the video stream when done
        cv2.destroyAllWindows()

    def callback(self, msg):
        try:
            br = CvBridge()

            # Output debugging information to the terminal
            rospy.loginfo("receiving video frame..............")

            # Convert ROS Image message to OpenCV image
            current_frame = br.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Do object detection
            color_tuning(current_frame)

            # Display image
            # cv2.imshow("raspi_cam", processed_frame)
            # cv2.waitKey(1)

        except CvBridgeError as e:
            print(e)

###############################################################################
################   PART 1   IMAGES PLAY AROUD   ###############################
###############################################################################
"""
# display image
img = cv2.imread("mpc.JPG")
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (7,7), 0)
imgCanny = cv2.Canny(img, 100, 100)

kernel = np.ones((5,5), np.uint8)
# kernel: structuring element used for dilation
# iterations: number of times dilation is applied.
imgDilation = cv2.dilate(imgCanny, kernel, iterations=1)
imgEroded = cv2.erode(imgDilation, kernel, iterations=2)

# Resize the image
imgResize = cv2.resize(img, (640, 480))

# crop the image
# for cropping the image, it is [height, width]
imgCropped = img[0:200, 300:500]


cv2.imshow('imgGray', imgGray)
cv2.imshow('imgBlur', imgBlur)
cv2.imshow('imgCanny', imgCanny)
cv2.imshow('imgDilation', imgDilation)
cv2.imshow('imgEroded', imgEroded)
cv2.imshow("resize", imgResize)
cv2.imshow("crop", imgCropped)
cv2.waitKey(0)
"""



###############################################################################
##############   PART 2   VIDEO READ AND WRITE   ##############################
###############################################################################
"""
# capture video
# The first argument of "cv2.VideoCapture" can be a video name or 0 which is the default for web cam.
# If you have more than 1 camera, you can use 1 for the second camera, 2 for third, etc
cap = cv2.VideoCapture(0)

# set the captured video
cap.set(3, 640)   # id num 3 is the width
cap.set(4, 480)   # id num 4 is the height
cap.set(10, 500)  # id num 10 is the brightness

# write to a video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))

while cap.isOpened():
    success, img = cap.read()

    if success == True:
        # using .get() to get video information. The properties can be found here:
        # https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html?
        print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # write to a video
        output.write(img)

        # change the image space from BGR to Gray
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow("video", img)

        # for display the video
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
output.release()
cap.release()
"""


###############################################################################
####################   PART 3   SHAPES AND TEXTS   ############################
###############################################################################
"""
img = np.zeros((512,512,3), np.uint8)

# Create lines on image
# (0,0) is the starting point, (200,300) is the ending point (height, width),
# (0,255,0) is the color of the line, 3 is the thickness.
cv2.line(img, (0,0), (200,300), (0,255,0), 3)

# Create rectangle on image
# (0,0) is the upper left corner, (250,350) is the lower right corner
cv2.rectangle(img, (0,0), (250,350), (0,0,255), 3)

# Create circle on image
# (400,50) is the center of the circle, 30 is the radius
cv2.circle(img, (400,50), 30, (255,255,0), 3)

# Put text on the image
cv2.putText(img, " openCV ", (250,250), cv2.FONT_HERSHEY_DUPLEX, 1, (0,150,0),1)

cv2.imshow("new", img)
cv2.waitKey(0)
"""



###############################################################################
###################   PART 4   WARP PERSPECTIVE   #############################
###############################################################################
"""
img = cv2.imread("test1.jpg")

# Define 4 source points src = np.float32([[,],[,],[,],[,]])
src = np.float32([[211, 719], [575, 460], [710, 460], [1117, 719]])

# Define 4 destination points dst = np.float32([[,],[,],[,],[,]])
dst = np.float32([[250, img.shape[0]], [250, 0], [img.shape[1] - 330, 0],
                  [img.shape[1] - 330, img.shape[0]]])

# M : transform matrix (3,3)
M = cv2.getPerspectiveTransform(src, dst)

# Compute the inverse perspective transform (inverse perspective transform to upwarp the image)
# by switching the source and destination points
Minv = cv2.getPerspectiveTransform(dst, src)

# Use cv2.warpPerspective() to warp your image to a top-down view
warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

cv2.imshow("warped image", warped)
cv2.waitKey(0)
"""


###############################################################################
###################   PART 5   JOINING IMAGES   ###############################
###############################################################################
"""
img = cv2.imread("test1.jpg")
imgResize = cv2.resize(img, (360,240))

# Horizontally stack image
horizontal_stack = np.hstack((imgResize,imgResize))

# Vertically stack image
vertical_stack = np.vstack((imgResize, imgResize))

cv2.imshow("Horstack", horizontal_stack)
cv2.imshow("Verstack", vertical_stack)
cv2.waitKey(0)
"""



###############################################################################
###################   PART 6   COLOR DETECTION   ##############################
###############################################################################
def empty():
    pass

def color_tuning(image):
    # Track the color space values
    cv2.namedWindow("TrackBars")
    cv2.resizeWindow("TrackBars", 640, 240)
    cv2.createTrackbar("Hue Min", "TrackBars", 0, 179, empty)
    cv2.createTrackbar("Hue Max", "TrackBars", 40, 179, empty)   # after some experiment, value 40 is good for detecting lane lines
    cv2.createTrackbar("Sat Min", "TrackBars", 0, 255, empty)
    cv2.createTrackbar("Sat Max", "TrackBars", 255, 255, empty)
    cv2.createTrackbar("Val Min", "TrackBars", 213, 255, empty)   # after some experiment, value 213 is good for detecting lane lines
    cv2.createTrackbar("Val Max", "TrackBars", 255, 255, empty)

    while True:
        imgHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
        h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
        s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
        s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
        v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
        v_max = cv2.getTrackbarPos("Val Max", "TrackBars")

        print(h_min, h_max, s_min, s_max, v_min, v_max)

        # We'll use those hsv values to filter out image, so we can get particular image in that range.
        # Create the mask in the range of these colors
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(imgHSV, lower, upper)

        # Add 2 image together to create a new image
        imgResult = cv2.bitwise_and(image, image, mask=mask)

        cv2.imshow("HSV", imgHSV)
        cv2.imshow("mask", mask)
        cv2.imshow("imgResult", imgResult)
        cv2.waitKey(1)




###############################################################################
###############   PART 7   CONTOURS/SHAPE DETECTION   #########################
###############################################################################
"""
def getContours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        print(area)
        if area > 500:
            cv2.drawContours(imgContour, cnt, -1, (0,0,255), 3)

            # get the perimeter of the contours
            perimeter = cv2.arcLength(cnt, True)
            print(perimeter)

            # Approximates a polygonal curve(s) with the specified precision.
            # If the len(approx) == 3, then it's a triangle.
            # If the len(approx) == 4, then it's a square.
            # If the len(approx) > 8, then it's probably a circle
            approx = cv2.approxPolyDP(cnt, perimeter*0.01, True)
            print(len(approx))

            # for drawing a bounding box, we need to know the upper left corner (x,y) and the right lower corner (w,h)
            x, y, w, h = cv2.boundingRect(approx)

            if len(approx) == 3:
                objectType = "Triangle"
            elif len(approx) == 4:
                aspRatio = w / float(h)
                if aspRatio > 0.95 and aspRatio < 1.05:
                    objectType = "Square"
                else:
                    objectType = "Rectangle"
            elif len(approx) > 4:
                objectType = "Circle"
            else:
                objectType = "None"

            # Drawing rectangle bounding boxes
            cv2.rectangle(imgContour, (x,y), (x+w,y+h), (0,255,0), 3)
            # output the shape in the bounding boxes
            cv2.putText(imgContour, objectType, (x+(w//2)-10, y+(h//2)-10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,255), 2)

img = cv2.imread("shapes.jpg")
imgContour = img.copy()

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (7,7), 1)
imgCanny = cv2.Canny(img, 100, 100)
getContours(imgCanny)

cv2.imshow("canny", imgCanny)
cv2.imshow("contour", imgContour)
cv2.waitKey(0)
"""

if __name__ == '__main__':
    try:
        CameraSubscriber()
    except rospy.ROSInterruptException:
        pass
