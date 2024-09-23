import argparse
import os
import os.path as osp
import time
import cv2
# import torch

import rospy
import message_filters
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError




# from loguru import logger




def ros_imageflow(image_data, depth_data):
    try:
        bridge = CvBridge()
        frame = bridge.imgmsg_to_cv2(image_data, desired_encoding='bgr8')
        depthFrame = bridge.imgmsg_to_cv2(depth_data, desired_encoding='passthrough')
        # hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        cv2.imshow('Camera', frame)
        cv2.waitKey(3)
    except CvBridgeError as e:
        print(e)

    

    


if __name__ == "__main__":
    rospy.init_node("camera_subscriber", anonymous = False)
    im_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
    dep_sub = message_filters.Subscriber('/camera/depth/image_raw', Image)
    sync = message_filters.ApproximateTimeSynchronizer([im_sub, dep_sub], 50, 0.5)
    # sync = message_filters.TimeSynchronizer([im_sub, dep_sub], 0.5)
    sync.registerCallback(ros_imageflow)
    

    


    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.logwarn('failed')