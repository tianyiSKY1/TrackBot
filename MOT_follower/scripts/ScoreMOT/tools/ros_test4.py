import argparse
import os
import os.path as osp
import time
import cv2
import torch

import rospy
import message_filters
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError




from loguru import logger




def ros_imageflow(depth_data):
    try:
        bridge = CvBridge()
        #cv_image = bridge.imgmsg_to_cv2(image_data, desired_encoding='bgr8')
        depthFrame = bridge.imgmsg_to_cv2(depth_data, desired_encoding='passthrough')
        cv2.imshow('Camera', depthFrame)
        cv2.waitKey(3)
    except CvBridgeError as e:
        print(e)
def main():
    rospy.init_node("camera_subscriber")
    # im_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
    dep_sub = message_filters.Subscriber('/camera/depth/image_raw', Image)
    sync = message_filters.ApproximateTimeSynchronizer([dep_sub], 10, 0.5)
    # sync = message_filters.TimeSynchronizer([im_sub, dep_sub], 1)
    sync.registerCallback(ros_imageflow)

    


if __name__ == "__main__":
    main()
    rospy.spin()