#!/usr/bin/env python 
# coding=utf-8
#1.编译器声明和2.编码格式声明
#1:为了防止用户没有将python安装在默认的/usr/bin目录，系统会先从env(系统环境变量)里查找python的安装路径，再调用对应路径下的解析器完成操作，也可以指定python3
#2:Python.X 源码文件默认使用utf-8编码，可以正常解析中文，一般而言，都会声明为utf-8编码

import rospy #引用ROS的Python接口功能包
import cv2, cv_bridge #引用opencv功能包。cv_bridge是ROS图像消息和OpenCV图像之间转换的功能包
from sensor_msgs.msg import Image #引用ROS内的图片消息格式
from cv_bridge import CvBridgeError

#定义一个图片转换的类，功能为：订阅ROS图片消息并转换为OpenCV格式处理，处理完成再转换回ROS图片消息后发布
class Image_converter:
 def __init__(self): #类成员初始化函数   
     self.bridge = cv_bridge.CvBridge() #初始化图片转换功能，cv_bridge.CvBridge()
     # self.image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.callback)
     #初始化订阅者,rospy.Publisher()功能是创建订阅者类并输出 self.image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.callback)
     self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.callback)
     self.depth_sub = rospy.Subscriber("/camera/depth/image_raw", Image, self.callback2)
     # self.image_pub = rospy.Publisher("cv_bridge_image", Image, queue_size=1) #初始化发布者,rospy.Publisher()功能是创建发布者类并输出。queue_size为队列长度，当消息发布后订阅者暂时没有接收处理，则该消息进入缓存循环发送，当队列满后最老的数据被踢出队列
 def callback2(self, depth_data): #订阅者接受到消息后的回调函数
     try: #尝试把订阅到的消息转换为opencv图片格式 
      depthFrame = self.bridge.imgmsg_to_cv2(depth_data, desired_encoding='passthrough')
     except CvBridgeError as e: #转换失败则把错误打印出来
      print(e)


     #直接使用OpenCV创建窗口显示图片
     cv2.imshow("depthFrame", depthFrame) 
     cv2.waitKey(10) #功能为刷新图像，若要创建窗口显示图片必须有这一函数。入口参数为延时时间，单位ms，为0则无限延时。函数返回延时时间内键盘按键的ASCII码值


 def callback(self, data): #订阅者接受到消息后的回调函数
     try: #尝试把订阅到的消息转换为opencv图片格式 
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
     except CvBridgeError as e: #转换失败则把错误打印出来
      print(e)


     #直接使用OpenCV创建窗口显示图片
     cv2.circle(cv_image, (40, 40), 20, (0, 255, 0), 2) #再画一个圆
     cv2.imshow("cv_image", cv_image) #OpenCV创建窗口显示图片

     cv2.waitKey(10) #功能为刷新图像，若要创建窗口显示图片必须有这一函数。入口参数为延时时间，单位ms，为0则无限延时。函数返回延时时间内键盘按键的ASCII码值

if __name__ == '__main__': #这段判断的作用是，如果本py文件是直接运行的则判断通过执行if内的内容，如果是import到其他的py文件中被调用(模块重用)则判断不通过
  rospy.init_node("OpencvBridge") #创建节点
  rospy.loginfo("cv_bridge_test node started") #打印ROS消息说明节点已开始运行
  Image_converter() #直接运行image_converter()函数创建类，该类在运行期间会一直存在。因为该类没有需要调用的函数，所以使用赋值的形式：a=image_converter()
  rospy.spin() #相当于while(1),当订阅者接收到新消息时调用回调函数
