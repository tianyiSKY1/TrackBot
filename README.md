# TrackBot
This is an ROS package for a mobile robot that implements visual multi-target tracking. The currently implemented functionality allows the robot to follow a selected person within its field of view. This function was developed based on the Wheeltec robot, so we are only open-source the multi-target tracking feature.
<p align="center">
  <img src="trackbot.jpeg" alt="Trackbot" width="400"/>
</p>

## Jetson
To address the limited onboard computing power of the mobile robot, we have offloaded the visual processing tasks to an NVIDIA Jetson platform. The multi-object tracking algorithm running on the Jetson achieves a processing speed of around 25 FPS. We have configured the robot’s movement system as the ROS master and the Jetson’s visual computation system as the ROS slave. Through ROS’s distributed communication framework, the selected target’s coordinates are transmitted via Wi-Fi to the ROS master, ultimately enabling multi-target tracking functionality.
<p align="center">
  <img src="jetson.jpeg" alt="Trackbot" width="600"/>
</p>

## Display
You can select any person on the screen to follow.
<p align="center">
  <img src="display2.jpeg" alt="Trackbot" width="400"/>
  <img src="display1.jpeg" alt="Trackbot" width="400"/>
</p>
