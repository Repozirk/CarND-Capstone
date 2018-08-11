#!/bin/bash
catkin_make
source devel/setup.sh
cd ros/src/tl_detector
unzip ssd_mobilenet_v1_coco_2017_11_17.zip
cd ../../
roslaunch launch/styx.launch