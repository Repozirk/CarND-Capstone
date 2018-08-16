#!/bin/bash
cd ros
catkin_make
source devel/setup.sh
cd src/tl_detector
unzip -o ssd_mobilenet_v1_coco_2017_11_17.zip
cd ../../
find . -type f -iname *.py -exec chmod +x {} \;
roslaunch launch/styx.launch