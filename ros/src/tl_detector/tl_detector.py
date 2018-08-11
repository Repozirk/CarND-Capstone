#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf as transform
import cv2
import yaml
import tensorflow as tf
import numpy as np
import math
import timeit
from tensorflow.python.client import timeline
import os
from time import sleep
from scipy.spatial import KDTree

#TODO: only run tl detection when there's an up coming traffic light - based on map data, in order to save processing time.

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
SHOW_TRAFFIC_LIGHT_GT = False

USE_SIMULATOR_TL_STATE = False #decide whether to use simulator's traffic light status or use actual detection output, should be false for final submission
STATE_COUNT_THRESHOLD = 2 #changed to 2 due to delay in processing (function call order)
TL_DISTANCE_LIMIT = 150
TRACE = False
#show processing time or debugging info
SHOW_PROCESSING_TIME = False
SHOW_DEBUGGING_INFO = False
CREATE_DATASET = False
if CREATE_DATASET:
    create_image_index = 290

#set whether simulator is being used or not
SIMULATOR_OR_NOT = True
IMAGE_SKIP_COUNT = 2 #every third /image_color will be processed for detection/classification
VIDEO_RECORD = False

IMAGE_WIDTH = 800
IMAGE_HEIGHT = 600

#uncomment the following if the video comes from rosbag file
if SIMULATOR_OR_NOT == False: 
    IMAGE_WIDTH = 1368
    IMAGE_HEIGHT = 1096
if VIDEO_RECORD:
    VID_RECORD_FRAMERATE = 4.0 #processing speed is roughly 10Hz
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    video = cv2.VideoWriter('video.avi',fourcc,VID_RECORD_FRAMERATE,(IMAGE_WIDTH,IMAGE_HEIGHT))

class TLDetector(object):
    def __init__(self):
    #def __init__(self):
        rospy.init_node('tl_detector')

        # What model to download.
        MODEL_NAME = './ssd_mobilenet_v1_coco_2017_11_17'
        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
        # tf.__version__
        if SHOW_DEBUGGING_INFO:
            print("loading tensorflow model and weights")
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        if SHOW_DEBUGGING_INFO:
            print(self.detection_graph)
            print("finished loading tensorflow model and weights")


        #create reusable sesion
        config = tf.ConfigProto()
        self.sess = tf.Session(graph=self.detection_graph, config=config)

        self.pose = None
        self.waypoints = None

        # Waypoint KD Tree
        self.waypoints_2d = None
        self.waypoints_tree = None

        self.camera_image = None
        self.lights = []
        self.tl_filtered_state = "UNKNOWN" #initialize to UNKNOWN

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)

        if (SIMULATOR_OR_NOT == True):
            sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)
        else:
            sub6 = rospy.Subscriber('/image_raw', Image, self.image_cb)


        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = transform.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        # [x,y] coordinates of stopping lines before traffic lights
        self.stop_line_positions = self.config['stop_line_positions']
        # Indices in self.waypoints of the waypoints closest to the respective stopping lines, as reported in self.stop_line_positions
        self.stop_line_idxs = None  # Can be initialised only after receiving the list of waypoints, see waypoints_cb()

        #set img_count in order to be able to skip camera frames
        self.img_count = 0

        rospy.spin()

        #the following can be used to control update rate instead of letting rospy.spin take care of things
        # update_rate = rospy.Rate(5)
        #
        # while not rospy.is_shutdown():
        #     update_rate.sleep()
  
    def run_inference_for_single_image(self, image):
        with self.detection_graph.as_default():
            # Get handles to input and output tensors
            if SHOW_PROCESSING_TIME:
                start_time = timeit.default_timer()
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                  tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                      tensor_name)

            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
            if SHOW_PROCESSING_TIME:
                print("tensor preprocessing time is {}".format(timeit.default_timer() - start_time))
            if SHOW_PROCESSING_TIME:
                start_time = timeit.default_timer()
            # Run inference
            if TRACE:
                options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                output_dict = self.sess.run(tensor_dict,
                                       feed_dict={image_tensor: np.expand_dims(image, 0)}, options=options, run_metadata=run_metadata)
                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()
                with open('Experiment_1.json', 'w') as f:
                    f.write(chrome_trace)
            else:
                output_dict = self.sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})
            if SHOW_PROCESSING_TIME:
                print("inside inference time is {}".format(timeit.default_timer() - start_time))
            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
        return output_dict


    def pose_cb(self, msg):
        self.prev_pose = self.pose
        self.pose = msg
        if (self.prev_pose and self.pose and self.prev_pose != self.pose):
            this_heading = math.atan2(self.pose.pose.position.y - self.prev_pose.pose.position.y,
                                      self.pose.pose.position.x - self.prev_pose.pose.position.x)
            this_distance = math.hypot(self.pose.pose.position.y - self.prev_pose.pose.position.y,
                                       self.pose.pose.position.x - self.prev_pose.pose.position.x)

            if (0.01 < this_distance):
                self.heading = this_heading

    # Note: waypoints remain static, this runs only once
    def waypoints_cb(self, waypoints):
        print("static waypoints received")
        self.waypoints = waypoints
        # Setup the Kd Tree which has log(n) complexity
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in
                                 waypoints.waypoints]
            self.waypoints_tree = KDTree(self.waypoints_2d)

    # WARNING: lights state will not be available in real life, only available when using simulator
    def traffic_cb(self, msg):
        self.lights = msg.lights
        #print(self.lights.state)

    def detect_tl(self, image):
        #must convert cv2's bgr format to rgb by doing image[...,::-1] trick for NN forward pass
        image_np = image[..., ::-1]

        #initialize tl_detected as false
        tl_detected = False

        #create copy of the image where rectangles will be drawn for visualization
        proc_image = image

        # Actual detection.
        if SHOW_PROCESSING_TIME:
            start_time = timeit.default_timer()
        output_dict = self.run_inference_for_single_image(image_np)
        if SHOW_PROCESSING_TIME:
            print("only inference elapsed time is {}".format(timeit.default_timer() - start_time))

        #store all detection classes and scores into separate np arrays
        det_classes = output_dict['detection_classes']
        det_scores = output_dict['detection_scores']

        #store index of the detected classes that correspond to 10
        tl_classes_index = np.where(det_classes==10)[0]

        #loop through each of the traffic light detections and weed out the ones below threshold
        TL_DET_THRESHOLD = 0.40

        #initialize status_red as UNKNOWN, if changed to red, will not return to UNKNOWN.
        #if changed to yellow/green, can change to red but not to UNKNOWN
        status_red = TrafficLight.UNKNOWN

        for det_tl_i in tl_classes_index:
            #print(det_tl)
            #print(det_scores[det_tl_i])

            if det_scores[det_tl_i] > TL_DET_THRESHOLD:
                cv2.imwrite('imgs/camera_image.jpeg', image)
                #draw the detection box for the filtered out traffic light detection
                ymin = int(output_dict['detection_boxes'][det_tl_i][0] * IMAGE_HEIGHT)
                xmin = int(output_dict['detection_boxes'][det_tl_i][1] * IMAGE_WIDTH)
                ymax = int(output_dict['detection_boxes'][det_tl_i][2] * IMAGE_HEIGHT)
                xmax = int(output_dict['detection_boxes'][det_tl_i][3] * IMAGE_WIDTH)
                #print("found valid traffic lights")

                #add rectangle patches to the copy of the image

                if SHOW_PROCESSING_TIME:
                    start_time = timeit.default_timer()

                # get bgr crop of the traffic light only from the original image
                tl_img_bgr = image[ymin:ymax, xmin:xmax]

                # get rgb crop from the original image
                tl_img = image_np[ymin:ymax, xmin:xmax]

                # get YUV and HSV formats
                tl_img_lab = cv2.cvtColor(tl_img_bgr, cv2.COLOR_BGR2LAB)

                #print(tl_img.shape)

                r = tl_img[:, :, 0] #extract red channel
                g = tl_img[:,:,1] #extract green channel
                l = tl_img_lab[:, :, 0]

                # get detected object's height
                (tl_img_height, tl_img_width, tl_img_channels) = tl_img.shape

                # L level threshold (L from LAB colorspace)
                L_THRES = 240
                # R level threshold
                RED_THRESHOLD = 240
                YELLOW_THRESHOLD = 240
                GREEN_THRESHOLD = 240

                # depending on the detected traffic light size, the threshold shall be adjusted
                # following is an equatiion driven from some data points manually gathered
                # 51,39    70,57     152,312      190   550  -- conservative
                COUNT_THRESHOLD = int(3 * tl_img_height - 200)

                #make sure the threshold doesn't get below 0
                if(COUNT_THRESHOLD <=14):
                    COUNT_THRESHOLD = 15 #this is the minimum number of red pixels detected to qualify as a red light

                if VIDEO_RECORD:
                    coord = (xmin, ymin-10)
                    font = cv2.FONT_HERSHEY_PLAIN
                    fontScale = 3
                    fontColor = (0, 0, 255)
                    lineType = cv2.LINE_AA
                    thickness = 2
                    #first draw a box around the detected item
                    cv2.rectangle(proc_image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

                # check image ratio:
                if tl_img_height > tl_img_width * 1.7: #changed from 1.8 to 1.7 as there were some cases like 36, 64
                    # if a valid traffic light has been detected after checking image ratio, turn the tl_detected flag to True
                    tl_detected = True

                    if SHOW_DEBUGGING_INFO:
                        print("size ratio good")
                        print(COUNT_THRESHOLD)
                        print(tl_img.shape)

                    # check for the most bright segment out of top and bottom segments, as we know we are only interested on the top and bottom
                    # where top segment would correspond to red light, and bottom segment would corespond to green light
                    seg_top = (l[0:1 * int(tl_img_height / 3), :] > L_THRES).sum()
                    seg_middle = (l[1 * int(tl_img_height / 3):2 * int(tl_img_height / 3), :] > L_THRES).sum()
                    seg_bottom = (l[2 * int(tl_img_height / 3):3 * int(tl_img_height / 3), :] > L_THRES).sum()
                    # check if top segment is greater than the bottom segment, otherwise it's not red light
                    global SIMULATOR_OR_NOT

                    if ((seg_top > seg_bottom) and (seg_top > seg_middle)) or (SIMULATOR_OR_NOT == True):
                        # now the top segment is the highest, but by chance this could mean that some very bright object or the sun could be shining on the top part of the traffic light
                        # in order to check that it really is red light, check for R channel content's magnitude.
                        if ((r[0:1 * int(tl_img_height / 3), :] > RED_THRESHOLD).sum() > COUNT_THRESHOLD):
                            #print("red")
                            if VIDEO_RECORD:
                                cv2.putText(proc_image, "RED  TL: " + self.tl_filtered_state, coord, font, fontScale, fontColor, thickness, lineType)
                            status_red=TrafficLight.RED
                        else:
                            #TODO: must account for back side of the traffic light - will all be black.. no real difference in R content in the three segments..
                            #if all the detected objects are the back side of the traffic lights due to poor driving, then consider it as unknown
                            #if R content in all 3 segments are low then it must be the black back side of traffic lights
                            #now this would only be for simulation mode.. for real case, the L factor should determine it.. well if all these tests fail, then it will be UNKNOWN anyway so good

                            if SIMULATOR_OR_NOT == False:
                                #if not simulator, this means it's unknown, could be bright light or anything
                                if VIDEO_RECORD:
                                    cv2.putText(proc_image, "UNKNOWN  TL: " + self.tl_filtered_state, coord, font,
                                                fontScale, fontColor, thickness, lineType)
                            else:
                                #if using simulator, brightness can't be a factor, which means you have to decipher whether green/yellow vs unknown using colors only

                                #if mid section doesn't have enough yellow, and bottom section doesn't have enough green, then deem it as back of traffic lights or invalid classifications
                                if SHOW_DEBUGGING_INFO:
                                    print("yellow pixels in the middle: {} vs {}".format((((r[1 * int(tl_img_height / 3):2 * int(tl_img_height / 3), :] > RED_THRESHOLD).sum()) + (g[1 * int(tl_img_height / 3):2 * int(tl_img_height / 3), :] > GREEN_THRESHOLD).sum()), COUNT_THRESHOLD))
                                    #*2 for YELLOW_THRESHOLD since yellow is R255 G255
                                    print("green pixels in the bottom: {} vs {}".format((g[2 * int(tl_img_height / 3):3 * int(tl_img_height / 3), :] > GREEN_THRESHOLD).sum(), COUNT_THRESHOLD))
                                yellowcond = ((((r[1 * int(tl_img_height / 3):2 * int(tl_img_height / 3), :] > RED_THRESHOLD).sum()) + (g[1 * int(tl_img_height / 3):2 * int(tl_img_height / 3), :] > GREEN_THRESHOLD).sum()) > COUNT_THRESHOLD)
                                greencond = (g[2 * int(tl_img_height / 3):3 * int(tl_img_height / 3), :] > GREEN_THRESHOLD).sum() > COUNT_THRESHOLD
                                yellow_or_green = yellowcond or greencond
                                if (yellow_or_green != True):
                                    #don't touch the status_red basically.. so UNKNOWN
                                    # status_red = TrafficLight.UNKNOWN
                                    if VIDEO_RECORD:
                                        cv2.putText(proc_image, "UNKNOWN  TL: " + self.tl_filtered_state, coord, font,
                                                    fontScale, fontColor, thickness, lineType)
                                else:
                                    #if not.. then for simulator only, consider it GREEN/YELLOW
                                    # for simulator, if in here, it means the detected light was not red, and there's no sun shining or anything, so it must mean YELLOW/GREEN
                                    status_red = 2  # GREEN = 2
                                    if VIDEO_RECORD:
                                        cv2.putText(proc_image, "GREEN/YELLOW  TL: " + self.tl_filtered_state, coord,
                                                    font, fontScale, fontColor, thickness, lineType)
                    else:
                        # if the top segment is not the brightest, consider the light as not red, and consider it yas yellow/green
                        #if RED was detected in the frame, just keep RED, otherwise change to YELLOW/GREEN
                        if status_red != TrafficLight.RED:
                            status_red = 2 #GREEN = 2

                        if VIDEO_RECORD:
                            cv2.putText(proc_image, "GREEN/YELLOW  TL: " + self.tl_filtered_state, coord, font, fontScale, fontColor, thickness, lineType)
                        #print("not red")
                else:
                    if SHOW_DEBUGGING_INFO:
                        print("size ratio not good")
                        print(tl_img.shape)
                    if VIDEO_RECORD:
                        cv2.putText(proc_image, "UNKNOWN  TL: " + self.tl_filtered_state, coord, font, fontScale, fontColor, thickness, lineType)
                    # no need to set it to UNKNOWN, already initialized to it, and if was assigned to RED or YELLOW/GREEN, shouldn't return to UNKNOWN
                    #status_red = TrafficLight.UNKNOWN

                if SHOW_DEBUGGING_INFO:
                    print((l[0:1 * int(tl_img_height / 3), :] > L_THRES).sum())
                    print((r[0:1 * int(tl_img_height / 3), :] > RED_THRESHOLD).sum())

                    print((l[1 * int(tl_img_height / 3):2 * int(tl_img_height / 3), :] > L_THRES).sum())
                    print((r[1 * int(tl_img_height / 3):2 * int(tl_img_height / 3), :] > RED_THRESHOLD).sum())

                    print((l[2 * int(tl_img_height / 3):3 * int(tl_img_height / 3), :] > L_THRES).sum())
                    print((r[2 * int(tl_img_height / 3):3 * int(tl_img_height / 3), :] > RED_THRESHOLD).sum())
                if SHOW_PROCESSING_TIME:
                    print("classification time is {}".format(timeit.default_timer() - start_time))

                if CREATE_DATASET and (status_red != TrafficLight.UNKNOWN):
                    global create_image_index
                    cv2.imwrite("./training_img/image" + str(create_image_index) + ".jpg", image)
                    create_image_index = create_image_index + 1
                    sleep(0.5)  # Time in seconds.
        return proc_image, status_red, tl_detected


    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        if(self.img_count == IMAGE_SKIP_COUNT): #IMAGE_SKIP_COUNT = 2 means skipping 2 frames. (processing only every third frame)
            self.img_count = 0 #reset image skip count
            self.has_image = True
            self.camera_image = msg
            light_wp, state = self.process_traffic_lights()

            #print("Received an image!")

            '''
            Publish upcoming red lights at camera frequency.
            Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
            of times till we start using it. Otherwise the previous stable state is
            used.
            '''

            if self.state != state:
                self.state_count = 0
                self.state = state
            elif self.state_count >= STATE_COUNT_THRESHOLD:
                self.last_state = self.state
                # if 3 consequent consistent light types are detected, finally classify the traffic light as something definitely
                if(self.state == TrafficLight.UNKNOWN):
                    self.tl_filtered_state = "UNKNOWN"
                elif(self.state == TrafficLight.RED):
                    self.tl_filtered_state = "RED"
                else:
                    self.tl_filtered_state = "YELOW/GREEN"
                light_wp = light_wp if state == TrafficLight.RED else -1
                self.last_wp = light_wp
                self.upcoming_red_light_pub.publish(Int32(light_wp))
            else:
                self.upcoming_red_light_pub.publish(Int32(self.last_wp))
            self.state_count += 1

        else:
            self.img_count = self.img_count + 1 #increment skip count


    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        closest_idx = self.waypoints_tree.query([x, y], 1)[1]
        return closest_idx

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv2_img = self.bridge.imgmsg_to_cv2(self.camera_image,"bgr8")
        #print(cv2_img.shape)


        #write images to video
        #print(cv2_img)
        #perform detection
        if SHOW_PROCESSING_TIME:
            start_time = timeit.default_timer()

        #run NN and CV detection/classification
        proc_image, status_red, tl_detected = self.detect_tl(cv2_img)

        if SHOW_PROCESSING_TIME:
            print("detect_tl elapsed time is {}".format(timeit.default_timer() - start_time))

        if VIDEO_RECORD:
            video.write(proc_image)
            #print("am i getting here")
        #video.write(cv2_img)
        if(status_red == TrafficLight.RED):
            print("red detected")
        elif(status_red == TrafficLight.UNKNOWN):
            print("traffic light not detected / can't classify")
        else:
            print("yellow/green detected")

        #Get classification for red light or not, didn't utilize the separate light classification module
        #maybe later when everything is done, the classification section can be moved to the separate classifier
        return status_red

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #initialize light status as Unknown = 4
        light = TrafficLight.UNKNOWN
        state = TrafficLight.UNKNOWN

        # List of positions that correspond to the line to stop in front of for a given intersection

        closest_light = None
        line_wp_idx = -1

        #if pose exists, go and find the closest light's index
        if (self.pose):
            # waypoint closest to current car pose
            car_wp_idx = self.get_closest_waypoint(self.pose.pose.position.x, self.pose.pose.position.y)
            # total number of waypoints
            diff = len(self.waypoints.waypoints)

            for i, light in enumerate(self.lights):
                line = self.stop_line_positions[i]
                temp_wp_idx = self.get_closest_waypoint(line[0], line[1])

                d = temp_wp_idx - car_wp_idx

                if d >= 0 and d < diff:
                    diff = d
                    closest_light = light
                    line_wp_idx = temp_wp_idx
                    #print("waypoint idx: {}".format(line_wp_idx))

        # if there exists a light coming up, then get the light state
        if closest_light:
            if USE_SIMULATOR_TL_STATE:
                # self.lights is an array, array of all the traffic lights (static locations). The state can be accessed as below
                # grab simulator traffic light status output (grab the 1st traffic light: self.lights[0])
                sim_tl_state = self.lights[0].state
                light = sim_tl_state
            else:
                # this call is where it calls the NN detection/classification
                state = self.get_light_state(closest_light)
                if state != TrafficLight.UNKNOWN:
                    light = state

            if SHOW_TRAFFIC_LIGHT_GT and (USE_SIMULATOR_TL_STATE == False):
                print("NN result: {}, simulator result: {}".format(state, sim_tl_state))

            print("filtered result: {}, USING SIMULATOR? {}".format(self.tl_filtered_state, USE_SIMULATOR_TL_STATE))
            print("------------------------------------------")
            return line_wp_idx, light  # if it isn't unknown light status
        # if there's no close traffic light, then return light status as UNKNOWN(default) and idx = -1
        else:
            return -1, TrafficLight.UNKNOWN
        #rospy.loginfo("ptl: state=" + str(state) + " sim=" + str(sim_tl_state))

if __name__ == '__main__':

    try:
        TLDetector()
        # stop video writing and release the video
        if VIDEO_RECORD:
            cv2.destroyAllWindows()
            video.release()

    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
