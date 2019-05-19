import rospy
import tf
from geometry_msgs.msg import TransformStamped
#geometry_msgs.msg.TransformStamped
import numpy as np
from scipy.stats import multivariate_normal as mvn
from scipy.spatial import distance
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import math
from duckietown_msgs.msg import WheelsCmdStamped


import datetime as dt
import matplotlib.animation as animation
import tmp102

from tf.transformations import euler_from_quaternion

import operator

#actual landmark values - change
obstacle = np.array([0.4572,-0.1524]) #ACTUAL location of obstacle - array of x and y - can also put inside class

class ViconSubscriber(object):
    """
    Subscribes to Vicon topics to get an object's pose.
    """

    def __init__(self, name):
        """
        Parameters
        ----------
        name: str
            Object name, as defined in the Vicon system.
            Used to find the Vicon topic to subscribe to.
            e.g. if name=jackal3, will subscribe to the vicon/jackal3/jackal3 topic.
        """
        self.vicon_sub = rospy.Subscriber('/vicon/%s/%s' % (name, name), TransformStamped,
                                          self.vicon_callback)
        self.x = None
        self.y = None
        self.angle = None
        self.name = name

    def vicon_callback(self, data):
        """
        Parameters
        ----------
        data: TransformStamped
        """
        self.x = data.transform.translation.x
        self.y = data.transform.translation.y

        # get rotation quaternion
        q = data.transform.rotation
        euler = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.angle = euler[2] # z-axis rotation in radians

    def check_received(self):
        """ Returns True if the subscriber has received Vicon data, False otherwise."""
        if self.x is None or self.y is None or self.angle is None:
            return False
        else:
            return True

    def get_pose(self):
        return np.array([self.x, self.y, self.angle])


class Localization(object):

    def __init__(self):
        self.wheel_topic = '/duckiebot1/wheels_driver_node/wheels_cmd'
        self.wheel_commands = rospy.Subscriber(self.wheel_topic, WheelsCmdStamped, callback = self.wheel_callback)
        self.wheel_dx = 0
        self.wheel_dy = 0

        #Velocity tracker
        self.vel_left = None
        self.vel_right = None

        #Initialize particles
        self.numParticles = 10
        self.initialSet = np.zeros([self.numParticles, 3])
        self.particleSet = np.zeros([self.numParticles, 3])

        #Sample uniformly over +/- span in x/y
        spanB = -1 #-0.5 in meters?
        spanE = 1 #0.5
        piB = 0
        piE = 2*math.pi

        xSamples = np.random.uniform(spanB, spanE, self.numParticles)
        ySamples = np.random.uniform(spanB, spanE, self.numParticles)
        thetaSamples = np.random.uniform(piB, piE, self.numParticles)

        for k in range(0, self.numParticles):
            self.particleSet[k, :] = [xSamples[k], ySamples[k], thetaSamples[k]] #DOES THIS WORK?

        self.initialSet = self.particleSet

        #Time tracker
        self.prev_time = rospy.get_rostime()
        self.current_time = rospy.get_rostime()

        #listener for robot and obstacle
        self.robot_vicon = ViconSubscriber("duckiebot1")
        self.obs_vicon = ViconSubscriber("cup2")
        self.sigma = np.array([0.05, 0.05, 0.1])
        self.sig = np.array([0.1, 0.1]) #NEED TO GUESS
        self.diameter = 0.1143

        self.plot = False

    def global_to_local(self,q,p):
        TF = np.array([[math.cos(q[2]), math.sin(q[2])],[-math.sin(q[2]), -math.cos(q[2])]])
        globalPoint = np.array([p[0], p[1]])
        return np.dot(TF,globalPoint)

    #ISSUE
    def wheel_callback(self,data):
        self.vel_left = data.vel_left
        self.vel_right = data.vel_right

        self.current_time = rospy.get_rostime()
        dt = (self.current_time.secs * math.pow(10, 9) + self.current_time.nsecs) - (self.prev_time.secs * math.pow(10,9) + self.prev_time.nsecs)
        self.prev_time = self.current_time

        dist_left = self.vel_left * dt
        dist_right = self.vel_right * dt
        avg = (dist_left + dist_right) / 2
        theta = (dist_right - dist_left) / self.diameter

        self.wheel_dx = avg * math.cos(theta)
        self.wheel_dy = avg * math.sin(theta)

    def run(self):
        print("in run")
        rate = rospy.Rate(15)

        #wait until we start getting data
        while self.vel_left is None or self.vel_right is None:
            rate.sleep()

        while not rospy.is_shutdown():
            print('in loop')
            #change in time
            self.current_time = rospy.get_rostime()
            dt = (self.current_time.secs * math.pow(10, 9) + self.current_time.nsecs) - (self.prev_time.secs * math.pow(10,9) + self.prev_time.nsecs)
            self.prev_time = self.current_time

            # Robot and obstacle location
            robot_pose = self.robot_vicon.get_pose()
            obs_pose = self.obs_vicon.get_pose()

            #data with noise
            obs_new = np.random.normal(loc = obs_pose, scale = self.sigma)
            robot_new = np.random.normal(loc = robot_pose, scale = self.sigma)

            particleWeights = np.zeros(self.numParticles)

            dist_left = self.vel_left * dt
            dist_right = self.vel_right * dt
            avg = (dist_left + dist_right) / 2
            theta = (dist_right - dist_left) / self.diameter

            self.wheel_dx += avg * math.cos(theta)
            self.wheel_dy += avg * math.sin(theta)

            #calculate predicted values and then weight
            for j in range(0, len(self.particleSet)):
                newLocX = self.particleSet[j][0] + self.wheel_dx
                newLocY = self.particleSet[j][1] + self.wheel_dy
                newLoctheta = self.particleSet[j][2] + theta
                #propagated particle
                newLoc = np.array([newLocX, newLocY, newLoctheta]) #predicted robot's location (I want predicted obstacle's location given my particle?)

                #local frame
                localLoc = self.global_to_local(newLoc, obstacle)
                localLoc = localLoc[:2]
                
                #actual
                actualLoc = self.global_to_local(robot_new, obs_new)

                particleWeights[j] = mvn.pdf(localLoc, mean = actualLoc, cov = self.sigma)

            sum_weights = np.sum(particleWeights)
            particleWeights = particleWeights/sum_weights

            #code for plotting
            #index, max_weight = max(enumerate(particleWeights), key=operator.itemgetter(1))
            #final_particle = particleSet[index] #take the particle with the max weight

            #select indices
            newParticles = np.zeros([self.numParticles, 3])
            particleIndices = range(0, self.numParticles)  # np.random.choice only selects from 1D arrays - select by index
            selectedIndices = np.random.choice(particleIndices, self.numParticles, p=particleWeights)

            for j in range(0, len(selectedIndices)):
                newParticles[j] = self.particleSet[selectedIndices[j]]

            self.particleSet = newParticles
            
            

            self.wheel_dx = 0
            self.wheel_dy = 0

            rate.sleep() #for how long? Can we use this to take measurements every 5 seconds?

if __name__ == "__main__":
    rospy.init_node('localization') #what should this be?
    detector = Localization()
    detector.run()
