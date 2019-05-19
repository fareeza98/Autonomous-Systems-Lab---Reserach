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

import operator

#actual landmark values - change
obstacle = None #ACTUAL location of obstacle - array of x and y - can also put inside class

class Localization(object):

    def __init__(self):
        self.wheel_topic = '/duckiebot1/wheels_driver_node/wheels_cmd'
        self.wheel_commands = rospy.Subscriber(wheel_topic, WheelsCmdStamped, callback = self.wheel_callback)
        self.wheel_dx = 0
        self.wheel_dy = 0
        
        #Velocity tracker
        self.vel_left = None
        self.vel_right = None
        
        #Initialize particles
        self.numParticles = 200
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
        
        self.particleSet = [xSamples, ySamples, thetaSamples] #DOES THIS WORK?
        
        #Time tracker
        self.prev_time = rospy.get_rostime()
        self.current_time = rospy.get_rostime()
        
        #listener for robot and obstacle
        self.listener = tf.TransformListener()
        self.robot_node = '/vicon/Sphero2/Sphero2'
        self.robot = rospy.Subscriber(robot_node, TransformStamped, callback = self.robot_callback)
        self.obstacle_node = None #set later
        self.sigma = np.array([0.05, 0.05, 0.1])
        sig = np.array([0.1, 0.1]) #NEED TO GUESS
        self.diameter = 0.1143
    
        self.plot = False
    
    def global_to_local(self,q,p):
        TF = np.array([[math.cos(q[2]), math.sin(q[2])],[-math.sin(q[2]), -math.cos(q[2])]])
        globalPoint = np.array([p[0], p[1]])
        return np.dot(TF,globalPoint)
    
    def robot_callback(self, r_data):
        self.robot_data = r_data
    
    #ISSUE
    def wheel_callback(self,data):
        self.vel_left = data.vel_left
        self.vel_right = data.vel_right
        
        self.current_time = rospy.get_rostime()
        dt = (self.current_time.secs * math.pow(10, 9) + self.current_time.nsecs) - (self.prev_time.secs * math.pow(10,9) + self.prev_time.nsces)
        self.prev_time = self.current_time
        
        dist_left = self.vel_left * dt
        dist_right = self.vel_right * dt
        avg = (dist_left + dist_right) / 2
        theta = (dist_right - dist_left) / self.diameter
    
        self.wheel_dx = avg * math.cos(theta)
        self.wheel_dy = avg * math.sin(theta)
    
    def run(self):
        rate = rospy.Rate(15)

        #wait until we start getting data
        while self.vel_left is None or self.vel_right is None:
            rate.sleep()
        
        while not rospy.is_shutdown():
            
            #change in time
            self.current_time = rospy.get_rostime()
            dt = (self.current_time.secs * math.pow(10, 9) + self.current_time.nsecs) - (self.prev_time.secs * math.pow(10,9) + self.prev_time.nsces)
            
            self.prev_time = self.current_time
            
            # Robot location and goal proximity
            try:
                # Get robot position in the local, table frame
                (trans,rot) = self.listener.lookupTransform(robot_node,obstacle_node, rospy.Time(0)) #get robot in obstacle's frame
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                print 'Vicon problem?'
                continue
            
            #the obstacle's actual dx and dy - sensor evaluation
            euler = tf.transformations.euler_from_quaternion(rot)
            localPose = np.array([trans[0],trans[1],euler[2]])
            lp = np.array([localPose[0],localPose[1]])
            
            #add noise to vicon data
            lp = np.random.normal(loc = lp, scale = self.sigma)

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
                #propagated particle
                newLoc = np.array([newLocX, newLocY]) #predicted robot's location (I want predicted obstacle's location given my particle?)
                
                #local frame
                localLoc = self.global_to_local(newLoc, obstacle)
                
                particleWeights[j] = mvn.pdf(localLoc, mean = lp, cov = self.sig)
                    
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
                
            #figure out how to keep plotting on same figure
            if not self.plot:
                fig = plt.figure(1) #create a new plot only if no plots exist
            mSize = 3
                
            plt.plot(obstacle[0], obstacle[1], 's', markersize = 5, color='y')
                
            for j in range(0, len(self.initialSet)):
                plt.plot(self.particleSet[j,0], self.particleSet[j, 1], 'o', markersize= mSize, color='b')
            
            plt.plot(self.robot_data.transform.translation.y, self.robot_data.transform.translation.x , 'o', markersize=mSize, color='r')
            
            plt.show()
            plt.clf()

            self.wheel_dx = 0
            self.wheel_dy = 0

            rate.sleep() #for how long? Can we use this to take measurements every 5 seconds?


if __name__ == "__main__":
    rospy.init_node('localization') #what should this be?
    detector = Localization()
    detector.run()

