import numpy as np
from scipy.stats import multivariate_normal as mvn
from scipy.spatial import distance
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
#from mpl_toolkits.mplot3d import Axes3D
import math

import operator

def global_to_local(q,p):
    TF = np.array([[math.cos(q[2]), math.sin(q[2])],[-math.sin(q[2]), -math.cos(q[2])]])
    globalPoint = np.array([p[0], p[1]])
    return np.dot(TF,globalPoint)

#actual landmark values
landmarkA = [-2, 2]
landmarkB = [0.75, -1.25]
landmarkC = [-1.1, 0.8]
landmarkD = [1.5, 2.1]

incr = 0;
speed = 0.5 #stays constant
data = open('diffRobot_localizeRobot.txt', 'r')
lines = data.readlines()

dataSet = [] #arrays of time, control, landamrks 1 to 4
robot = []

for line in lines:
    time, robotState, control, landmark1, landmark2, landmark3, landmark4 = line.split(",")

    robotState = robotState[2:-1]
    rx, ry, theta = robotState.split()
    robotState = []
    rx = float(rx)
    ry = float(ry)
    theta = float(theta)
    robotState.append(rx)
    robotState.append(ry)
    robotState.append(theta)

    robot.append(robotState)

    control = control[2:-1]
    control = float(control)
    time = float(time)

    landmark1 = landmark1[2:-1]
    l1, l2 = landmark1.split()
    landmark1 = []
    l1 = float(l1)
    l2 = float(l2)
    landmark1.append(l1)
    landmark1.append(l2)

    landmark2 = landmark2[2:-1]
    l1, l2 = landmark2.split()
    landmark2 = []
    l1 = float(l1)
    l2 = float(l2)
    landmark2.append(l1)
    landmark2.append(l2)

    landmark3 = landmark3[2:-1]
    l1, l2 = landmark3.split()
    landmark3 = []
    l1 = float(l1)
    l2 = float(l2)
    landmark3.append(l1)
    landmark3.append(l2)

    landmark4 = landmark4[2:-2]
    l1, l2 = landmark4.split()
    landmark4 = []
    l1 = float(l1)
    l2 = float(l2)
    landmark4.append(l1)
    landmark4.append(l2)

    new = []
    new.append(time)
    new.append(control)
    new.append(landmark1)
    new.append(landmark2)
    new.append(landmark3)
    new.append(landmark4)
    dataSet.append(new)

#Initialize particles
numParticles = 200

particleSet = np.zeros([numParticles, 3])  # empty array of particles

#Sample uniformly over +/- span in x/y
spanB = -4 #-4
spanE = 4 #3
piB = 0
piE = 2*math.pi

#initialize all particles to initial position of robot
xSamples = np.random.uniform(spanB, spanE, numParticles)
ySamples = np.random.uniform(spanB, spanE, numParticles)
thetaSamples = np.random.uniform(piB, piE, numParticles)

for i in range(0, numParticles): #FIND NP FUNCTION INSTEAD
    particleSet[i, :] = [xSamples[i], ySamples[i], thetaSamples[i]]
initialSet = particleSet  # Save initial set for plotting later

#sensor noise
sigma = np.array([0.05, 0.05, 0.1])
sig = np.array([0.1, 0.1])

#change in time
dt = 0

#propagate particles
for i in range(0, len(dataSet)):
    control = dataSet[i][1] # control value

    if (i < len(dataSet) - 1):
        dt = math.fabs(dataSet[i + 1][0] - dataSet[i][0]) #diff betwn 2 consecutive time stamps
    else:
        dt = dt
    #go through each particle in Particle Set and modify it according to the control value - propagate
    for x in range(len(particleSet)):
        theta = particleSet[x][2]
        #changed the theta I am using to propagate particle
        dx = speed * math.cos(theta) * dt
        dy = speed * math.sin(theta) * dt
        dtheta = control * dt

        particleSet[x] = [particleSet[x][0] + dx, particleSet[x][1] + dy, particleSet[x][2] + dtheta]

    newParticles = np.zeros([numParticles, 3])
    for j in range(0, len(particleSet)): #sample a zero mean gaussian
        #resample from a gaussian distribution with the particle as the mean
        newParticles[j] = np.random.normal(loc = particleSet[j], scale = sigma)
    particleSet = newParticles

    #then do sensor evaluation

    #sensor data -> actual landmark dx and dy from robot
    #calculated data -> expected/predicted dx and dy from robot

    sensorAx = dataSet[i][2][0]
    sensorAy = dataSet[i][2][1]
    sensorA = np.zeros([1, 2])
    sensorA[0, :] = [sensorAx, sensorAy]

    sensorBx = dataSet[i][3][0]
    sensorBy = dataSet[i][3][1]
    sensorB = np.zeros([1, 2])
    sensorB[0, :] = [sensorBx, sensorBy]

    sensorCx = dataSet[i][4][0]
    sensorCy = dataSet[i][4][1]
    sensorC = np.zeros([1, 2])
    sensorC[0, :] = [sensorCx, sensorCy]

    sensorDx = dataSet[i][5][0]
    sensorDy = dataSet[i][5][1]
    sensorD = np.zeros([1, 2])
    sensorD[0, :] = [sensorDx, sensorDy]

    # Generate particle weights
    particleWeights = np.zeros(numParticles)

    for j in range(0, len(particleSet)):

        #find dx and dy from particle to landmarks
        A = landmarkA - particleSet[j][:2]
        B = landmarkB - particleSet[j][:2]
        C = landmarkC - particleSet[j][:2]
        D = landmarkD - particleSet[j][:2]

        #weights of the particles
        pwA = mvn.pdf(A, mean = sensorA[0], cov = sig)
        pwB = mvn.pdf(B, mean = sensorB[0], cov = sig)
        pwC = mvn.pdf(C, mean = sensorC[0], cov = sig)
        pwD = mvn.pdf(D, mean = sensorD[0], cov = sig)

        #multiply all the weights so that the overall weight of the particle corrsponds
        #to how close the particle is to the actual location of the robot
        particleWeights[j] = pwB * pwC * pwD # REMOVED pwA

    sum_weights = np.sum(particleWeights)
    particleWeights = particleWeights/sum_weights

    #code for plotting
    index, max_weight = max(enumerate(particleWeights), key=operator.itemgetter(1))
    final_particle = particleSet[index] #take the particle with the max weight

    #back to real code
    newParticles = np.zeros([numParticles, 3])
    particleIndices = range(0, numParticles)  # np.random.choice only selects from 1D arrays - select by index
    selectedIndices = np.random.choice(particleIndices, numParticles, p=particleWeights)

    for j in range(0, len(selectedIndices)):
        newParticles[j] = particleSet[selectedIndices[j]]

    particleSet = newParticles

    '''
    fig = plt.figure(1)
    mSize = 3

    plt.plot(landmarkA[0], landmarkA[1], 's', markersize = 5, color='y')
    plt.plot(landmarkB[0], landmarkB[1], 's', markersize = 5, color='orange')
    plt.plot(landmarkC[0], landmarkC[1], 's', markersize = 5, color='pink')
    plt.plot(landmarkD[0], landmarkD[1], 's', markersize = 5, color='cyan')

    plt.plot(final_particle[0], final_particle[1], 'o', markersize = mSize, color = 'b')
    plt.plot(robot[i][0], robot[i][1], 'o', markersize = mSize, color = 'r')

    plt.show()
    plt.clf()
    '''
    
    fig = plt.figure(1)
    plt.ylim(-4,4)
    plt.xlim(-4,4)
    landmark1 = mpatches.Patch(color = 'black', label = 'landmark 1')
    landmark2 = mpatches.Patch(color = 'pink', label = 'landmark 2')
    landmark3 = mpatches.Patch(color = 'cyan', label = 'landmark 2')
    green = mpatches.Patch(color = 'green', label = 'Initial Dataset')
    yellow = mpatches.Patch(color = 'yellow', label = 'Predicted Robot Position')
    red = mpatches.Patch(color = 'red', label = 'Actual Robot Position')
    
    plt.legend(handles=[green, yellow, red])
    #ax = fig.add_subplot(111, projection='3d')
    mSize = 3
    
    #plt.plot(landmarkA[0], landmarkA[1], 's', markersize = 10, color='blue')
    plt.plot(landmarkB[0], landmarkB[1], 's', markersize = 10, color='black')
    plt.plot(landmarkC[0], landmarkC[1], 's', markersize = 10, color='pink')
    plt.plot(landmarkD[0], landmarkD[1], 's', markersize = 10, color='cyan')

    """
    #robot[i] contains the robot's state from sensor data
    #sensorA is dx and dy of landmark from the sensor data - similarly we have B, C and D
    Ax = robot[i][:2] + sensorA[0]
    Bx = robot[i][:2] + sensorB[0]
    Cx = robot[i][:2] + sensorC[0]
    Dx = robot[i][:2] + sensorD[0]
    
    plt.plot(Ax[0], Ax[1], 's', markersize = 5, color='r') #red (-2, -2) versus yellow (at -2, 2)
    plt.plot(Bx[0], Bx[1], 's', markersize = 5, color='g') #green versus orange - pretty close
    plt.plot(Cx[0], Cx[1], 's', markersize = 5, color='b') #blue versus pink
    plt.plot(Dx[0], Dx[1], 's', markersize = 5, color='black') #black versus cyan
    """

    #plt.plot(final_particle[0], final_particle[1], 'o', markersize = 3 * mSize, color = 'y')
    
    for j in range(0, len(initialSet)):
        plt.plot(particleSet[j,0], particleSet[j, 1], 'o', markersize= 0.75 * mSize, color='y')
    
    #if (i == len(robot) - 1):
    plt.plot(robot[i][0], robot[i][1], 'o', markersize=2 * mSize, color='r')

    if (i < 3):
        for m in range(0, len(initialSet)):
            plt.plot(initialSet[m, 0], initialSet[m, 1], 'o', markersize=0.75 * mSize, color='g')

    #plt.show()
    #plt.clf()


    plt.savefig(str(incr) + '.png')
    incr = incr + 1
    #plt.show()
    plt.clf()




'''
    def robot_callback(self, robot_msg):
    self.robot_data_x = robot_msg.transform.translation.x
    self.robot_data_y = robot_msg.transform.translation.y
    
    self.robot_data_z = robot_msg.transform.translation.z
    self.robot_rotate_x = robot_msg.transform.rotation.x
    self.robot_rotate_y = robot_msg.transform.rotation.y
    self.robot_rotate_z = robot_msg.transform.rotation.z
    self.robot_rotate_w = robot_msg.transform.rotation.w
    
    def obstacle_callback(self, obs_msg):
    self.obstacle_data_x = obs_msg.transform.translation.x
    self.obstacle_data_y = obs_msg.transform.translation.y
    self.obstacle_data_z = obs_msg.transform.translation.z
    
    self.obstacle_rotate_x = obs_msg.transform.rotation.x
    self.obstacle_rotate_y = obs_msg.transform.rotation.y
    self.obstacle_rotate_z = obs_msg.transform.rotation.z
    self.obstacle_rotate_w = obs_msg.transform.rotation.w
    '''
