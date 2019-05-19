#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import math
import pdb

class diffRobotLocalization:
    def __init__(self):
        self.tr = 0.07
        self.r = 0.03
        self.len = 0.1

        # Controller properties
        self.P = 10.0
        self.I = 0.01*0
        self.D = 5.0*0
        self.errorHistory = []
        self.integratedError = 0
        self.errorDot = 0
        # Robot limits
        self.wLim = 1
        self.currentWaypoint = np.zeros(2)
        self.waypointIndex = 0
        self.closeEnough = 0.05
        self.v = 0.5

    def global_to_local(self,q,p):
        TF = np.array([[math.cos(q[2]), math.sin(q[2])],[-math.sin(q[2]), -math.cos(q[2])]])
        globalPoint = np.array([p[0], p[1]])
        return np.dot(TF,globalPoint)

    def dynamics(self,q,control,time):
        theta = q[2]
        xDot = self.v*math.cos(theta)
        yDot = self.v*math.sin(theta)
        thetaDot = control
        qDot = np.array([xDot, yDot, thetaDot])
        return qDot

    def simulate(self,q0,timeSpan,timeStep,waypoints):
        t0 = timeSpan[0]
        tf = timeSpan[1]
        self.waypoints = waypoints
        self.currentWaypoint = waypoints[0]
        timeVals = np.array([float(t0), float(tf)])  # initialize
        self.timeStep = timeStep
        numTimeSteps = (tf-t0)/timeStep
        data, control, time = self.euler_method(q0, [0], self.dynamics, self.PID_controller, timeVals, numTimeSteps)
        return data, control, time

    def PID_controller(self,q,time):
        waypointHeading = self.currentWaypoint - q[0:2]
        if np.linalg.norm(waypointHeading) < self.closeEnough:
            self.waypointIndex = self.waypointIndex + 1
            if self.waypointIndex == len(self.waypoints):
                self.v = 0
                return 0
            else:
                self.currentWaypoint = self.waypoints[self.waypointIndex]
            waypointHeading = self.currentWaypoint - q[0:2]
        print("Current waypoint: " + str(self.currentWaypoint))
        print("State: " + str(q))
        print("Waypoint heading: " + str(waypointHeading))
        angleToWaypoint = math.atan2(waypointHeading[1],waypointHeading[0])
        print("Angle to waypoint: " + str(angleToWaypoint))
        if angleToWaypoint < 0:
            angleToWaypoint = angleToWaypoint + 2*math.pi
        angleDiff = angleToWaypoint - q[2]

        print("Proto angle diff: " + str(angleDiff))

        if angleDiff > math.pi:
            error = abs(math.pi - angleDiff)
        elif angleDiff < -math.pi:
            error = abs(math.pi + angleDiff)
        else:
            error = angleDiff
        print("Angle difference: " + str(angleDiff))

        print("Error: " + str(error))

        if len(self.errorHistory) < 2:
            self.errorDot = 0
        else:
            self.errorDot = (self.errorHistory[-1] - self.errorHistory[-2]) / self.timeStep
        self.errorHistory.append(error)
        self.integratedError = self.integratedError + (self.timeStep * error)
        control = (self.P * error) + (self.I * self.integratedError) + (self.D * self.errorDot)
        if abs(control) > self.wLim:
            control = self.wLim*np.sign(control)

        print("Control: " + str(control) + "\n")
        return control

    def euler_method(self, q0, u0, dfun, ufun, timeVals, numSteps):
        ti = timeVals[0]
        tf = timeVals[1]
        time = np.linspace(ti, tf, numSteps)
        timeStep = (tf - ti) / numSteps

        q = q0  # column vector
        qValues = np.zeros((len(time) + 1, len(q0)), dtype=float)
        qValues[0, :] = q0
        uValues = np.zeros((len(time) + 1, len(u0)), dtype=float)

        for idx, t in enumerate(time):
            # Determine control
            uVec = ufun(q, t)
            uValues[idx,:] = uVec # Control applied at this state
            # Dynamics, integrate
            qDot = dfun(q, uVec, t)
            qn = q + qDot * timeStep
            # Angle wrap
            if qn[2] > 2*math.pi:
                qn[2] = qn[2] - 2*math.pi
            qValues[idx + 1, :] = qn # Store next state value

            # Debug
            #print "Current waypoint: " + str(self.currentWaypoint)
            #print "State: " + str(q)
            #print "Control: " + str(uVec)

            q = qn

        return qValues, uValues, time

    def generate_sensor_readings(self,data,u,time,landmarks,sensorSigma,processSigma,fileName):
        robotName = fileName + "_localizeRobot"
        landmarkName = fileName + "_localizeLandmarks"
        robotFile = open(robotName,'w+')
        landmarkFile = open(landmarkName,'w+')
        sampledStateAggregate = []
        sampledLandmarksAggregate = []
        for i in range(0,len(time)):
            state = data[i]
            control = u[i]
            sampledState = np.random.normal(state,processSigma)
            sampledStateAggregate.append(sampledState)
            robotLocalizationString = str(time[i]) + ", " + str(sampledState) + ", " + str(control)
            if i == len(time):
                landmarkControl = [0,0]
            else:
                landmarkControl = [-(data[i+1][0] - data[i][0]), -(data[i+1][1] - data[i][1])]
            landmarkLocalizationString = str(time[i]) + ", " + str(state) +  ", " + str(landmarkControl)
            for idx,landmark in enumerate(landmarks):
                # Sensors report location in global frame
                globalLandmarkSample = np.random.normal(landmark,sensorSigma)
                robotLocalizationString = robotLocalizationString + ", " + str(globalLandmarkSample)
                sampledLandmarksAggregate.append(globalLandmarkSample)
                #robotLocalizationString = robotLocalizationString + ", " + str(globalLandmarkSample)
                # Landmarks move in a local frame
                # Global to local function here

                landmarkDiff = [landmark[0] - state[0], landmark[1] - state[1]] # global to local conversion
                landmarkDiff = self.global_to_local(state, landmark)
                sampledLandmark = np.random.normal(landmarkDiff,sensorSigma)
                landmarkLocalizationString = landmarkLocalizationString + ", " + str(sampledLandmark)
            robotFile.write(robotLocalizationString + "\n")
            landmarkFile.write(landmarkLocalizationString + "\n")

        robotFile.close()
        landmarkFile.close()

        configFile = open("config.txt","w+")
        configFile.write("Constant speed: " + str(self.v) + "\n")
        configFile.write("Actual landmark locations: \n")
        for idx,landmark in enumerate(landmarks):
            configFile.write("Landmark " + str(idx) + ": " + str(landmark) + "\n")
        configFile.write("\n")
        configFile.write("Robot localization data: time, state(x,y,theta), control input at this state(thetaDot), landmark sensor values (global frame, (x,y), in numerical order) \n")
        configFile.write("Landmark localization data: time, robot state, landmark control (dx,dy - same for all landmarks), sampled landmark state (global frame, (x,y), in numerical order \n")
        configFile.write("Sensor noise values: " + str(sensorSigma) + "\n")
        configFile.write("Processd noise values: " + str(processSigma) + "\n")
        configFile.close()

        return sampledStateAggregate, sampledLandmarksAggregate


    def plotState(self,data,waypoints=None,landmarks=None):
        plt.figure(1)
        if waypoints is not None:
            for waypoint in waypoints:
                plt.plot(waypoint[0], waypoint[1], 's', markersize=17, color='r')
        if landmarks is not None:
            for landmark in landmarks:
                plt.plot(landmark[0], landmark[1], 'd', markersize=17, color='g')
        for dataPoint in data:
            plt.plot(dataPoint[0], dataPoint[1], 'o', color='b')
        plt.axis('equal')
        plt.show()

    def short_line(self,point,theta):
        length = 0.2
        pointEnd = [point[0]+length*math.cos(theta), point[1] + math.sin(theta)]
        return [point,pointEnd]



if __name__ == '__main__':

    # Simulate the robot
    start = np.array([0,0,0])
    timeVals = [0,50]
    timeStep = 0.1

    waypoint1 = [1, 1]
    waypoint2 = [-1,2]
    waypoint3 = [-3, -1]
    waypoint4 = [2, 0]

    waypoints = [waypoint1, waypoint2, waypoint3, waypoint4, waypoint3, waypoint2]

    newRobot = diffRobotLocalization()
    data, control, time = newRobot.simulate(start,timeVals,timeStep,waypoints)

    landmark1 = [-2, -2]
    landmark2 = [0.75, -1.25]
    landmark3 = [-1.1, 0.8]
    landmark4 = [1.5, 2.1]

    landmarks = [landmark1, landmark2, landmark3, landmark4]

    #newRobot.plotState(data,waypoints,landmarks)

    sensorSigma = [0.1, 0.1]
    processSigma = [0.05, 0.05, 0.1]

    # Write data to file
    ssa, sla = newRobot.generate_sensor_readings(data,control,time,landmarks,sensorSigma,processSigma,"diffRobot")

    #newRobot.plotState(ssa, None, sla)
