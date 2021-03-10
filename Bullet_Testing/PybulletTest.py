# import pybullet as p
# import time
# import pybullet_data
#
# physicsClient = p.connect(p.GUI)# p.DIRECT for non-graphical version
#
# p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
# p.setGravity(0,0,-10)
# planeId = p.loadURDF("plane.urdf")
# cubeStartPos = [0,0,1]
# cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
# boxId = p.loadURDF("r2d2.urdf",cubeStartPos, cubeStartOrientation)
#
# for i in range (10000):
#    p.stepSimulation()
#    time.sleep(1./240.)
#    cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
# print(cubePos,cubeOrn)
#
# p.disconnect()

import pybullet as p
import time
import math
import pybullet_data
import torch

from torch import nn
from torch.nn import functional as F
from datetime import datetime
#clid = p.connect(p.SHARED_MEMORY)

class SmallUnet(nn.Module):
    # similar to unet, see fig 1: https://arxiv.org/pdf/1505.04597.pdf
    def __init__(self, features=(3, 16, 32, 64, 128)):
        super().__init__()
        self.features = features
        self.l = []
        self.r = []

        for fi, fo in zip(features, features[1:]):
            self.l.append(conv_block(fi, fo))
        for fi, fo in zip(features[1:], features[2:]):
            self.r.append(conv_block(fo + fi, fi))

        self.pool = nn.MaxPool2d(2, 2)
        self.upscale = nn.UpsamplingBilinear2d(scale_factor=2)

        # register the modules
        self._l = nn.Sequential(*self.l)
        self._r = nn.Sequential(*self.r)

    def forward(self, x):
        fl = []
        for l in self.l[:-1]:
            x = l(x)
            fl.append(x)
            x = self.pool(x)
        x = self.l[-1](x)
        for r, f in zip(self.r[::-1], fl[::-1]):
            x = self.upscale(x)
            x = torch.cat((x, f), 1)
            x = r(x)
        return x




p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf", [0, 0, -0.3], useFixedBase=True)
kukaId = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)
p.resetBasePositionAndOrientation(kukaId, [0, 0, 0], [0, 0, 0, 1])
kukaEndEffectorIndex = 6
numJoints = p.getNumJoints(kukaId)
# PandaId = p.loadURDF("franka_panda/panda.urdf", [1, 0, 0], useFixedBase=True)


viewMatrix = p.computeViewMatrix(
    cameraEyePosition=[-1, 0, 0.5],
    cameraTargetPosition=[-0.3, 0, 0],
    cameraUpVector=[0, 1, 0])
projectionMatrix = p.computeProjectionMatrixFOV(
    fov=45.0,
    aspect=1.0,
    nearVal=0.1,
    farVal=1.1)


Takeimage = p.addUserDebugParameter("Capture RGB", 1, 0, 0)
capture = p.readUserDebugParameter(Takeimage)
prev = 1

if (numJoints != 7):
  exit()

# p.loadURDF("cube_small.urdf", [2, 2, 5])
# p.loadURDF("cube_small.urdf", [-2, -2, 5])
p.loadURDF("lego/lego.urdf", [-0.5, 0, 5])

#lower limits for null space
ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
#upper limits for null space
ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
#joint ranges for null space
jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]
#restposes for null space
rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
#joint damping coefficents
jd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

for i in range(numJoints):
  p.resetJointState(kukaId, i, rp[i])

p.setGravity(0, 0, -9.81)
t = 0.
prevPose = [0, 0, 0]
prevPose1 = [0, 0, 0]
hasPrevPose = 0
useNullSpace = 0

count = 0
useOrientation = 1
useSimulation = 1
useRealTimeSimulation = 1
p.setRealTimeSimulation(useRealTimeSimulation)
#trailDuration is duration (in seconds) after debug lines will be removed automatically
#use 0 for no-removal
trailDuration = 15

logId1 = p.startStateLogging(p.STATE_LOGGING_GENERIC_ROBOT, "LOG0001.txt", [0, 1, 2])
logId2 = p.startStateLogging(p.STATE_LOGGING_CONTACT_POINTS, "LOG0002.txt", bodyUniqueIdA=2)

# for i in range(5):
#   print("Body %d's name is %s." % (i, p.getBodyInfo(i)[1]))

while 1:
  if (useRealTimeSimulation):
    dt = datetime.now()
    t = (dt.second / 60.) * 2. * math.pi
  else:
    t = t + 0.1

  capture = p.readUserDebugParameter(Takeimage)
  if capture != prev:
    width, height, rgbImg, depthImg, segImg = p.getCameraImage(
        width=224,
        height=224,
        viewMatrix=viewMatrix,
        projectionMatrix=projectionMatrix)
    prev = capture
    print(capture)

  if (useSimulation and useRealTimeSimulation == 0):
    p.stepSimulation()

  for i in range(1):
    pos = [-0.4, 0.2 * math.cos(t), 0. + 0.2 * math.sin(t)]
    #end effector points down, not up (in case useOrientation==1)
    orn = p.getQuaternionFromEuler([0, -math.pi, 0])

    if (useNullSpace == 1):
      if (useOrientation == 1):
        jointPoses = p.calculateInverseKinematics(kukaId, kukaEndEffectorIndex, pos, orn, ll, ul,
                                                  jr, rp)
      else:
        jointPoses = p.calculateInverseKinematics(kukaId,
                                                  kukaEndEffectorIndex,
                                                  pos,
                                                  lowerLimits=ll,
                                                  upperLimits=ul,
                                                  jointRanges=jr,
                                                  restPoses=rp)
    else:
      if (useOrientation == 1):
        jointPoses = p.calculateInverseKinematics(kukaId,
                                                  kukaEndEffectorIndex,
                                                  pos,
                                                  orn,
                                                  jointDamping=jd)
      else:
        jointPoses = p.calculateInverseKinematics(kukaId, kukaEndEffectorIndex, pos)

    if (useSimulation):
      for i in range(numJoints):
        p.setJointMotorControl2(bodyIndex=kukaId,
                                jointIndex=i,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=jointPoses[i],
                                targetVelocity=0,
                                force=500,
                                positionGain=0.03,
                                velocityGain=1)
    else:
      #reset the joint state (ignoring all dynamics, not recommended to use during simulation)
      for i in range(numJoints):
        p.resetJointState(kukaId, i, jointPoses[i])

  ls = p.getLinkState(kukaId, kukaEndEffectorIndex)
  if (hasPrevPose):
    p.addUserDebugLine(prevPose, pos, [0, 0, 0.3], 1, trailDuration)
    p.addUserDebugLine(prevPose1, ls[4], [1, 0, 0], 1, trailDuration)
  prevPose = pos
  prevPose1 = ls[4]
  hasPrevPose = 1
