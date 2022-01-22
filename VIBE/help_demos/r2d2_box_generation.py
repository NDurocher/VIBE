import time
import pickle
from copy import deepcopy
import numpy as np
import pybullet as p
import pybullet_data
import pybullet_utils.bullet_client
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from transform3d import Transform
from tqdm.notebook import tqdm
from IPython.display import HTML, display
import pkgutil
egl = pkgutil.get_loader('eglRenderer')

import torch
from torch import nn
from torch.nn import functional as F

from RobotCell import *
from generate_random_cube import *

if __name__ == '__main__':
    print("hello")

    # robot_cell = RobotCell(n_legos=20)
    DURATION = 10000

    physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # used by loadURDF
    p.setGravity(0, 0, -9.8)
    planeId = p.loadURDF("plane.urdf")
    cubeStartPos = [0, 0, 0.1]
    cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
    boxId = p.loadURDF("r2d2.urdf", cubeStartPos, cubeStartOrientation)
    gemId = p.loadURDF("duck_vhacd.urdf", [2, 2, 1], p.getQuaternionFromEuler([0, 0, 0]))

    # generate and load the boxes
    box_generate_number = 10
    generate_random_urdf_box(save_path='../generated_urdfs/', how_many=box_generate_number)
    for i in range(box_generate_number):
        file = "./generated_urdfs/box_" + str(i) + ".urdf"
        newId = p.loadURDF(file, [0, 0, 0.2 * i], cubeStartOrientation)

    # init positions
    gemPos, gemOrn = p.getBasePositionAndOrientation(gemId)
    cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)

    step_forward = 0.01

    for i in range(DURATION):
        p.stepSimulation()
        time.sleep(1. / 240.)
        gemPos, gemOrn = p.getBasePositionAndOrientation(gemId)

        cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
        delta_pos = (step_forward*np.cos(i*0.01), step_forward*np.sin(i*0.01), 0)
        new_cube_pos = (cubePos[0] + delta_pos[0], cubePos[1] + delta_pos[1], cubePos[2] + delta_pos[2])
        p.resetBasePositionAndOrientation(boxId, new_cube_pos, cubeOrn)

        oid, lk, frac, pos, norm = p.rayTest(cubePos, gemPos)[0]
        # rt = p.rayTest(cubePos, gemPos)
        # print("rayTest: %s" % rt[0][1])
        print("rayTest: Norm: ")
        print(norm)
        p.applyExternalForce(objectUniqueId=boxId, linkIndex=-1, forceObj=pos
                             , posObj=gemPos, flags=p.WORLD_FRAME)
    print(cubePos, cubeOrn)
    p.disconnect()


