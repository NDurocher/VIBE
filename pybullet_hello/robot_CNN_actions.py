import csv
import math
import os
import random
import time

from myRobotCell import *
import pickle
import pandas as pd

from pybullet_hello.CNN_Train import CNN


def move_robot_with_cnn():
    """ load the model of the CNN and """
    # load cnn module
    Resnet = CNN(False).cpu()
    # use it to

    from pybullet_hello.robot_generate_expert_data import get_smart_random_grasp_release_positions
    positions = get_smart_random_grasp_release_positions(1)
    for i in range(len(positions)):
        pick_position, release_loc = positions[i]
        physicsClient = p.connect(p.GUI)
        # physicsClient = p.connect(p.DIRECT)
        robot_cell = RobotCell(pick_position, release_loc)  # start simulation with robot & cube
        print(i, pick_position, release_loc)

        pos_acc = 0.018
        no_attempts = 0
        # just take picture and then take action
        take_actions_no = 100
        for i in range(take_actions_no):
            # while True:
            img_now = robot_cell.take_image(view_matrix=p.computeViewMatrix((0, 0, 2), (0, 0, 0), (0, -1, 0)),
                                      projection_matrix=p.computeProjectionMatrixFOV(45, 1, 0.01, 10))
            probs = Resnet.RUN(img_now, live=True)
            id_to_action_name = {0: 'e', 1: 'n', 2: 's', 3: 'w'}
            action_to_take = id_to_action_name[int(np.argmax(probs))]
            robot_cell.move_action(action_to_take, save_path=None)
            if action_to_take == 'r':
                break

        p.disconnect()
        time.sleep(0.2)

if __name__ == "__main__":
    print(os.getcwd())
    move_robot_with_cnn()
