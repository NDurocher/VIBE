import csv
import math
import os
import random
import time

from myRobotCell import *
import pickle
import pandas as pd

from pybullet_hello.CNN_Train import CNN
from pybullet_hello.robot_generate_expert_data import get_random_tcp_start_pos


def move_robot_with_cnn():
    """ load the model of the CNN and """
    # load cnn module
    Resnet = CNN(False)
    # use it to

    from pybullet_hello.robot_generate_expert_data import get_smart_random_grasp_release_positions
    n_steps_taken = 0

    positions = get_smart_random_grasp_release_positions(1)
    for i in range(len(positions)):
        pick_position, release_loc = positions[i]
        # pick_position = (-0.1, -0.1, 0)
        # release_loc = (0.1, 0.1, 0)
        physicsClient = p.connect(p.GUI)
        # physicsClient = p.connect(p.DIRECT)
        start_tcp_pos = get_random_tcp_start_pos()

        robot_cell = RobotCell(pick_position, release_loc, n_steps_taken,
                               start_tcp_pos=start_tcp_pos)  # start simulation with robot & cube

        pos_acc = 0.018
        no_attempts = 0
        # just take picture and then take action
        take_actions_no = 200
        for i in range(take_actions_no):
            # while True:
            img_now = robot_cell.take_image(view_matrix=p.computeViewMatrix((0, 0, 2), (0, 0, 0), (0, -1, 0)),
                                      projection_matrix=p.computeProjectionMatrixFOV(45, 1, 0.01, 10))
            # plt.imshow(img_now)

            probs = Resnet.RUN(img_now, live=True).cpu()
            print("probabilities of actions :", probs)

            # id_to_action_name = {0: 'e', 1: 'n', 2: 's', 3: 'w'}  # old 4 actions
            id_to_action_name = {0: 'e', 1: 'g', 2: 'n', 3: 'r', 4: 's', 5: 'w'}  # new 6 actions
            sampled_prob_id = int(np.random.choice([0, 1, 2, 3, 4, 5], 1, p=np.array(probs)))
            highest_prob_id = int(np.argmax(probs))
            # print("highest_prob_id = %d  sampled_prob_id = %d" % (highest_prob_id, sampled_prob_id))

            action_to_take = id_to_action_name[highest_prob_id]
            # action_to_take = id_to_action_name[sampled_prob_id]
            print("chosen action: ", action_to_take)
            robot_cell.move_action(action_to_take, save_path=None)
            n_steps_taken += 1

            if action_to_take == 'r':
                break

        p.disconnect()
        time.sleep(0.2)

if __name__ == "__main__":
    print(os.getcwd())
    move_robot_with_cnn()
