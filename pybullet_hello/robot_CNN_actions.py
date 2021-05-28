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
            img_now = robot_cell.take_top_image()
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


def is_tcp_in_good_release(current_pos, release_loc):
    """
    release save area:
        x = [0.05; 0.28]
        y = [-0.07; 0.07]

    <box size="0.25 0.25 0"/>
    """
    x_pos = current_pos[0]
    y_pos = current_pos[1]
    if release_loc[0]-0.25/2 <= x_pos <= release_loc[0]+0.25/2 and release_loc[1]-0.25/2 <= y_pos <= release_loc[1]+0.25/2:
        return True
    return False


def performance_robot_with_cnn(do_gui, no_tries, model_name, demonstrate=False, random_colours=False):
    """ load the model of the CNN and """
    # load cnn module
    Resnet = CNN(False, model_name)

    from pybullet_hello.robot_generate_expert_data import get_smart_random_grasp_release_positions
    n_steps_taken = 0

    success_no = 0
    start_tcp_pos = (0, -0.1, 0.6)  # get_random_tcp_start_pos()
    positions = get_smart_random_grasp_release_positions(no_tries)

    for i in range(1, len(positions)):
        pick_position, release_loc = positions[i]
        if do_gui or demonstrate:
            physicsClient = p.connect(p.GUI)
        else:
            physicsClient = p.connect(p.DIRECT)

        robot_cell = RobotCell(pick_position, release_loc, n_steps_taken,
                               start_tcp_pos=start_tcp_pos, random_box=random_colours)  # start simulation with robot & cube

        # second_cube_pos = (robot_cell.cube_position[1]+0.2, robot_cell.cube_position[0]-0.2, 0)
        # second_cube_id = p.loadURDF(os.getcwd()+"/generated_urdfs/box_example_2.urdf", second_cube_pos, p.getQuaternionFromEuler((0, 0, 0)))

        goal_achieved = False
        tried_grasping = False
        actions_taken = 0
        winrate = 0
        last_few_actions = []
        while not goal_achieved:
            # just take picture and then take action

            img_now = robot_cell.take_top_image()
            # plt.imshow(img_now)

            probs = Resnet.RUN(img_now, live=True).cpu()
            # print("probabilities of actions :", probs)
            id_to_action_name = {0: 'e', 1: 'g', 2: 'n', 3: 'r', 4: 's', 5: 'w'}  # new 6 actions
            sampled_prob_id = int(np.random.choice([0, 1, 2, 3, 4, 5], 1, p=np.array(probs)))
            highest_prob_id = int(np.argmax(probs))
            # print("highest_prob_id = %d  sampled_prob_id = %d" % (highest_prob_id, sampled_prob_id))

            action_to_take = id_to_action_name[highest_prob_id]
            # action_to_take = id_to_action_name[sampled_prob_id]
            # print("chosen action: ", action_to_take)
            robot_cell.move_action(action_to_take, save_path=None)
            n_steps_taken += 1
            # print("current_pos = ", robot_cell.world_t_tool().p)
            # print("release_loc = ", release_loc)
            actions_taken += 1

            """ keep track of repeated grasps! """
            detected_stuck_on_grasp = False
            if actions_taken < 7:
                last_few_actions.append(action_to_take)
            else:
                last_few_actions.pop(0)
                last_few_actions.append(action_to_take)
            if len(set(last_few_actions)) == 1 and last_few_actions[0] == 'g':
                detected_stuck_on_grasp = True
                print('detected stuck on grasping!')

            if action_to_take == 'r':
                # check if the release was in the correct area - if yes then
                current_pos = robot_cell.world_t_tool().p
                if is_tcp_in_good_release(current_pos, release_loc):
                    if not demonstrate:
                        goal_achieved = True
                        success_no += 1
                        p.disconnect()
                        time.sleep(2)
                        # exit("testing")

            winrate = success_no / (i+1)
            print("model = %s | traj_no = %d out of %d <winrate = %f> | actions_taken = %d  last action=%s" %
                  (model, i, no_tries, winrate, actions_taken, action_to_take))
            tried_grasping = tried_grasping or action_to_take == 'g'

            if actions_taken >= 80 and not tried_grasping or detected_stuck_on_grasp or actions_taken >= 150:
                print("robot stuck - didnt achieve the goal")
                if not demonstrate:
                    goal_achieved = True
                    p.disconnect()
                    time.sleep(2)

        # print("========== try=%d winrate = %f  ==========" % (i, winrate))
    return winrate


if __name__ == "__main__":
    print(os.getcwd())
    # move_robot_with_cnn()
    performance_l = []
    no_tries = 100

    model_names = (
        # 'natural_p50.pth',
        # 'natural_p50_without_sim.pth',
        # 'natural_p100.pth',
        # 'natural_p100_without_sim.pth',
        # 'natural_p150.pth',
        # 'natural_p150_without_sim.pth',
        # 'natural_p200.pth',
        # 'natural_p200_without_sim.pth',
        # 'natural_p250.pth',
        # 'natural_p250_2.pth',
        # 'natural_p250_without_sim.pth',

        'colouredcubes.pth',

    )

    for model in model_names:
        # winrate = performance_robot_with_cnn(do_gui=True, no_tries=no_tries, model_name=model, demonstrate=True)
        winrate = performance_robot_with_cnn(do_gui=False, no_tries=no_tries, model_name=model, demonstrate=False, random_colours=True)
        # performance_robot_with_cnn(do_gui=False, no_tries=100, model_name=model_names[0])
        results_now = {"model": model, "no_tries": no_tries, "winrate": winrate}
        performance_l.append(results_now)
        print("<!> results_now = \n", results_now)
        df = pd.DataFrame(performance_l)
        df.to_csv("permormance_models.csv")

    print("performance_l = \n", performance_l)
