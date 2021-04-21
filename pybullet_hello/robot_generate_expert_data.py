import csv
import math
import os
import random
import time

from myRobotCell import *


def move_tcp_to_point_grasp_release(robot_cell, goal_position, goal, position_accuracy, save_path, no_tries):
    """
    moves the TCP to the goal point and returns if the goal was achieved - cube grapsed or released
    goal might be only 'grasp' or 'release'
    """
    if goal not in ('grasp', 'release'):
        raise RuntimeError("only 'grasp' or 'release' are accepted!")

    current_pos = robot_cell.world_t_tool().p
    delta_x = goal_position[0] - current_pos[0]
    delta_y = goal_position[1] - current_pos[1]
    # print("delta_x = %f  |  delta_y = %f" % (delta_x, delta_y))
    dist = math.sqrt( delta_x**2 + delta_y**2)
    print("dist = ", dist)
    goal_achieved = False

    tries_threshold = 30
    if dist < 0.047 and dist >= position_accuracy:
        no_tries += 1
    # check if can attempt grasp
    if dist < position_accuracy or no_tries == tries_threshold:
        if no_tries == tries_threshold:
            print("stuck in place -> try action now!")
            time.sleep(30)
        # can try grasp
        if goal == 'grasp':
            robot_cell.move_action('g', save_path)
            if robot_cell.is_gripper_closed():
                print("grasped!")
                goal_achieved = True
                return goal_achieved, no_tries

        if goal == 'release':
            robot_cell.move_action('r', save_path)
            if not robot_cell.is_gripper_closed():
                print("released!")
                goal_achieved = True
                return goal_achieved, no_tries

    # if have to move right or left!
    if abs(delta_x) > abs(delta_y):
        if delta_x >= 0:
            robot_cell.move_action('e', save_path)
        else:
            robot_cell.move_action('w', save_path)

    else:  # have to move up or down
        if delta_y >= 0:
            robot_cell.move_action('n', save_path)
        else:
            robot_cell.move_action('s', save_path)

    if robot_cell.n_actions_taken % 10 == 0:
        n_size = len(os.listdir(save_path + '/north/.'))
        s_size = len(os.listdir(save_path + '/south/.'))
        e_size = len(os.listdir(save_path + '/east/.'))
        w_size = len(os.listdir(save_path + '/west/.'))
        g_size = len(os.listdir(save_path + '/grasp/.'))
        r_size = len(os.listdir(save_path + '/release/.'))
        print("pictures so far: n=%d, s=%d, e=%d, w=%d, g=%d, r=%d" % ( n_size, s_size, e_size, w_size, g_size, r_size ))

    return goal_achieved, no_tries


def get_smart_random_grasp_release_positions(n_trajectories, th_distance=0.2):
    """
    reachability area:
        x = [-0.2; 0.2]
        y = [-0.15; 0.15]
    randomize grasp place normally
    then randomize release place in the correct distance from the grasp place
    """
    positions = []

    for i in range(n_trajectories):
        grasp_x = round(random.uniform(-0.2, 0.2), 4)
        grasp_y = round(random.uniform(-0.15, 0.15), 4)

        distance = 0
        while distance < th_distance:
            release_x = round(random.uniform(-0.2, 0.2), 4)
            release_y = round(random.uniform(-0.15, 0.15), 4)
            distance = math.sqrt( math.pow(release_x - grasp_x, 2) + math.pow(release_y - grasp_y, 2))
            # print("distance = ", distance)
        positions.append( ( (grasp_x, grasp_y, 0), (release_x, release_y, 0) ) )

    print("Generated %d positions" % len(positions))
    # print(positions)
    return positions


def get_random_grasp_release_positions(n_trajectories):
    """
    grasp save area:
        x = [0; 0.111]
        y = [0; 0.21]

    release save area:
        x = [0.05; 0.28]
        y = [-0.07; 0.07]
    """
    # grasp_loc = (0, 0.2, 0)  # position of spawned cube
    # release_loc = (0.28, 0, 0)
    positions = []

    for i in range(n_trajectories):
        grasp_x = round(random.uniform(0, 0.11), 4)
        grasp_y = round(random.uniform(0, 0.21), 4)
        release_x = round(random.uniform(0.05, 0.28), 4)
        release_y = round(random.uniform(-0.07, 0.07), 4)
        positions.append( ( (grasp_x, grasp_y, 0), (release_x, release_y, 0) ) )

    return positions


def get_trajectories_actions_pick_place():
    # TODO: record and save the state!
    """
    randomizes the pick and place rotations and returns the expert demonstration
    """
    path_to_save = os.getcwd() + "/expert_trajectories/try_smart_fast"
    # positions = get_random_grasp_release_positions(5)
    positions = get_smart_random_grasp_release_positions(1000)
    # exit(1)

    # TODO: positions = get_randomized_pick_place_xy_locations
    for i in range(len(positions)):
        pick_position, release_loc = positions[i]

        # path_to_save += str(i)
        if not os.path.exists(path_to_save):
            os.mkdir(path_to_save)
            if not os.path.exists(path_to_save + '/north'):
                os.mkdir(path_to_save + '/north')
            if not os.path.exists(path_to_save + '/south'):
                os.mkdir(path_to_save + '/south')
            if not os.path.exists(path_to_save + '/east'):
                os.mkdir(path_to_save + '/east')
            if not os.path.exists(path_to_save + '/west'):
                os.mkdir(path_to_save + '/west')
            if not os.path.exists(path_to_save + '/grasp'):
                os.mkdir(path_to_save + '/grasp')
            if not os.path.exists(path_to_save + '/release'):
                os.mkdir(path_to_save + '/release')

        # physicsClient = p.connect(p.GUI)
        physicsClient = p.connect(p.DIRECT)
        robot_cell = RobotCell(pick_position, release_loc)  # start simulation with robot & cube
        print(i, pick_position, release_loc)

        obj_grasped = False
        obj_placed = False

        pos_acc = 0.018
        no_attempts = 0
        while not obj_grasped:
            obj_grasped, no_attempts = move_tcp_to_point_grasp_release(robot_cell, goal_position=pick_position, goal='grasp', position_accuracy=pos_acc, save_path=path_to_save, no_tries=no_attempts)

        no_attempts = 0
        while not obj_placed:
            obj_placed, no_attempts = move_tcp_to_point_grasp_release(robot_cell, goal_position=release_loc, goal='release', position_accuracy=pos_acc, save_path=path_to_save, no_tries=no_attempts)
        robot_cell.reset()
        p.disconnect()
        time.sleep(0.2)


def get_lacking_grasp_release_img():
    path_to_save = os.getcwd() + "/expert_trajectories/grasp_release"
    positions = get_smart_random_grasp_release_positions(700)

    for i in range(len(positions)):
        pick_position, release_loc = positions[i]
        if not os.path.exists(path_to_save):
            os.mkdir(path_to_save)
            if not os.path.exists(path_to_save + '/grasp'):
                os.mkdir(path_to_save + '/grasp')
            if not os.path.exists(path_to_save + '/release'):
                os.mkdir(path_to_save + '/release')

        # physicsClient = p.connect(p.GUI)
        physicsClient = p.connect(p.DIRECT)
        robot_cell = RobotCell(pick_position, release_loc)  # start simulation with robot & cube
        robot_cell.n_actions_taken += i

        robot_cell.move((pick_position[0], pick_position[1], 0.5))
        robot_cell.move(pick_position)
        robot_cell.move_action('g', path_to_save)

        robot_cell.move((release_loc[0], release_loc[1], 0.5))
        robot_cell.move(release_loc)
        robot_cell.move_action('r', path_to_save)

        p.disconnect()
        time.sleep(0.2)


if __name__ == "__main__":
    print(os.getcwd())
    get_trajectories_actions_pick_place()
    # get_lacking_grasp_release_img()
