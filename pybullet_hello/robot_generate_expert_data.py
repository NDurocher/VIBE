import csv
import math
import os
import random
import time

import numpy as np

from myRobotCell import *


def move_tcp_to_point_grasp_release(robot_cell, goal_position, goal, position_accuracy, save_path, no_tries, tries_threshold=40):
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
    stuck_detected = False

    if dist < 0.07 and dist >= position_accuracy:
        no_tries += 1
    # check if can attempt grasp
    if dist < position_accuracy or no_tries == tries_threshold:
        if no_tries == tries_threshold:
            print("<!!!!!!!!> stuck in place -> try action now!")
            stuck_detected = True
            return goal_achieved, no_tries, stuck_detected
            # time.sleep(30)
        # can try grasp
        if goal == 'grasp':
            robot_cell.move_action('g', save_path)
            if robot_cell.is_gripper_closed():
                print("grasped!")
                goal_achieved = True
                return goal_achieved, no_tries, stuck_detected

        if goal == 'release':
            robot_cell.move_action('r', save_path)
            if not robot_cell.is_gripper_closed():
                print("released!")
                goal_achieved = True
                return goal_achieved, no_tries, stuck_detected

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

    return goal_achieved, no_tries, stuck_detected


def select_action_xy(delta_x, delta_y, sim_factor):
    """ if have to move right or left and check for similar actions """
    # check for similar actions
    similar_ratios = 1 - sim_factor < delta_y / delta_x < 1 + sim_factor

    if abs(delta_x) > abs(delta_y):
        if delta_x >= 0:
            action = 'e'
        else:
            action = 'w'

    else:  # have to move up or down
        if delta_y >= 0:
            action = 'n'
        else:
            action = 's'

    # if similar ratios: we konw the action taken, then we can decide the similar action on the other axis
    similar_action = None
    if similar_ratios:
        if action == "e" or action == "w":
        # similar_action = north or south
            if delta_y >= 0 :
                similar_action = "n"
            else:
                similar_action = "s"
        if action == "n" or action == "s":
            # similar_action = north or south
            if delta_x >= 0:
                similar_action = "e"
            else:
                similar_action = "w"
    return action, similar_action


def move_tcp_to_point_grasp_release_save_similar(robot_cell, goal_position, goal, position_accuracy, try_path, no_tries, n_steps_taken, tries_threshold=40):
    """
    IF THE delta_x IS CLOSE TO THE delta_y THEN SAVE SAME PICTURE TO 2 DIRECTORIES!
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
    stuck_detected = False

    if 0.07 > dist >= position_accuracy:
    # if dist <= position_accuracy:
        no_tries += 1
    # check if can attempt grasp
    if dist < position_accuracy or no_tries == tries_threshold and (goal == 'grasp' or 'release'):
        if no_tries == tries_threshold:
            print("<!!!!!!!!> stuck in place -> try action now!")
            stuck_detected = True
            # exit('stuck in place!')
            return goal_achieved, no_tries, stuck_detected

        # can try grasp
        if goal == 'grasp':
            action = 'g'
            img = robot_cell.move_action_get_img(action)
            if robot_cell.is_gripper_closed():
                print("grasped!")
                goal_achieved = True
                imageio.imwrite(get_save_action_path(try_path, action) + '/' + str(n_steps_taken) + '.jpg',
                                img)
                return goal_achieved, no_tries, stuck_detected
        if goal == 'release':
            action = 'r'
            img = robot_cell.move_action_get_img(action)
            if not robot_cell.is_gripper_closed():
                print("released!")
                goal_achieved = True
                imageio.imwrite(get_save_action_path(try_path, action) + '/' + str(n_steps_taken) + '.jpg',
                                img)
                return goal_achieved, no_tries, stuck_detected

    action, similar_action = select_action_xy(delta_x, delta_y, sim_factor=0.2)

    # take action
    img = robot_cell.move_action_get_img(action)
    # save images to correct dirs!
    imageio.imwrite(get_save_action_path(try_path, action) + '/' + str(n_steps_taken) + '.jpg', img)
    if similar_action:
        # save images to correct dirs!
        # print("action:", action)
        # print("n_actions_taken:", robot_cell.n_actions_taken)
        # print("similar_action:", similar_action)
        # print("saves similar actions!")
        imageio.imwrite(get_save_action_path(try_path, similar_action) + '/' + str(n_steps_taken) + '_s.jpg', img)
        # exit(1)

    # present the state of dirs
    if robot_cell.n_actions_taken % 10 == 0:
        n_size = len(os.listdir(try_path + '/north/.'))
        s_size = len(os.listdir(try_path + '/south/.'))
        e_size = len(os.listdir(try_path + '/east/.'))
        w_size = len(os.listdir(try_path + '/west/.'))
        g_size = len(os.listdir(try_path + '/grasp/.'))
        r_size = len(os.listdir(try_path + '/release/.'))
        print("pictures so far: n=%d, s=%d, e=%d, w=%d, g=%d, r=%d" % ( n_size, s_size, e_size, w_size, g_size, r_size ))

    return goal_achieved, no_tries, stuck_detected


def get_img_move_tcp_to_point_grasp_release(robot_cell, goal_position, goal, position_accuracy, no_tries, n_steps_taken, tries_threshold=40):
    """
    IF THE delta_x IS CLOSE TO THE delta_y THEN SAVE SAME PICTURE TO 2 DIRECTORIES!
    moves the TCP to the goal point and returns if the goal was achieved - cube grapsed or released
    goal might be only 'grasp' or 'release'
    """
    if goal not in ('grasp', 'release'):
        raise RuntimeError("only 'grasp' or 'release' are accepted!")

    current_pos = robot_cell.world_t_tool().p
    delta_x = goal_position[0] - current_pos[0]
    delta_y = goal_position[1] - current_pos[1]

    # print("delta_x = %f  |  delta_y = %f" % (delta_x, delta_y))
    dist = math.sqrt( delta_x**2 + delta_y**2 )
    print("dist = ", dist)
    goal_achieved = False
    stuck_detected = False
    img = None
    similar_action = None

    if 0.07 > dist >= position_accuracy:
    # if dist <= position_accuracy:
        no_tries += 1
    # check if can attempt grasp
    if dist < position_accuracy or no_tries == tries_threshold and (goal == 'grasp' or 'release'):
        if no_tries == tries_threshold:
            print("<!!!!!!!!> stuck in place -> try action now!")
            stuck_detected = True
            # exit('stuck in place!')
            return img, None, similar_action, goal_achieved, no_tries, stuck_detected

        # can try grasp
        if goal == 'grasp':
            action = 'g'
            img = robot_cell.move_action_get_img(action)
            if robot_cell.is_gripper_closed():
                print("grasped!")
                goal_achieved = True
                # imageio.imwrite(get_save_action_path(try_path, action) + '/' + str(n_steps_taken) + '.jpg', img)
                return img, action, similar_action, goal_achieved, no_tries, stuck_detected
        if goal == 'release':
            action = 'r'
            img = robot_cell.move_action_get_img(action)
            if not robot_cell.is_gripper_closed():
                print("released!")
                goal_achieved = True
                # imageio.imwrite(get_save_action_path(try_path, action) + '/' + str(n_steps_taken) + '.jpg', img)
                return img, action, similar_action, goal_achieved, no_tries, stuck_detected

    action, similar_action = select_action_xy(delta_x, delta_y, sim_factor=0.2)

    # take action
    img = robot_cell.move_action_get_img(action)
    # save images to correct dirs!
    # imageio.imwrite(get_save_action_path(try_path, action) + '/' + str(n_steps_taken) + '.jpg', img)
    if similar_action:
        # save images to correct dirs!
        # print("action:", action)
        # print("n_actions_taken:", robot_cell.n_actions_taken)
        # print("similar_action:", similar_action)
        # print("saves similar actions!")
        # imageio.imwrite(get_save_action_path(try_path, similar_action) + '/' + str(n_steps_taken) + '_s.jpg', img)
        pass
        # exit(1)
    return img, action, similar_action, goal_achieved, no_tries, stuck_detected


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


def get_random_tcp_start_pos():
    """ reachability area:
        x = [-0.2; 0.4]
        y = [-0.28; -0.1] """
    z_up = 0.6
    tcp_x = round(random.uniform(-0.2, 0.4), 4)
    tcp_y = round(random.uniform(-0.28, -0.1), 4)

    start_tcp_pos = (tcp_x, tcp_y, z_up)
    # start_tcp_pos = (-0.2, -0.28, z_up)
    return start_tcp_pos


def get_trajectories_actions_pick_place(how_many):
    # TODO: record and save the state!
    """
    randomizes the pick and place rotations and returns the expert demonstration
    """
    path_to_save = os.getcwd() + "/expert_trajectories/try_smart_fast"
    # positions = get_random_grasp_release_positions(5)
    # exit(1)

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
    # exit('dirs')

    # TODO: positions = get_randomized_pick_place_xy_locations
    n_steps_taken = 0

    # for i in range(len(positions)):
    trajs_generated = 0

    while trajs_generated != how_many:
        positions = get_smart_random_grasp_release_positions(1)

        pick_position, release_loc = positions[0]
        physicsClient = p.connect(p.GUI)
        # physicsClient = p.connect(p.DIRECT)

        start_tcp_pos = get_random_tcp_start_pos()

        robot_cell = RobotCell(pick_position, release_loc, n_steps_taken, start_tcp_pos=start_tcp_pos)  # start simulation with robot & cube
        print(trajs_generated, pick_position, release_loc)

        stuck_detected = False
        obj_grasped = False
        obj_placed = False

        pos_acc = 0.018  # for grasp / release
        tries_threshold = 24  # to get out of stuck position
        no_attempts = 0
        imgs_trajectory_l = []

        while not obj_grasped and not stuck_detected:
            img, action, similar_action, obj_grasped, no_attempts, stuck_detected = get_img_move_tcp_to_point_grasp_release(robot_cell, goal_position=pick_position, goal='grasp', position_accuracy=pos_acc, no_tries=no_attempts, tries_threshold=tries_threshold, n_steps_taken=n_steps_taken)
            n_steps_taken += 1
            imgs_trajectory_l.append({'img': img, 'action': action, 'similar_action': similar_action})

        no_attempts = 0
        while not obj_placed and obj_grasped and not stuck_detected:
            img, action, similar_action, obj_placed, no_attempts, stuck_detected = get_img_move_tcp_to_point_grasp_release(robot_cell, goal_position=release_loc, goal='release', position_accuracy=pos_acc, no_tries=no_attempts, tries_threshold=tries_threshold, n_steps_taken=n_steps_taken)
            n_steps_taken += 1
            imgs_trajectory_l.append({'img': img, 'action': action, 'similar_action': similar_action})

            # check for the dropping of the cube
            if action != "r":
                if p.getBasePositionAndOrientation(robot_cell.cube_id)[0][2] < robot_cell.world_t_tool().p[2]/2:
                    stuck_detected = True
                    # while True:
                    #     print("TEST THE GRASPING")
                    #     time.sleep(100)

        if not stuck_detected and obj_grasped and obj_placed:
            # save the imgs to dirs given the list
            for traj_step, img_info in enumerate(imgs_trajectory_l):
                imageio.imwrite(get_save_action_path(path_to_save, img_info['action']) + '/' + str(traj_step) + '_' + str(trajs_generated) + '.jpg', img_info['img'])
                if img_info['similar_action']:
                    imageio.imwrite(get_save_action_path(path_to_save, img_info['similar_action']) + '/' + str(traj_step) + '_similar.jpg', img_info['img'])
            trajs_generated += 1

        print("================= trajs_generated = ", trajs_generated)
        # present the state of dirs
        n_size = len(os.listdir(path_to_save + '/north/.'))
        s_size = len(os.listdir(path_to_save + '/south/.'))
        e_size = len(os.listdir(path_to_save + '/east/.'))
        w_size = len(os.listdir(path_to_save + '/west/.'))
        g_size = len(os.listdir(path_to_save + '/grasp/.'))
        r_size = len(os.listdir(path_to_save + '/release/.'))
        print("pictures so far: n=%d, s=%d, e=%d, w=%d, g=%d, r=%d" % (n_size, s_size, e_size, w_size, g_size, r_size))

        # exit("see what about stucking")

        robot_cell.reset()
        p.disconnect()
        time.sleep(0.2)


def get_lacking_grasp_release_img(how_many):
    path_to_save = os.getcwd() + "/expert_trajectories/grasp_release"
    positions = get_smart_random_grasp_release_positions(how_many)

    for i in range(len(positions)):
        print(" ======= progress: %d out of %d" % (i, len(positions)))

        pick_position, release_loc = positions[i]
        if not os.path.exists(path_to_save):
            os.mkdir(path_to_save)
            if not os.path.exists(path_to_save + '/grasp'):
                os.mkdir(path_to_save + '/grasp')
            if not os.path.exists(path_to_save + '/release'):
                os.mkdir(path_to_save + '/release')

        # physicsClient = p.connect(p.GUI)
        physicsClient = p.connect(p.DIRECT)

        start_tcp_pos = get_random_tcp_start_pos()

        robot_cell = RobotCell(pick_position, release_loc, i,
                               start_tcp_pos=start_tcp_pos)  # start simulation with robot & cube
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
    get_trajectories_actions_pick_place(150)
    # get_lacking_grasp_release_img(2000)
