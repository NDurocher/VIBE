import csv
import os

from myRobotCell import *
import pickle
import pandas as pd


def get_continious(objects_l):
    """ just to fix the list of list of list to finally make it one list to keep track of images and qs per timestep """
    # print(len(objects_l))
    fixed_objs = []
    for obj in objects_l:
        # print(img)
        if obj:
            # print("img = ", img)
            for _obj in obj:
                try:
                    if _obj.any():  # for images
                        # print("_obj = ", _obj)
                        fixed_objs.append(_obj)
                except:
                    if obj:
                        fixed_objs.append(_obj)
                    # pass
    # print(len(fixed_objs))
    return fixed_objs


def get_flat_list(my_list):
    flat_list = []
    for item in my_list:
        if (isinstance(item, list)):
            get_flat_list(item)
        else:
            flat_list.append(item)
    return flat_list


def save_states_to_csv(states_loaded):
    # use pandas to change list of class objects to dataframe and save it
    if type(states_loaded) != list:
        states_loaded = pickle.load(open('saved_pickles/states_flat.obj', 'rb'))
    states_flat = []
    # states_flat = get_flat_list(states_loaded)
    for obj in states_loaded:
        if type(obj) == State:
            states_flat.append(obj)
        else:
            for ob in obj:
                if type(ob) == State:
                    states_flat.append(ob)
                else:
                    for o in ob:
                        if type(o) == State:
                            states_flat.append(o)
                        else:
                            for o_ in o:
                                if type(o_) == State:
                                    states_flat.append(o_)
                                else:
                                    for o__ in o_:
                                        if type(o__) == State:
                                            states_flat.append(o__)
    list_of_dict = []
    for obj in states_flat:
        # print(obj.__dict__)
        list_of_dict.append(obj.__dict__)

    df = pd.DataFrame(list_of_dict)
    df.to_csv("expert_trajectories/test1.csv", sep="$")


def actions_ugly_path():
    physicsClient = p.connect(p.GUI)

    cube_start_pos = (0.5, 0, 0)  # position of spawned cube
    robot_cell = RobotCell_K(cube_start_pos)  # start simulation with robot & cube

    z_grasp = 0.03 - 0.01

    pick_position = (robot_cell.cube_position[0], robot_cell.cube_position[1])
    place_position = (0.3, -0.2)
    print("\npick_position = ", pick_position)
    print("\nplace_position = ", place_position)

    save = False
    record = True

    while robot_cell.world_t_tool().p.copy()[0] <= pick_position[0]:
        robot_cell.move_action('e')
    while robot_cell.world_t_tool().p.copy()[1] <= pick_position[1]:
        robot_cell.move_action('n')
    robot_cell.move_action('g')
    while robot_cell.world_t_tool().p.copy()[1] >= place_position[1]:
        robot_cell.move_action('s')
    robot_cell.move_action('r')


def move_tcp_to_point_grasp_release(robot_cell, goal_position, goal, position_accuracy, save_path):
    """
    moves the TCP to the goal point and returns if the goal was achieved - cube grapsed or released
    goal might be only 'grasp' or 'release'
    """
    if goal not in ('grasp', 'release'):
        raise RuntimeError("only 'grasp' or 'release' are accepted!")

    current_pos = robot_cell.world_t_tool().p
    delta_x = goal_position[0] - current_pos[0]
    delta_y = goal_position[1] - current_pos[1]
    print("delta_x = %f  |  delta_y = %f" % (delta_x, delta_y))
    goal_achieved = False

    # check if can attempt grasp
    if abs(delta_x) < position_accuracy and abs(delta_y) < position_accuracy:

        # can try grasp
        if goal == 'grasp':
            robot_cell.move_action('g', save_path)
            if robot_cell.is_gripper_closed():
                print("grasped!")
                goal_achieved = True
                return goal_achieved

        if goal == 'release':
            robot_cell.move_action('r', save_path)
            if not robot_cell.is_gripper_closed():
                print("released!")
                goal_achieved = True
                return goal_achieved

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
    return goal_achieved


def get_trajectories_actions_pick_place(path_to_save):
    # TODO: record and save the state!
    """
    randomizes the pick and place rotations and returns the expert demonstration
    """
    physicsClient = p.connect(p.GUI)

    cube_start_pos = (0.5, 0, 0)  # position of spawned cube
    robot_cell = RobotCell_K(cube_start_pos)  # start simulation with robot & cube

    pick_position = (robot_cell.cube_position[0], robot_cell.cube_position[1])
    place_position = (0.3, -0.3)

    print("\npick_position = ", pick_position)
    print("\nplace_position = ", place_position)

    pos_acc = 0.03

    positions = []
    positions.append((pick_position, place_position))
    # TODO: positions = get_randomized_pick_place_xy_locations
    for i in range(len(positions)):
        start_pos, end_pos = positions[i]
        print(i, start_pos, end_pos)

        obj_grasped = False
        obj_placed = False
        while not obj_grasped:
            obj_grasped = move_tcp_to_point_grasp_release(robot_cell, goal_position=pick_position, goal='grasp', position_accuracy=pos_acc, save_path=path_to_save)
        while not obj_placed:
            obj_placed = move_tcp_to_point_grasp_release(robot_cell, goal_position=place_position, goal='release', position_accuracy=pos_acc, save_path=path_to_save)


def present_robot_actions():
    """
    sometimes it doesnt work best because of the number of steps taken in each axis
    it seems like if it wants to stretch too much in jiggles in one place
    sometimes it uses curves, probably to dodge the singularities

    Important! Need to unify the dimensions of each step!
    """
    physicsClient = p.connect(p.GUI)

    cube_start_pos = (0.5, 0, 0)  # position of spawned cube
    robot_cell = RobotCell_K(cube_start_pos)  # start simulation with robot & cube

    for i in range(3):
        per_movement = 60
        # go EAST and back
        for j in range(60):
            robot_cell.move_action(action='e')
        for j in range(60):
            robot_cell.move_action(action='w')

        # go WEST and back
        for j in range(60):
            robot_cell.move_action(action='w')
        for j in range(60):
            robot_cell.move_action(action='e')

        # go NORTH and back
        for j in range(40):
            robot_cell.move_action(action='n')
        for j in range(20):
            robot_cell.move_action(action='s')

        for j in range(2):
            robot_cell.move_action(action='g')
            robot_cell.move_action(action='r')


def present_checking_gripper():
    physicsClient = p.connect(p.GUI)

    cube_start_pos = (0.5, 0, 0)  # position of spawned cube
    robot_cell = RobotCell_K(cube_start_pos)  # start simulation with robot & cube

    robot_cell.gripper_close()
    print("gripper closed:", robot_cell.is_gripper_closed())
    print("self.q_target[-1]=", robot_cell.q_target[-1], "self.q_target[-2]=", robot_cell.q_target[-2])

    robot_cell.gripper_open()
    print("gripper opened:", robot_cell.is_gripper_closed())
    print("self.q_target[-1]=", robot_cell.q_target[-1], "self.q_target[-2]=", robot_cell.q_target[-2])


if __name__ == "__main__":
    print(os.getcwd())
    trajectories_save_path = os.getcwd() + "/expert_trajectories/try_0"
    # print("pd.__version__", pd.__version__)
    # actions_ugly_path()
    get_trajectories_actions_pick_place(trajectories_save_path)

    # present_robot_actions()
    # present_checking_gripper()
