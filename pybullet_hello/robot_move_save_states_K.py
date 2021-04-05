import csv

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


def try_RobotCell_save_path():
    # TODO: Save step_no, joint positions and velocities of robot and Keypoint locations
    physicsClient = p.connect(p.GUI)

    cube_start_pos = (0.5, 0, 0)  # position of spawned cube
    robot_cell = RobotCell(cube_start_pos)  # start simulation with robot & cube

    goal_pos = (robot_cell.cube_position[0], robot_cell.cube_position[1])  # extract cube xy position
    print("\ngoal_pos = ", goal_pos)

    z_grasp = 0.03 - 0.01
    record = True

    states_1 = robot_cell.move(pos=[0.5, -0.2, 1], instant=False, record=record, save=True)
    states_2 = robot_cell.attempt_grasp(xy=goal_pos, z_grasp=z_grasp, record=record, save=True)
    states_3 = robot_cell.move(pos=[0.5, -0.2, 1], instant=False, record=record, save=True)
    states_4 = robot_cell.move(pos=[0.5, -0.2, z_grasp], instant=False, record=record, save=True)
    states_5 = robot_cell.gripper_open(save=True)
    states_6 = robot_cell.move(pos=[0.5, -0.2, 1], instant=False, record=record, save=True)

    reg_list = [states_1, states_2, states_3, states_4, states_5, states_6]

    # save pickle
    fileObj = open('saved_pickles/states_flat.obj', 'wb')
    pickle.dump(reg_list, fileObj)
    fileObj.close()

    # try to save list to csv
    save_states_to_csv(states_loaded=reg_list)


if __name__ == "__main__":
    # print("pd.__version__", pd.__version__)
    try_RobotCell_save_path()
