import csv

from myRobotCell import *
import pickle
import pandas as pd

def present_the_qs():
    """
    just present why there are 9 qs
    :return:
    """
    physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
    robot_cell = RobotCell(physicsClient, n_legos=20)

    for i in range(100):
        p.stepSimulation()
        # robot_cell.set_q(np.zeros(9))  # it turns out that we have to put 9 qs

        robot_cell.set_q(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]))
        time.sleep(1. / 2.)
        robot_cell.set_q(np.array([1, 0, 0, 0, 0, 0, 0, 0, 0]))
        time.sleep(1. / 2.)
        robot_cell.set_q(np.array([1, 1, 0, 0, 0, 0, 0, 0, 0]))
        time.sleep(1. / 2.)
        robot_cell.set_q(np.array([1, 1, 1, 0, 0, 0, 0, 0, 0]))
        time.sleep(1. / 2.)
        robot_cell.set_q(np.array([1, 1, 1, 1, 0, 0, 0, 0, 0]))
        time.sleep(1. / 2.)
        robot_cell.set_q(np.array([1, 1, 1, 1, 1, 0, 0, 0, 0]))
        time.sleep(1. / 2.)
        robot_cell.set_q(np.array([1, 1, 1, 1, 1, 1, 0, 0, 0]))
        time.sleep(1. / 2.)
        robot_cell.set_q(np.array([1, 1, 1, 1, 1, 1, 1, 0, 0]))
        time.sleep(1. / 2.)
        robot_cell.set_q(np.array([1, 1, 1, 1, 1, 1, 1, 1, 0]))
        time.sleep(1. / 2.)
        robot_cell.set_q(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]))
        time.sleep(1. / 2.)
    p.disconnect()


def present_the_move():
    """
    just present moving the tcp
    :return:
    """
    physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
    robot_cell = RobotCell(physicsClient, n_legos=20)

    for i in range(100):
        p.stepSimulation()
        robot_cell.move(pos=[i % 4, i % 4, 0])
        if i % 4 == 0:
            robot_cell.gripper_close()
        if i % 7 == 0:
            robot_cell.gripper_open()
    p.disconnect()


def test_non_return_RobotCell():
    physicsClient = p.connect(p.GUI)

    cube_pos = (0.5, 0, 0)  # position of spawned cube
    robot_cell = RobotCell_nonrecord(cube_pos)  # start simulation with robot & cube

    goal_pos = (robot_cell.cube_position[0], robot_cell.cube_position[1])  # extract cube xy position
    print("\ngoal_pos = ", goal_pos)

    z_grasp = 0.03 - 0.01
    record = True

    robot_cell.attempt_grasp(xy=goal_pos, z_grasp=z_grasp, record=record)
    robot_cell.move(pos=[0.5, -0.2, 1], instant=False, record=record)

    robot_cell.move(pos=[0.5, -0.2, z_grasp], instant=False, record=record)

    robot_cell.gripper_open()

    robot_cell.move(pos=[0.5, -0.2, 1], instant=False, record=record)

    p.disconnect()


def get_flat_list(my_list):
    flat_list = []
    for item in my_list:
        if(isinstance(item,list)):
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


def test_RobotCell_save_path():
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
    # present_the_qs()
    # present_the_move()
    # test_non_return_RobotCell()  # to run and test if its ok without saving anything - different Class

    test_RobotCell_save_path()



