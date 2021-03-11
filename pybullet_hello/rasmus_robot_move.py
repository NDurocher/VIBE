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


if __name__ == "__main__":

    # present_the_qs()
    # present_the_move()
    # TODO: separately from robot load the cube
    # TODO: when moving also save the state of the q?

    physicsClient = p.connect(p.GUI)

    cube_pos = (0.5, 0, 0)  # position of spawned cube
    robot_cell = RobotCell(cube_pos)  # start simulation with robot & cube

    goal_pos = (robot_cell.cube_position[0], robot_cell.cube_position[1]) #  extract cube xy position
    print("\ngoal_pos = ", goal_pos)

    z_grasp = 0.03 - 0.01
    record = True

    images_path, qs_path = [], []
    images, qs = robot_cell.attempt_grasp(xy=goal_pos, z_grasp=z_grasp, record=record)
    imgs_qs = robot_cell.move(pos=[0.5, -0.2, 1], instant=False, record=record)
    images_path.append(imgs_qs[0])
    qs_path.append(imgs_qs[1])

    imgs_qs = robot_cell.move(pos=[0.5, -0.2, z_grasp], instant=False, record=record)
    images_path.append(imgs_qs[0])
    qs_path.append(imgs_qs[1])

    imgs_qs = robot_cell.gripper_open()
    images_path.append(imgs_qs[0])
    qs_path.append(imgs_qs[1])

    imgs_qs = robot_cell.move(pos=[0.5, -0.2, 1], instant=False, record=record)
    images_path.append(imgs_qs[0])
    qs_path.append(imgs_qs[1])

    # print(cubePos, cubeOrn)
    p.disconnect()

    # TODO: weird dimensions of the images and qs
    # I was expecting it all in 1 list
    images_path = get_continious(images_path)
    qs_path = get_continious(qs_path)

    # TODO: have to fix saving the data to csv file - problem with the qs
    df_states = pd.DataFrame(list(zip(images_path, qs_path)), columns=['images_path', 'qs_path'])
    df_states.to_csv("path_state.csv", sep='@')

    # print("qs")
    # print(type(qs))
    # print(qs)
    #
    fileObj = open('fixed_imag_q_problem.obj', 'wb')
    pickle.dump({"images_path": images_path, 'qs_path': qs_path}, fileObj)
    fileObj.close()

    # imgs = np.array(images, dtype=np.float32) / 255 - 0.5
    # data = list(zip(imgs, qs))
