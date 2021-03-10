from myRobotCell import *


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
        if i% 4 == 0:
            robot_cell.gripper_close()
        if i % 7 == 0:
            robot_cell.gripper_open()
    p.disconnect()


if __name__ == "__main__":

    # present_the_qs()
    # present_the_move()

    physicsClient = p.connect(p.GUI)

    robot_cell = RobotCell(physicsClient, n_legos=20)

    height_of_block = 0.03 - 0.01

    goal_pos = robot_cell.cube_xy
    print("\ngoal_pos = ", goal_pos)
    success, images = robot_cell.attempt_grasp(xy=goal_pos, z_grasp=height_of_block, record=True)
    robot_cell.move(pos=[0.5, -0.2, 1], instant=False, record=True)
    robot_cell.move(pos=[0.5, -0.2, height_of_block], instant=False, record=True)
    robot_cell.gripper_open()
    robot_cell.move(pos=[0.5, -0.2, 1], instant=False, record=True)


    # print(cubePos, cubeOrn)
    p.disconnect()