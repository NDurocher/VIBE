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
    cube_start_pos = (0.5, 0, 0)  # position of spawned cube
    robot_cell = RobotCell(cube_start_pos)  # start simulation with robot & cube

    goal_pos = (robot_cell.cube_position[0], robot_cell.cube_position[1])  # extract cube xy position
    print("\ngoal_pos = ", goal_pos)

    z_grasp = 0.03 - 0.01
    record = True

    robot_cell.move(pos=[0.5, -0.2, 1], instant=False, record=record, save=True)
    robot_cell.attempt_grasp(xy=goal_pos, z_grasp=z_grasp, record=record, save=True)
    robot_cell.move(pos=[0.5, -0.2, 1], instant=False, record=record, save=True)
    robot_cell.move(pos=[0.5, -0.2, z_grasp], instant=False, record=record, save=True)
    robot_cell.gripper_open(save=True)
    robot_cell.move(pos=[0.5, -0.2, 1], instant=False, record=record, save=True)
    p.disconnect()


if __name__ == "__main__":

    # print("pd.__version__", pd.__version__)
    # present_the_qs()
    present_the_move()
