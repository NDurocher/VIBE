from myRobotCell import *
from matplotlib import pyplot as plt

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


def check_cameras():
    """
    play with the cameras
    I found this function take_top_image in the myRobotCell.py. so I used the same view matrix.
    Maybe play with the FoV to fit the whole scene.
    """
    physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
    cube_start_pos = (0.5, 0, 0)  # position of spawned cube
    robot_cell = RobotCell(cube_start_pos)  # start simulation with robot & cube
    #img = robot_cell.take_image(view_matrix=p.computeViewMatrix((0.5, 1, 1), (0, 0, 0.3), (0, 0, 1)),
    #                            projection_matrix=p.computeProjectionMatrixFOV(30, 1, 0.01, 10))
    img = robot_cell.take_image(view_matrix=p.computeViewMatrix((0, 0, 2), (0, 0, 0), (0, -1, 0)),
    projection_matrix = p.computeProjectionMatrixFOV(45, 1, 0.01, 10))

    plt.imshow(img)
    plt.show()
    p.disconnect()


def present_the_move():
    """
    just present moving the tcp with pics from the camera above. Make sure you have Sci View enabled on Pycharm.
    You can look at Pictures folder for sample pictures
    :return:
    """
    physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
    cube_start_pos = (0, 0.2, 0)  # position of spawned cube
    robot_cell = RobotCell(cube_start_pos)  # start simulation with robot & cube

    goal_pos = (robot_cell.cube_position[0], robot_cell.cube_position[1])  # extract cube xy position
    print("\ngoal_pos = ", goal_pos)

    z_grasp = 0.04
    record = True

    robot_cell.move(pos=[0, 0.2, 1], instant=False, record=record, save=True)
    img = robot_cell.take_image(view_matrix=p.computeViewMatrix((0, 0, 2), (0, 0, 0), (0, -1, 0)),
                          projection_matrix=p.computeProjectionMatrixFOV(45, 1, 0.01, 10))
    plt.imshow(img)
    plt.show()

    robot_cell.attempt_grasp(xy=goal_pos, z_grasp=z_grasp, record=record, save=True)
    img = robot_cell.take_image(view_matrix=p.computeViewMatrix((0, 0, 2), (0, 0, 0), (0, -1, 0)),
                          projection_matrix=p.computeProjectionMatrixFOV(45, 1, 0.01, 10))
    plt.imshow(img)
    plt.show()

    robot_cell.move(pos=[0, 0, 0.5], instant=False, record=record, save=True)
    img = robot_cell.take_image(view_matrix=p.computeViewMatrix((0, 0, 2), (0, 0, 0), (0, -1, 0)),
                          projection_matrix=p.computeProjectionMatrixFOV(55, 1, 0.01, 10))
    plt.imshow(img)
    plt.show()

    robot_cell.move(pos=[0.5, 0, z_grasp], instant=False, record=record, save=True)
    img = robot_cell.take_image(view_matrix=p.computeViewMatrix((0, 0, 2), (0, 0, 0), (0, -1, 0)),
                          projection_matrix=p.computeProjectionMatrixFOV(45, 1, 0.01, 10))
    plt.imshow(img)
    plt.show()

    robot_cell.gripper_open(save=True)
    img = robot_cell.take_image(view_matrix=p.computeViewMatrix((0, 0, 2), (0, 0, 0), (0, -1, 0)),
                          projection_matrix=p.computeProjectionMatrixFOV(45, 1, 0.01, 10))
    plt.imshow(img)
    plt.show()

    robot_cell.move(pos=[0.5, 0, 1], instant=False, record=record, save=True)

    p.disconnect()


if __name__ == "__main__": # Uncomment the functions you need

    # print("pd.__version__", pd.__version__)
    #present_the_qs()
    check_cameras()
    present_the_move()

