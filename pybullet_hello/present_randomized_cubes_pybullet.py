import time

from myRobotCell import *

from pybullet_hello.CNN_Train import CNN
from pybullet_hello.robot_generate_expert_data import *


def present_colourful_cubes():
    positions = get_smart_random_grasp_release_positions(1)
    pick_position, release_loc = positions[0]  # (grasp_x, grasp_y, 0)

    physicsClient = p.connect(p.GUI)
    # physicsClient = p.connect(p.DIRECT)
    p.setTimeStep(0.01)
    p.setGravity(0, 0, -9.8)

    start_tcp_pos = get_random_tcp_start_pos()
    robot_cell = RobotCell((pick_position[0]-100, pick_position[1]-100, 0), (release_loc[0]-100, release_loc[1]-100, 0), 0,
                           start_tcp_pos=start_tcp_pos)  # start simulation with robot & cube

    # find all of the cubes!
    path = os.getcwd()+"/generated_urdfs/"
    found_files = os.listdir(path)
    # print("found_files", found_files)
    box_urdfs = []
    for file in found_files:
        if file.startswith("box_") and file.endswith(".urdf"):
            box_urdfs.append(file)

    print("box_urdfs", box_urdfs)

    # present all of the cubes
    for i in range(len(box_urdfs)):
        chosen_file = path + box_urdfs[i]
        print("chosen_file", chosen_file)
        # exit('test')
        robot_cell.cube_id = p.loadURDF(chosen_file, (-1 + i*0.1 ,0, 0.02), p.getQuaternionFromEuler((0, 0, 0)))
    for i in range(1000):
        p.stepSimulation()

    time.sleep(100)


if __name__ == "__main__":
    present_colourful_cubes()