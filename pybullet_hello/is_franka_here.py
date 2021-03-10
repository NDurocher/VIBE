"""
source from
https://www.etedal.net/2020/04/pybullet-panda.html
"""

import os
import pybullet as p
import pybullet_data

if __name__ == "__main__":
    p.connect(p.GUI)
    pandaUid = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "franka_panda/panda.urdf"),useFixedBase=True)

    while True:
        p.stepSimulation()
