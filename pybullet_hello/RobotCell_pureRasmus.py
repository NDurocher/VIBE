from pybullet_hello.help_demos.r2d2_box_generation import *

class RobotCell:
    def __init__(self, n_legos, dt=0.01):

        self.dt = dt
        self.p = pybullet_utils.bullet_client.BulletClient(p.DIRECT)
        self.p.setTimeStep(dt)
        self.p.setGravity(0, 0, -9.8)
        self.p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")

        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.p.loadURDF("plane.urdf")
        self.legos = [self.p.loadURDF("lego/lego.urdf") for i in range(n_legos)]
        self.n_grasped = 0
        self.rid = self.p.loadURDF(
            "franka_panda/panda.urdf",
            (0, -0.55, 0), self.p.getQuaternionFromEuler((0, 0, np.pi / 2)),
            useFixedBase=True
        )

        joint_infos = [self.p.getJointInfo(self.rid, i) for i in range(self.p.getNumJoints(self.rid))]
        joint_infos = [ji for ji in joint_infos if ji[2] is not p.JOINT_FIXED]
        self.joint_idxs = [ji[0] for ji in joint_infos]
        self.joint_lower_limits = np.array([ji[8] for ji in joint_infos])
        self.joint_upper_limits = np.array([ji[9] for ji in joint_infos])
        self.joint_lower_limits[6] = -np.pi * 2
        self.joint_upper_limits[6] = np.pi * 2
        self.joint_range = self.joint_upper_limits - self.joint_lower_limits
        self.joint_rest = (self.joint_lower_limits + self.joint_upper_limits) / 2
        self.joint_rest[-2:] = 0.04
        self.joint_rest[1] -= np.pi / 4
        self.q_target = np.zeros(9)

        for i, joint_idx in enumerate(self.joint_idxs):
            self.p.changeDynamics(self.rid, joint_idx, linearDamping=0, angularDamping=0)

        self.reset()

    def set_q(self, q):
        for i, joint_idx in enumerate(self.joint_idxs):
            self.p.resetJointState(self.rid, joint_idx, q[i])
        self.set_q_target(q)

    def set_q_target(self, q):
        self.q_target = q
        self.p.setJointMotorControlArray(
            self.rid, self.joint_idxs, p.POSITION_CONTROL, q,
            forces=[100] * 7 + [15] * 2, positionGains=[1] * 9
        )

    def reset_robot(self):
        self.set_q(self.joint_rest)

    def reset(self):
        self.reset_robot()
        for lego in self.legos:
            pos = *np.random.uniform(-0.1, 0.1, 2), 0.03
            quat = np.random.randn(4)
            quat = quat / np.linalg.norm(quat)
            self.p.resetBasePositionAndOrientation(lego, pos, quat)
            self.p.changeDynamics(lego, -1, mass=0.1)
        self.n_grasped = 0
        for i in range(50):  # let them fall to rest on the table
            self.p.stepSimulation()

    def take_image(self, res=400,
                   view_matrix=p.computeViewMatrix((0.5, 1, 1), (0, 0, 0.3), (0, 0, 1)),
                   projection_matrix=p.computeProjectionMatrixFOV(30, 1, 0.01, 10)):
        img, depth, seg = self.p.getCameraImage(
            res, res, view_matrix, projection_matrix,
            shadow=1,
        )[2:]
        return img[..., :3]

    def take_top_image(self, res=224, fov=8):
        view_matrix = p.computeViewMatrix((0, 0, 2), (0, 0, 0), (0, -1, 0))
        projection_matrix = p.computeProjectionMatrixFOV(fov, 1, 0.01, 10)
        return self.take_image(res, view_matrix, projection_matrix)

    def top_px_to_world(self, x, y, z=0.01, res=224, fov=8):
        depth = 2 - z
        p = np.array((x, y)) / res - 0.5  # from -.5 to .5
        width_view = np.tan(np.deg2rad(fov) / 2) * depth * 2
        p = p * width_view * (-1, 1)
        return p

    def show(self, img=None):
        plt.Figure(figsize=(5, 5))
        plt.imshow(self.take_image() if img is None else img)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def show_top(self):
        self.show(self.take_top_image())

    def ik(self, world_t_tool: Transform, max_iter=100):
        q = list(self.p.calculateInverseKinematics(
            self.rid, 11, world_t_tool.p, world_t_tool.quat,
            # list(self.joint_lower_limits), list(self.joint_upper_limits),
            # list(self.joint_range), list(self.joint_rest),
            maxNumIterations=max_iter
        ))
        q[-1] = q[-2] = self.q_target[-1]
        return q

    def world_t_tool(self):
        pos, quat = self.p.getLinkState(self.rid, 11)[:2]
        return Transform(p=pos, quat=quat)

    def move(self, pos, theta=0., speed=2., acc=10., instant=False, record=False):
        world_t_tool_desired = Transform(p=pos, rpy=(0, np.pi, np.pi / 2 + theta))
        if instant:
            self.set_q(self.ik(world_t_tool_desired))
        else:
            world_t_tool_start = self.world_t_tool()
            tool_start_t_tool_desired = world_t_tool_start.inv @ world_t_tool_desired
            dist = np.linalg.norm(tool_start_t_tool_desired.xyz_rotvec)
            images = []
            for i, s in enumerate(lerp(dist, speed, acc, self.dt)):
                world_t_tool_target = world_t_tool_start @ (tool_start_t_tool_desired * s)
                self.set_q_target(self.ik(world_t_tool_target))
                self.p.stepSimulation()
                if record and i % 4 == 0:  # fps = 1/dt / 4
                    images.append(self.take_image())
            return images

    def gripper_move(self, d_desired, record=False, speed=1., acc=3.):
        d_start = self.q_target[-1]
        move = d_desired - d_start
        images = []
        for i, s in enumerate(lerp(abs(move), speed, acc, self.dt)):
            self.q_target[-1] = self.q_target[-2] = d_start + s * move
            self.set_q_target(self.q_target)
            self.p.stepSimulation()
            if record and i % 4 == 0:
                images.append(self.take_image())
        return images

    def gripper_close(self, record=False):
        return self.gripper_move(0, record)

    def gripper_open(self, record=False):
        return self.gripper_move(0.03, record)

    def attempt_grasp(self, xy=(0, 0), theta=0, z_grasp=0.01, z_up=0.1, record=False):
        self.q_target[-1] = self.q_target[-2] = 0.03
        self.move((*xy, z_up), theta, instant=True)
        images = []
        images += self.move((*xy, z_grasp), theta, record=record)
        images += self.gripper_close(record=record)
        images += self.move((*xy, z_up), theta, record=record)

        success = False
        for lego in self.legos:
            if self.p.getBasePositionAndOrientation(lego)[0][2] > z_up / 2:
                self.p.resetBasePositionAndOrientation(lego, (0, 0, -1), (0, 0, 0, 1))  # move away
                self.p.changeDynamics(lego, -1, mass=0)  # static
                success = True
                self.n_grasped += 1

        return success, images