import os
import sys
import time
import pickle
from copy import deepcopy
import numpy as np
import pybullet as p
import pybullet_data
import pybullet_utils.bullet_client
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from transform3d import Transform


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

# TODO: maybe do joints stiffer so the jaws dont wobble
class RobotCell:
    def __init__(self, cube_position, dt=0.01):

        self.dt = dt
        # instead of the local pybullet env just use global?
        # self.p = pybullet_utils.bullet_client.BulletClient(p.DIRECT)
        p.setTimeStep(dt)
        p.setGravity(0, 0, -9.8)
        # p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")

        # self.legos = [p.loadURDF("lego/lego.urdf") for i in range(n_legos)]
        # load cube to pick
        # self.cube_xy = (0, 0)
        # self.cube_position = (self.cube_xy[0], self.cube_xy[1], 0)
        self.cube_position = cube_position
        # self.cube_id = p.loadURDF("generated_urdfs/box_example.urdf", self.cube_position, p.getQuaternionFromEuler((0, 0, 0)))
        self.cube_id = p.loadURDF("generated_urdfs/box_8.urdf", self.cube_position, p.getQuaternionFromEuler((0, 0, 0)))

        """
        different boxes have different behaviour
        maybe when we change shape of the box we should also change mass?
        """
        p.changeDynamics(bodyUniqueId=self.cube_id,
                         linkIndex=-1,
                         mass=1.1,  # this mass works with the box_example.urdf but doesnt with other boxes
                         lateralFriction=sys.maxsize,
                         spinningFriction=sys.maxsize,
                         rollingFriction=sys.maxsize,
                         restitution=0.0,
                         linearDamping=0.0,
                         angularDamping=0.0,
                         contactStiffness=sys.maxsize,
                         contactDamping=sys.maxsize)

        self.n_grasped = 0
        self.rid = p.loadURDF(
            os.path.join(pybullet_data.getDataPath(), "franka_panda/panda.urdf"),
            (0, -0.55, 0), p.getQuaternionFromEuler((0, 0, np.pi / 2)),
            useFixedBase=True
        )
        print("self.rid ", self.rid)
        print("p.getNumJoints(self.rid) ", p.getNumJoints(self.rid))

        joint_infos = [p.getJointInfo(self.rid, i) for i in range(p.getNumJoints(self.rid))]
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
            p.changeDynamics(self.rid, joint_idx, linearDamping=0, angularDamping=0)
        self.gripper_open()
        self.reset()

    def set_q(self, q):
        for i, joint_idx in enumerate(self.joint_idxs):
            p.resetJointState(self.rid, joint_idx, q[i])
        self.set_q_target(q)

    def set_q_target(self, q):
        self.q_target = q
        p.setJointMotorControlArray(
            self.rid, self.joint_idxs, p.POSITION_CONTROL, q,
            forces=[100] * 7 + [15] * 2, positionGains=[1] * 9
        )

    def reset_robot(self):
        self.set_q(self.joint_rest)

    def reset(self):
        self.reset_robot()
        # for lego in self.legos:
        #     pos = *np.random.uniform(-0.1, 0.1, 2), 0.03
        #     quat = np.random.randn(4)
        #     quat = quat / np.linalg.norm(quat)
        #     p.resetBasePositionAndOrientation(lego, pos, quat)
        #     p.changeDynamics(lego, -1, mass=0.1)
        self.n_grasped = 0
        for i in range(50):  # let them fall to rest on the table
            p.stepSimulation()

    def take_image(self, res=400,
                   view_matrix=p.computeViewMatrix((0.5, 1, 1), (0, 0, 0.3), (0, 0, 1)),
                   projection_matrix=p.computeProjectionMatrixFOV(30, 1, 0.01, 10)):
        img, depth, seg = p.getCameraImage(
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
        q = list(p.calculateInverseKinematics(
            self.rid, 11, world_t_tool.p, world_t_tool.quat,
            # list(self.joint_lower_limits), list(self.joint_upper_limits),
            # list(self.joint_range), list(self.joint_rest),
            maxNumIterations=max_iter
        ))
        q[-1] = q[-2] = self.q_target[-1]
        return q

    def world_t_tool(self):
        pos, quat = p.getLinkState(self.rid, 11)[:2]
        return Transform(p=pos, quat=quat)

    def move(self, pos, theta=0., speed=1.2, acc=5., instant=False, record=False):
        world_t_tool_desired = Transform(p=pos, rpy=(0, np.pi, np.pi / 2 + theta))
        if instant:
            self.set_q(self.ik(world_t_tool_desired))
        else:
            world_t_tool_start = self.world_t_tool()
            tool_start_t_tool_desired = world_t_tool_start.inv @ world_t_tool_desired
            dist = np.linalg.norm(tool_start_t_tool_desired.xyz_rotvec)
            images = []
            qs = []
            for i, s in enumerate(lerp(dist, speed, acc, self.dt)):
                world_t_tool_target = world_t_tool_start @ (tool_start_t_tool_desired * s)
                self.set_q_target(self.ik(world_t_tool_target))
                p.stepSimulation()
                if record and i % 4 == 0:  # fps = 1/dt / 4
                    images.append(self.take_image())
                    qs.append(self.q_target)
            return images, qs

    def gripper_move(self, d_desired, record=False, speed=1., acc=3.):
        d_start = self.q_target[-1]
        move = d_desired - d_start
        images = []
        qs = []
        for i, s in enumerate(lerp(abs(move), speed, acc, self.dt)):
            self.q_target[-1] = self.q_target[-2] = d_start + s * move
            self.set_q_target(self.q_target)
            p.stepSimulation()
            if record and i % 4 == 0:
                images.append(self.take_image())
                qs.append(self.q_target)
        return images, qs

    def gripper_close(self, record=False):
        return self.gripper_move(0.0, record)

    def gripper_open(self, record=False):
        return self.gripper_move(0.03, record)

    def attempt_grasp(self, xy=(0, 0), theta=0, z_grasp=0.01, z_up=0.5, record=False):
        self.q_target[-1] = self.q_target[-2] = 0.03
        images, qs = [], []

        results = self.move((*xy, z_up), theta, instant=False)
        images.append(results[0])
        qs.append(results[1])

        print("type(results[0])", type(results[0]))
        # exit(1)

        results = self.move((*xy, z_grasp), theta, record=record)
        images.append(results[0])
        qs.append(results[1])

        results = self.gripper_close(record=record)
        images.append(results[0])
        qs.append(results[1])

        results = self.move((*xy, z_up), theta, record=record)
        images.append(results[0])
        qs.append(results[1])

        # success = False
        # for lego in self.legos:
        #     if p.getBasePositionAndOrientation(lego)[0][2] > z_up / 2:
        #         p.resetBasePositionAndOrientation(lego, (0, 0, -1), (0, 0, 0, 1))  # move away
        #         p.changeDynamics(lego, -1, mass=0)  # static
        #         success = True
        #         self.n_grasped += 1

        return get_continious(images), get_continious(qs)


def lerp(dist, vmax, amax, dt):
    if dist < 1e-6:
        return np.array([0, 1])
    # returns an array of interpolation values from between 0 and 1
    # representing the blend value from start to end.
    # dist is the distance between start and end,
    # speed=0 is assumed at start and end.
    # constant acceleration: v = a * t, s = 0.5 * a * t ** 2
    t_vmax = vmax / amax
    s_vmax = 0.5 * amax * t_vmax ** 2
    if s_vmax > dist * 0.5:
        t_vmax = (dist / amax) ** 0.5
        vmax = t_vmax * amax
        s_vmax = 0.5 * amax * t_vmax ** 2

    steps_ends = int(np.round(t_vmax / dt))
    steps_middle = int(np.round((dist - s_vmax * 2) / vmax / dt))

    s_start = 0.5 * amax * np.linspace(0, t_vmax, steps_ends + 1) ** 2
    s_middle = np.linspace(s_vmax, dist - s_vmax, steps_middle)
    s_end = dist - s_start[::-1]
    s = np.concatenate((s_start, s_middle[1:], s_end[1:]))
    return s / dist


class RobotCell_nonrecord:
    def __init__(self, cube_position, dt=0.01):

        self.dt = dt
        # instead of the local pybullet env just use global?
        # self.p = pybullet_utils.bullet_client.BulletClient(p.DIRECT)
        p.setTimeStep(dt)
        p.setGravity(0, 0, -9.8)
        # p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")

        # self.legos = [p.loadURDF("lego/lego.urdf") for i in range(n_legos)]
        # load cube to pick
        # self.cube_xy = (0, 0)
        # self.cube_position = (self.cube_xy[0], self.cube_xy[1], 0)
        self.cube_position = cube_position
        # self.cube_id = p.loadURDF("generated_urdfs/box_example.urdf", self.cube_position, p.getQuaternionFromEuler((0, 0, 0)))
        self.cube_id = p.loadURDF("generated_urdfs/box_8.urdf", self.cube_position, p.getQuaternionFromEuler((0, 0, 0)))

        """
        different boxes have different behaviour
        maybe when we change shape of the box we should also change mass?
        """
        p.changeDynamics(bodyUniqueId=self.cube_id,
                         linkIndex=-1,
                         mass=1.1,  # this mass works with the box_example.urdf but doesnt with other boxes
                         lateralFriction=sys.maxsize,
                         spinningFriction=sys.maxsize,
                         rollingFriction=sys.maxsize,
                         restitution=0.0,
                         linearDamping=0.0,
                         angularDamping=0.0,
                         contactStiffness=sys.maxsize,
                         contactDamping=sys.maxsize)

        self.n_grasped = 0
        self.rid = p.loadURDF(
            os.path.join(pybullet_data.getDataPath(), "franka_panda/panda.urdf"),
            (0, -0.55, 0), p.getQuaternionFromEuler((0, 0, np.pi / 2)),
            useFixedBase=True
        )
        print("self.rid ", self.rid)
        print("p.getNumJoints(self.rid) ", p.getNumJoints(self.rid))

        joint_infos = [p.getJointInfo(self.rid, i) for i in range(p.getNumJoints(self.rid))]
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
            p.changeDynamics(self.rid, joint_idx, linearDamping=0, angularDamping=0)
        self.gripper_open()
        self.reset()

    def set_q(self, q):
        for i, joint_idx in enumerate(self.joint_idxs):
            p.resetJointState(self.rid, joint_idx, q[i])
        self.set_q_target(q)

    def set_q_target(self, q):
        self.q_target = q
        p.setJointMotorControlArray(
            self.rid, self.joint_idxs, p.POSITION_CONTROL, q,
            forces=[100] * 7 + [15] * 2, positionGains=[1] * 9
        )

    def reset_robot(self):
        self.set_q(self.joint_rest)

    def reset(self):
        self.reset_robot()
        # for lego in self.legos:
        #     pos = *np.random.uniform(-0.1, 0.1, 2), 0.03
        #     quat = np.random.randn(4)
        #     quat = quat / np.linalg.norm(quat)
        #     p.resetBasePositionAndOrientation(lego, pos, quat)
        #     p.changeDynamics(lego, -1, mass=0.1)
        self.n_grasped = 0
        for i in range(50):  # let them fall to rest on the table
            p.stepSimulation()

    def take_image(self, res=400,
                   view_matrix=p.computeViewMatrix((0.5, 1, 1), (0, 0, 0.3), (0, 0, 1)),
                   projection_matrix=p.computeProjectionMatrixFOV(30, 1, 0.01, 10)):
        img, depth, seg = p.getCameraImage(
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
        q = list(p.calculateInverseKinematics(
            self.rid, 11, world_t_tool.p, world_t_tool.quat,
            # list(self.joint_lower_limits), list(self.joint_upper_limits),
            # list(self.joint_range), list(self.joint_rest),
            maxNumIterations=max_iter
        ))
        q[-1] = q[-2] = self.q_target[-1]
        return q

    def world_t_tool(self):
        pos, quat = p.getLinkState(self.rid, 11)[:2]
        return Transform(p=pos, quat=quat)

    def move(self, pos, theta=0., speed=1.2, acc=5., instant=False, record=False):
        world_t_tool_desired = Transform(p=pos, rpy=(0, np.pi, np.pi / 2 + theta))
        if instant:
            self.set_q(self.ik(world_t_tool_desired))
        else:
            world_t_tool_start = self.world_t_tool()
            tool_start_t_tool_desired = world_t_tool_start.inv @ world_t_tool_desired
            dist = np.linalg.norm(tool_start_t_tool_desired.xyz_rotvec)
            images = []
            # qs = []
            for i, s in enumerate(lerp(dist, speed, acc, self.dt)):
                world_t_tool_target = world_t_tool_start @ (tool_start_t_tool_desired * s)
                self.set_q_target(self.ik(world_t_tool_target))
                p.stepSimulation()
                if record and i % 4 == 0:  # fps = 1/dt / 4
                    images.append(self.take_image())
                    # qs.append(self.q_target)
            # return images, qs
        return None

    def gripper_move(self, d_desired, record=False, speed=1., acc=3.):
        d_start = self.q_target[-1]
        move = d_desired - d_start
        images = []
        qs = []
        for i, s in enumerate(lerp(abs(move), speed, acc, self.dt)):
            self.q_target[-1] = self.q_target[-2] = d_start + s * move
            self.set_q_target(self.q_target)
            p.stepSimulation()
            if record and i % 4 == 0:
                images.append(self.take_image())
            #     qs.append(self.q_target)
        # return images, qs

    def gripper_close(self, record=False):
        return self.gripper_move(0.0, record)

    def gripper_open(self, record=False):
        return self.gripper_move(0.03, record)

    def attempt_grasp(self, xy=(0, 0), theta=0, z_grasp=0.01, z_up=0.5, record=False):
        self.q_target[-1] = self.q_target[-2] = 0.03
        images, qs = [], []

        results = self.move((*xy, z_up), theta, instant=False)
        # images.append(results[0])
        # qs.append(results[1])
        #
        # print("type(results[0])", type(results[0]))
        # # exit(1)

        results = self.move((*xy, z_grasp), theta, record=record)
        # images.append(results[0])
        # qs.append(results[1])

        results = self.gripper_close(record=record)
        # images.append(results[0])
        # qs.append(results[1])

        results = self.move((*xy, z_up), theta, record=record)
        # images.append(results[0])
        # qs.append(results[1])

        # success = False
        # for lego in self.legos:
        #     if p.getBasePositionAndOrientation(lego)[0][2] > z_up / 2:
        #         p.resetBasePositionAndOrientation(lego, (0, 0, -1), (0, 0, 0, 1))  # move away
        #         p.changeDynamics(lego, -1, mass=0)  # static
        #         success = True
        #         self.n_grasped += 1

        # return get_continious(images), get_continious(qs)
        return None