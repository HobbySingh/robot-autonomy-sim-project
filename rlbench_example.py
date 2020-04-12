import numpy as np
import scipy as sp
from quaternion import from_rotation_matrix, quaternion

from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import *
import ipdb
from pyrep.const import ConfigurationPathAlgorithms as Algos

def skew(x):
    return np.array([[0, -x[2], x[1]],
                    [x[2], 0, -x[0]],
                    [-x[1], x[0], 0]])


def sample_normal_pose(pos_scale, rot_scale):
    '''
    Samples a 6D pose from a zero-mean isotropic normal distribution
    '''
    pos = np.random.normal(scale=pos_scale)
        
    eps = skew(np.random.normal(scale=rot_scale))
    R = sp.linalg.expm(eps)
    quat_wxyz = from_rotation_matrix(R)

    return pos, quat_wxyz


class RandomAgent:

    def act(self, obs):
        # delta_pos = [(np.random.rand() * 2 - 1) * 0.005, 0, 0]
        # delta_pos = [0, 0, 0]
        # delta_quat = [0, 0, 0, 1] # xyzw
        gripper_pos = [False]
        # return delta_pos + delta_quat + gripper_pos
        # ipdb.set_trace()
        return obs.joint_positions.tolist() + gripper_pos

class SceneObjects:

    def __init__(self, env):
        self._env = env
        self.objs_dict = {}
        objs = self._env._scene._active_task.get_base().get_objects_in_tree(exclude_base=True, first_generation_only=False)
        for obj in objs:
            # ipdb.set_trace()
            name = obj.get_name()
            self.objs_dict[name] = obj

    def set_positions(self):
        objs = self._env._scene._active_task.get_base().get_objects_in_tree(exclude_base=True, first_generation_only=False)
        x = 0
        for obj in objs:
            name = obj.get_name()

            if((name == 'crackers') or (name == 'crackers_visual')
                    or (name == 'chocolate_jello') or (name == 'chocolate_jello_visual')
                    or (name == 'strawberry_jello') or (name == 'strawberry_jello_visual')
                    or (name == 'tuna') or (name == 'tuna_visual')
                    or (name == 'spam') or (name == 'spam_visual')
                    or (name == 'coffee') or (name == 'coffee_visual')
                    or (name == 'mustard') or (name == 'mustard_visual')
                    or (name == 'sugar') or (name == 'sugar_visual')):

                obj.set_position([x, 0.03, 0.1])
                x += 0.01

            if((name == 'soup') or (name == 'soup_visual')):
                obj.set_position([0.3, 0, 0.8])

            if((name == 'soup_grasp_point')):
                obj.set_position([0.3, 0, 0.825])

            if(name == 'cupboard'):
                cupboard_pose = obj.get_position()
                cupboard_pose[2] += 0.75
                obj.set_position(cupboard_pose)

    def get_noisy_poses(self):
        self._pos_scale = [0.005] * 3
        self._rot_scale = [0.01] * 3

        objs = self._env._scene._active_task.get_base().get_objects_in_tree(exclude_base=True, first_generation_only=False)
        obj_poses = {}
        for obj in objs:
            name = obj.get_name()
            pose = obj.get_pose()

            pos, quat_wxyz = sample_normal_pose(self._pos_scale, self._rot_scale)
            gt_quat_wxyz = quaternion(pose[6], pose[3], pose[4], pose[5])
            perturbed_quat_wxyz = quat_wxyz * gt_quat_wxyz

            pose[:3] += pos
            pose[3:] = [perturbed_quat_wxyz.x, perturbed_quat_wxyz.y, perturbed_quat_wxyz.z, perturbed_quat_wxyz.w]

            obj_poses[name] = pose

        return obj_poses

def execute_path(path, gripper_open):

    path_points = path._path_points.reshape(-1, path._num_joints)

    if(gripper_open):
        path_joints = np.hstack((path_points, np.ones((path_points.shape[0], 1))))
    else:
        path_joints = np.hstack((path_points, np.zeros((path_points.shape[0], 1))))

    i = 0
    while not path._path_done and i < path_joints.shape[0]:
        task.step(path_joints[i])
        i += 1


if __name__ == "__main__":
    action_mode = ActionMode(ArmActionMode.ABS_JOINT_POSITION) # See rlbench/action_modes.py for other action modes
    env = Environment(action_mode, '', ObservationConfig(), False, static_positions=False)
    task = env.get_task(PutGroceriesInCupboard) # available tasks: EmptyContainer, PlayJenga, PutGroceriesInCupboard, SetTheTable
    agent = RandomAgent()
    descriptions, obs = task.reset()
    print(descriptions)

    # Register all objects in the environment
    scene_objs = SceneObjects(env)
    # clear scene
    scene_objs.set_positions()
    task.step(obs.joint_positions.tolist() + [True])

    # get noisy poses
    obj_poses = scene_objs.get_noisy_poses()
    task.step(obs.joint_positions.tolist() + [True])

    if(True):
        # go to object
        position = obj_poses['soup_grasp_point'][0:3]
        path = env._robot.arm.get_path(position=position, quaternion=obj_poses['soup_grasp_point'][3:],
                                       max_configs = 500, trials = 1000, algorithm=Algos.RRTstar)
        execute_path(path, gripper_open=True)

        # Close gripper
        task.step(obs.joint_positions.tolist() + [False])

        # attach object to gripper
        print("Grasp: ", env._robot.gripper.grasp(scene_objs.objs_dict['soup']))
        task.step(obs.joint_positions.tolist() + [False])

        # pick it up
        position[2] += 0.25
        path = env._robot.arm.get_path(position=position,
                                       quaternion=obj_poses['soup_grasp_point'][3:], max_configs = 500, trials = 1000,
                                       ignore_collisions=False, algorithm=Algos.RRTstar)
        execute_path(path, gripper_open=False)

    while True:
        # Getting noisy object poses
        obj_poses = scene_objs.get_noisy_poses()

        # Getting various fields from obs
        current_joints = obs.joint_positions
        gripper_pose = obs.gripper_pose
        rgb = obs.wrist_rgb
        depth = obs.wrist_depth
        mask = obs.wrist_mask

        # Perform action and step simulation
        action = agent.act(obs)
        obs, reward, terminate = task.step(action)

        # if terminate:
        #     break

    env.shutdown()
