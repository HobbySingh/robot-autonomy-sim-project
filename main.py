import numpy as np
import scipy as sp
from quaternion import from_rotation_matrix, quaternion
from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import *
from pyrep.const import ConfigurationPathAlgorithms as Algos
from scene_reset import *

import ipdb


def sample_normal_pose(pos_scale, rot_scale):
    '''
    Samples a 6D pose from a zero-mean isotropic normal distribution
    '''
    pos = np.random.normal(scale=pos_scale)

    eps = skew(np.random.normal(scale=rot_scale))
    R = sp.linalg.expm(eps)
    quat_wxyz = from_rotation_matrix(R)

    return pos, quat_wxyz

class Scene:

    def __init__(self, env, task, mode):
        self._env = env
        self._scene_objs = {}
        self._task = task
        self._pos_scale = [0.005] * 3 # noise params
        self._rot_scale = [0.01] * 3
        self._mode = mode

    def register_objs(self):
        '''
        This function creates a dictionary {obj_name : class_object of actual object}
        '''
        objs = self._env._scene._active_task.get_base().get_objects_in_tree(exclude_base=True, first_generation_only=False)
        for obj in objs:
            name = obj.get_name()
            self._scene_objs[name] = obj

        self.update_reset_positions()

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

        self.update()

    def update(self, joint_positions=None, gripper_state=None):
        if(gripper_state and joint_positions):
            obs = self._task._scene.get_observation()
            gripper_state = self._env._robot.gripper.get_open_amount()
            if(self._mode == "abs_joint_pos"):
                self._task.step(obs.joint_positions.tolist() + [gripper_state])
        else:
            if(self._mode == "abs_joint_pos"):
                self._task.step(joint_positions.tolist() + [gripper_state])


    def get_noisy_poses(self):

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

    def where_to_place(self):
        # TODO: where to place the objects while reset


    def reset(self):
        '''
        TODO
         1. Check for every box in a sequence, from closer to farther
         2. Generate a series of waypoints to pick the object and place it in its set loc.
        '''


if __name__ == "__main__":

    # Initializes environment and task
    mode = "abs_joint_pos" # ee_pose_plan
    action_mode = ActionMode(ArmActionMode.ABS_JOINT_POSITION) # See rlbench/action_modes.py for other action modes
    env = Environment(action_mode, '', ObservationConfig(), False, static_positions=False)
    task = env.get_task(PutGroceriesInCupboard) # available tasks: EmptyContainer, PlayJenga, PutGroceriesInCupboard, SetTheTable
    agent = RandomAgent()
    task.reset()

    scene = Scene(env, task, mode)  # Initialize our scene class
    scene.register_objs() # Register all objects in the environment

    # TODO - RL Forward Policy
    scene.set_positions() # Run an episode of forward policy or set object locations manually

    scene.reset()

    while True:

        scene.update()

    env.shutdown()