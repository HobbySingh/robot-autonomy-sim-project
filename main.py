import numpy as np
import scipy as sp
from quaternion import from_rotation_matrix, quaternion
from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import *
from pyrep.const import ConfigurationPathAlgorithms as Algos
import pyrep
from pyrep.objects.shape import Shape
from scipy.spatial.transform import Rotation as R
import forward
import reset
from rlbench.backend.spawn_boundary import SpawnBoundary
import copy
import ipdb

GROCERY_NAMES = [
    'crackers',
    'chocolate jello',
    'strawberry jello',
    'soup',
    'tuna',
    'spam',
    'coffee',
    'mustard',
    'sugar',
]


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


class Scene:

    def __init__(self, env, task, mode):
        self._env = env
        self._scene_objs = {}
        self._task = task
        self._pos_scale = [0.005] * 3  # noise params
        self._rot_scale = [0.01] * 3
        self._mode = mode

    def register_objs(self):
        '''
        This function creates a dictionary {obj_name : class_object of actual object}
        '''
        objs = self._env._scene._active_task.get_base().get_objects_in_tree(exclude_base=True,
                                                                            first_generation_only=False)
        for obj in objs:
            name = obj.get_name()
            self._scene_objs[name] = obj

    def preset_positions(self):
        '''
        This function sets the positions of objects to desired locations
        :return:
        '''
        objs = self._env._scene._active_task.get_base().get_objects_in_tree(exclude_base=True,
                                                                            first_generation_only=False)
        for obj in objs:
            name = obj.get_name()

            if ((name == 'chocolate_jello')):
                # obj.set_position([0.4357, 0, 1.38])
                obj.rotate([0, 1.57, 0])

            if ((name == 'crackers')):
                obj.rotate([0, 1.57, 0])

            # if(name == 'cupboard'):
            #     cupboard_pose = obj.get_position()
            #     cupboard_pose[2] += 0.75
            #     obj.set_position(cupboard_pose)

        self.update()

    def update(self, joint_positions=None, move_arm=False, ignore_collisions=False):
        '''
        Finds path to target pose and executes it
        Can be run with move_arm false, it won't take any action
        but will update the environment
        :param joint_positions: joint positions/ gripper pose depending upon mode
        :param step_with_action: actual path to be executed or just updating the env.
        :param ignore_collisions:  ignore collisions or not
        :return:
        '''
        obs = self._task._scene.get_observation()

        if (move_arm):
            if (self._mode == "abs_joint_pos"):
                path = env._robot.arm.get_path(position=joint_positions[0:3], quaternion=joint_positions[3:],
                                               max_configs=500, trials=1000, algorithm=Algos.BiTRRT,
                                               ignore_collisions=ignore_collisions)
                self.execute_path(path)
            else:
                action = joint_positions.tolist()
                self._task.step(action)
        else:
            if (self._mode == "abs_joint_pos"):
                self._task.step(obs.joint_positions.tolist())
            else:
                self._task.step(obs.gripper_pose.tolist())

    def get_noisy_poses(self):
        '''
        This function returns noisy poses for the objects in the scene
        :return:
        '''
        objs = self._env._scene._active_task.get_base().get_objects_in_tree(exclude_base=True,
                                                                            first_generation_only=False)
        obj_poses = {}
        for obj in objs:
            name = obj.get_name()
            pose = obj.get_pose()

            # pos, quat_wxyz = sample_normal_pose(self._pos_scale, self._rot_scale)
            # gt_quat_wxyz = quaternion(pose[6], pose[3], pose[4], pose[5])
            # perturbed_quat_wxyz = quat_wxyz * gt_quat_wxyz

            # pose[:3] += pos
            # pose[3:] = [perturbed_quat_wxyz.x, perturbed_quat_wxyz.y, perturbed_quat_wxyz.z, perturbed_quat_wxyz.w]

            obj_poses[name] = pose

        return obj_poses

    def pre_grasp(self, grasp_vect):
        pre_grasp_point = grasp_vect
        pre_grasp_point[2] += 0.3
        return pre_grasp_point

    def execute_path(self, path):
        path_points = path._path_points.reshape(-1, path._num_joints)
        path_joints = path_points

        i = 0
        while not path._path_done and i < path_joints.shape[0]:
            self._task.step(path_joints[i])
            i += 1

if __name__ == "__main__":

    # Initializes environment and task
    mode = "abs_joint_pos"
    # mode = "ee_pose_plan"
    if (mode == "ee_pose_plan"):
        action_mode = ActionMode(ArmActionMode.ABS_EE_POSE_PLAN)  # See rlbench/action_modes.py for other action modes
    elif (mode == "abs_joint_pos"):
        action_mode = ActionMode(ArmActionMode.ABS_JOINT_POSITION)
    else:
        raise Exception('Mode not Found')

    env = Environment(action_mode, '', ObservationConfig(), False, static_positions=False)
    task = env.get_task(PutGroceriesInCupboard)
    task.reset()

    '''
    Step 1: Init 
    Initialize scene, register objects, preset positions
    '''
    scene = Scene(env, task, mode)  # Initialize the scene
    scene.register_objs()  # Register all objects in the environment
    # scene.preset_positions()
    '''
    Step 2: Forward Policy 
    Place selected items in cupboard
    '''
    forward.reset_to_cupboard(scene)
    '''
    Step 3: Reset
    1. Reset the environment by removing items from the cupboard
    2. Rearrange the items on the table
    '''
    reset.reset_on_table(scene)

    env.shutdown()
