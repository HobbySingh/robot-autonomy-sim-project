import numpy as np
import scipy as sp
from quaternion import from_rotation_matrix, quaternion
from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import *
from pyrep.const import ConfigurationPathAlgorithms as Algos
import copy
import ipdb
import pyrep
import math
from rlbench.backend.spawn_boundary import SpawnBoundary
from pyrep.objects.shape import Shape
from scipy.spatial.transform import Rotation as R


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


    def set_positions(self):
        objs = self._env._scene._active_task.get_base().get_objects_in_tree(exclude_base=True, first_generation_only=False)
        x = 0
        for obj in objs:
            name = obj.get_name()

            if((name == 'chocolate_jello')):
                    # or (name == 'crackers') or (name == 'crackers_visual')
                    # or (name == 'strawberry_jello') or (name == 'strawberry_jello_visual')
                    # or (name == 'tuna') or (name == 'tuna_visual')
                    # or (name == 'spam') or (name == 'spam_visual')
                    # or (name == 'coffee') or (name == 'coffee_visual')
                    # or (name == 'mustard') or (name == 'mustard_visual')
                    # or (name == 'sugar') or (name == 'sugar_visual')):

                # obj.set_position([0.4357, 0, 1.38])
                obj.rotate([0, 1.57, 0])

            if((name == 'crackers')):
                obj.rotate([0, 1.57, 0])
            #
            # if((name == 'soup_grasp_point')):
            #     obj.set_position([0.3, 0, 0.825])

            # if(name == 'cupboard'):
            #     cupboard_pose = obj.get_position()
            #     cupboard_pose[2] += 0.75
            #     obj.set_position(cupboard_pose)

        self.update()

    def update(self, joint_positions=None, move_arm = False, ignore_collisions=False):
        '''
        Finds path to target pose and executes it
        Can be run with step_with_action false, it won't take any action
        but will update the environment
        :param joint_positions: joint positions/ gripper pose depending upon mode
        :param step_with_action: actual path to be executed or just updating the env.
        :param ignore_collisions:  ignore collisions or not
        :return:
        '''
        obs = self._task._scene.get_observation()

        if(move_arm):
            if(self._mode == "abs_joint_pos"):
                path = env._robot.arm.get_path(position=joint_positions[0:3], quaternion=joint_positions[3:],
                                               max_configs = 500, trials = 1000, algorithm=Algos.BiTRRT,
                                               ignore_collisions=ignore_collisions)
                self.execute_path(path)
            else:
                action = joint_positions.tolist()
                self._task.step(action)
        else:
            if(self._mode == "abs_joint_pos"):
                self._task.step(obs.joint_positions.tolist())
            else:
                self._task.step(obs.gripper_pose.tolist())


    def get_noisy_poses(self):

        objs = self._env._scene._active_task.get_base().get_objects_in_tree(exclude_base=True, first_generation_only=False)
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

    def where_to_place(self, curr_obj_name):
        # TODO: where to place the objects while reset
        curr_obj = self._scene_objs[curr_obj_name[:-12]]
        obj_grasp_point = self._scene_objs[curr_obj_name]

        bb0 = curr_obj.get_bounding_box()
        half_diag = (bb0[0]**2 + bb0[2]**2)**0.5
        h = curr_obj.get_pose()[2]-0.25

        while True:
            check = True
            a = np.random.uniform(0,0.25)
            b = np.random.uniform(0, 0.4)
            theta = np.random.uniform(0, 2*math.pi)

            x = a*math.cos(theta) + 0.25
            y = b*math.sin(theta)

            # print(x,y,h)

            obj_poses = self.get_noisy_poses()
            # action = [x,y,h] + list(obj_poses[name+'_grasp_point'][3:]) + [False]
            
            objs = self._env._scene._active_task.get_base().get_objects_in_tree(exclude_base=True, first_generation_only=False)
            for obj in objs:
                # print(obj.get_name())
                pose = obj.get_pose()
                dist = np.sum((pose[0:2]-np.array([x,y]))**2) ** 0.5
                bb = obj.get_bounding_box()
                if dist < half_diag + (bb[0]**2 + bb[2]**2)**0.5:
                    check = False
                    break
            
            if not check: 
                continue
            else:
                break
        #[x, y, z, q1, q2, q3, q4]
        return np.array([x,y,h] + obj_grasp_point.get_pose()[3:].tolist())
        # return np.array([x,y,h] + [0,0,0,1])


    def reset(self):
        '''
        TODO
         1. Check for every box in a sequence, from closer to farther
         2. Generate a series of waypoints to pick the object and place it in its set loc.
        '''
        obj_poses = self.get_noisy_poses()
        grasp_points = [] #[x, y, z, q1, q2, q3, q4]
        # iterate through all the objects
        for k, v in obj_poses.items():
            v[2] = v[2] + 0.035 # keep some distance b/w suction cup and object
            if 'grasp' not in k:
                pass
            else:
                grasp_points.append((k,v))

        # sort object positions based on distance from the base
        # grasp_points = sorted(grasp_points, key = lambda x: (x[0]**2 + x[1]**2))


        while grasp_points:
            try:

                obj_name, gsp_pt = grasp_points.pop(0)

                # h = self._scene_objs[obj_name[:-12]].get_pose()[2] + 0.1

                print("Grasping: ", obj_name[:-12])
                pre_gsp_pt = self.pre_grasp(gsp_pt.copy())

                print("Move to pre-grasp point for: ", obj_name[:-12])
                self.update(pre_gsp_pt, move_arm=True)

                print("Move to grasp point for: ", obj_name[:-12])
                self.update(gsp_pt, move_arm=True, ignore_collisions=True)

                print("Attach object to gripper: " + obj_name[:-12], env._robot.gripper.grasp(scene._scene_objs[obj_name[:-12]]))
                self.update(move_arm=False)

                print("Just move up while holding: ", obj_name[:-12])
                self.update(pre_gsp_pt, move_arm=True, ignore_collisions=True)

                while True:
                    print("Trying new positions to randomly place")

                    shape_obj = Shape(obj_name[:-12])
                    # ipdb.set_trace()
                    status, place_pt, rotation = self._task._task.boundary.find_position_on_table(shape_obj, min_distance=0.2)
                    r  = R.from_euler('xyz', rotation)

                    # ipdb.set_trace()
                    place_pt[2] = gsp_pt[2] + 0.02
                    place_pt = np.array(place_pt + (gsp_pt[3:]).tolist())
                    # place_pt = np.array(place_pt + [r.as_quat()[3], r.as_quat()[2], 0, 0])
                    # place_pt = np.array(place_pt + [ 0.707, 0.707, 0, 0])
                    # if(obj_name[:-12] == 'chocolate_jello'):
                    #     r = (R.from_euler('xyz', [1.57, 0, 1.57])).as_quat()
                    #     place_pt = np.array([0.4757, 0.0439, 1.4, r[0], r[1], r[2], r[3]])

                    pre_place_pt = self.pre_grasp(place_pt.copy())
                    try:
                        print("Going to pre_place_pt with gripper close")
                        self.update(pre_place_pt, move_arm=True, ignore_collisions=False)
                        print("Going to place_pt with gripper close")
                        self.update(place_pt, move_arm=True)
                        break
                    except:
                        print("Path not found")
                        continue

                print("opening gripper")
                print("DeGrasp: " + obj_name[:-12])
                env._robot.gripper.release()
                self.update()

                print("Going in air")
                self.update(pre_place_pt, move_arm=True)

            except pyrep.errors.ConfigurationPathError:
                print("Could Not find Path")
                env._robot.gripper.release()
        return

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
    if(mode == "ee_pose_plan"):
        action_mode = ActionMode(ArmActionMode.ABS_EE_POSE_PLAN) # See rlbench/action_modes.py for other action modes
    elif(mode == "abs_joint_pos"):
        action_mode = ActionMode(ArmActionMode.ABS_JOINT_POSITION)
    else:
        raise Exception('Mode not Found')


    env = Environment(action_mode, '', ObservationConfig(), False, static_positions=False)
    task = env.get_task(PutGroceriesInCupboard) # available tasks: EmptyContainer, PlayJenga, PutGroceriesInCupboard, SetTheTable
    task.reset()

    scene = Scene(env, task, mode)  # Initialize our scene class
    scene.register_objs() # Register all objects in the environment

    # scene.set_positions() # Run an episode of forward policy or set object locations manually

    scene.reset()

    while True:

        scene.update()

        scene.reset()

    env.shutdown()
