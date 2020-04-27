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
from util import get_approach_pose, get_approach_pose

def reset_on_table(scene):

    env = scene._env
    objs = scene._env._scene._active_task.get_base().get_objects_in_tree(exclude_base=True, first_generation_only=False)

    objs_by_name = {}

    for obj in objs:
        obj_name = obj.get_name()
        # print(obj_name)
        objs_by_name[obj_name] = obj

    for obj in objs[0:9]: # With assumption that there is no collision in the process and none of the objects change pose unless we explicitly do that

        obj_name = obj.get_name()
        pose = obj.get_pose()
        bb = obj.get_bounding_box()

        initial_grasp_point = objs_by_name[obj_name + '_grasp_point'].get_pose()
        print('here')
        print(initial_grasp_point)
        grasp_points = get_approach_pose(obj_name,pose,bb, initial_grasp_point.copy(), incupboard = True)
        print('here')
        print(grasp_points[0])
        print(initial_grasp_point)
        grasp_points[0] = initial_grasp_point
        grasp_points[0][2] += 0.035

        i = 0

        while grasp_points:
            print(i)
            i += 1
            try:
                gsp_pt = grasp_points.pop(0)
                print(gsp_pt)

                print("Grasping: ", obj_name)
                # pre_gsp_pt = scene.pre_grasp(gsp_pt.copy())
                # pre_gsp_pt = pre_grasp_points.pop(0)
                pre_gsp_pt = scene.pre_grasp(gsp_pt.copy())
                print(pre_gsp_pt)

                print("Move to pre-grasp point for: ", obj_name)
                scene.update(pre_gsp_pt, move_arm=True)

                print("Move to grasp point for: ", obj_name)
                scene.update(gsp_pt, move_arm=True)

                print("Attach object to gripper: " + obj_name, env._robot.gripper.grasp(scene._scene_objs[obj_name]))
                scene.update(move_arm=False)

                print("Just move up while holding: ", obj_name)
                scene.update(pre_gsp_pt, move_arm=True, ignore_collisions=True)

                while True:
                    print("Trying new positions to randomly place")

                    shape_obj = Shape(obj_name)
                    status, place_pt, rotation = scene._task._task.boundary.find_position_on_table(shape_obj, min_distance=0.1)
                    place_pt[2] = gsp_pt[2] + 0.025
                    place_pt = np.array(place_pt + [0.707,0.707,0,0])

                    pre_place_pt = scene.pre_grasp(place_pt.copy())
                    try:
                        print("Going to pre_place_pt with gripper close")
                        scene.update(pre_place_pt, move_arm=True)
                        print("Going to place_pt with gripper close")
                        scene.update(place_pt, move_arm=True)
                        break
                    except:
                        print("Path not found")
                        continue

                print("opening gripper")
                print("DeGrasp: " + obj_name)
                env._robot.gripper.release()
                scene.update()

                print("Going in air")
                scene.update(pre_place_pt, move_arm=True)
                break

            except pyrep.errors.ConfigurationPathError:
                print("Could Not find Path")
                env._robot.gripper.release()
    
    return