import numpy as np
import pyrep
import util_2
import ipdb

def create_place_points(obj_poses):

    success_position = obj_poses['success']
    place_points = get_mirror_poses(success_position, 0.1)
    return place_points


def get_mirror_poses(base_pose, delta=0.1):
    left = np.array(base_pose)
    right = np.array(base_pose)
    left[1] -= delta
    right[1] += delta
    return [base_pose]


def create_waypoint_sequence(place_point, scene):
    obj_poses = scene.get_noisy_poses()
    waypoint3 = obj_poses['waypoint3']
    waypoint3[0] = waypoint3[0] - 0.05
    waypoint4 = obj_poses['waypoint4']
    waypoint4[0] = waypoint4[0] - 0.05
    place_point[0] = waypoint4[0]
    place_point[2:] = waypoint4[2:]

    return [waypoint3, waypoint4, place_point]


def reset_to_cupboard(scene):

    obj_poses = scene.get_noisy_poses()
    waypoint3 = obj_poses['waypoint3']
    waypoint4 = obj_poses['waypoint4']
    gt_grasp_points = []  # [x, y, z, q1, q2, q3, q4]
    for k, v in obj_poses.items():
        v[2] = v[2] + 0.035  # keep some distance b/w suction cup and object
        if 'grasp' in k:
            if 'jello' in k or 'sugar' in k:
                gt_grasp_points.append((k, v))
            else:
                pass
        else:
            continue

    place_pts = create_place_points(obj_poses)
    for idx, element in enumerate(place_pts):
        try:

            obj_name, initial_grasp_pt = gt_grasp_points.pop(0)
            print("Placing in cupboard: ", obj_name[:-12])

            pose = scene._scene_objs[obj_name[:-12]].get_pose()
            bb = scene._scene_objs[obj_name[:-12]].get_bounding_box()

            if(pose[2] > 1.2):
                incupboard = True
            else:
                incupboard = False
            grasp_points, pre_grasp_points = util_2.get_approach_pose(obj_name[:-12], pose, bb, initial_grasp_pt.copy(),
                                                               incupboard=incupboard)

            i=0
            while(grasp_points):
                i+=1
                try:
                    print("Trying Grasp Pose: ", i)

                    gsp_pt = grasp_points.pop(0)
                    print("Grasping: ", obj_name[:-12])
                    # pre_gsp_pt = scene.pre_grasp(gsp_pt.copy())
                    pre_gsp_pt = pre_grasp_points.pop(0)

                    print("Move to pre-grasp point for: ", obj_name[:-12])
                    scene.update(pre_gsp_pt, move_arm=True)

                    print("Move to grasp point for: ", obj_name[:-12])
                    scene.update(gsp_pt, move_arm=True, ignore_collisions=True)

                    print("Attach object to gripper: " + obj_name[:-12],
                          scene._env._robot.gripper.grasp(scene._scene_objs[obj_name[:-12]]))
                    scene.update(move_arm=False)

                    print("Just move to pre-grasp point while holding: ", obj_name[:-12])
                    scene.update(pre_gsp_pt, move_arm=True, ignore_collisions=True)

                    print("Place the object inside cupboard")
                    path_sequence = create_waypoint_sequence(element.copy(), scene)
                    for waypoint in path_sequence:
                        scene.update(waypoint, move_arm=True, ignore_collisions=False)

                    print("Release :", obj_name[:-12])
                    scene._env._robot.gripper.release()
                    scene.update(move_arm=False)
                    grasp_points = None

                except pyrep.errors.ConfigurationPathError:
                    print("Could Not find Path for grasp pose number :", i)
                    scene._env._robot.gripper.release()


        except pyrep.errors.ConfigurationPathError:
            print("Could Not find Path")
            scene._robot.gripper.release()

    # scene.update(waypoint4, move_arm=True, ignore_collisions=True)
    scene.update(waypoint3, move_arm=True, ignore_collisions=True)
    return grasp_points
