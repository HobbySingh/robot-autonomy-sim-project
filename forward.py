import numpy as np
import pyrep

def create_place_points(obj_poses):

    success_position = obj_poses['success']
    place_points = get_mirror_poses(success_position, 0.1)
    return place_points


def get_mirror_poses(base_pose, delta=0.1):
    left = np.array(base_pose)
    right = np.array(base_pose)
    left[1] -= delta
    right[1] += delta
    return [left, right, base_pose]


def create_waypoint_sequence(place_point, scene):
    obj_poses = scene.get_noisy_poses()
    waypoint3 = obj_poses['waypoint3']
    waypoint4 = obj_poses['waypoint4']
    place_point[0] = waypoint4[0]
    place_point[2:] = waypoint4[2:]

    return [waypoint3, waypoint4, place_point]


def reset_to_cupboard(scene):

    obj_poses = scene.get_noisy_poses()
    waypoint3 = obj_poses['waypoint3']
    waypoint4 = obj_poses['waypoint4']
    grasp_points = []  # [x, y, z, q1, q2, q3, q4]
    for k, v in obj_poses.items():
        v[2] = v[2] + 0.035  # keep some distance b/w suction cup and object
        if 'grasp' in k:
            if 'jello' in k or 'sugar' in k:
                grasp_points.append((k, v))
            else:
                pass
        else:
            continue

    place_pts = create_place_points(obj_poses)
    for idx, element in enumerate(place_pts):
        try:

            obj_name, gsp_pt = grasp_points.pop(0)

            print("Grasping: ", obj_name[:-12])
            pre_gsp_pt = scene.pre_grasp(gsp_pt.copy())

            print("Move to pre-grasp point for: ", obj_name[:-12])
            scene.update(pre_gsp_pt, move_arm=True)

            print("Move to grasp point for: ", obj_name[:-12])
            scene.update(gsp_pt, move_arm=True, ignore_collisions=True)

            print("Attach object to gripper: " + obj_name[:-12],
                  scene._env._robot.gripper.grasp(scene._scene_objs[obj_name[:-12]]))
            scene.update(move_arm=False)

            print("Just move up while holding: ", obj_name[:-12])
            scene.update(pre_gsp_pt, move_arm=True, ignore_collisions=True)

            path_sequence = create_waypoint_sequence(element.copy(), scene)
            for waypoint in path_sequence:
                scene.update(waypoint, move_arm=True, ignore_collisions=False)
            scene._env._robot.gripper.release()
            scene.update(move_arm=False)


        except pyrep.errors.ConfigurationPathError:
            print("Could Not find Path")
            scene._robot.gripper.release()

    scene.update(waypoint4, move_arm=True, ignore_collisions=True)
    scene.update(waypoint3, move_arm=True, ignore_collisions=True)
    return grasp_points
