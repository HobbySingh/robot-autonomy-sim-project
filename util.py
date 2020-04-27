from scipy.spatial.transform import Rotation as R
import numpy as np

def get_approach_pose(obj_name, obj_pose, bounding_box):

    grasps = []

    offset = 0.035

    quart_obj = obj_pose[-4:]
    r = R.from_quat(quart_obj)
    rot_mat = r.as_matrix()
    r = r.as_euler('xyz', degrees=True)

    pos_wrt_objframe = obj_pose[0:3]

    gripper_pose = R.from_euler('xyz', [-180,0,180], degrees=True)

    target_pose = rot_mat.copy()
    Rx = R.from_euler('xyz', [-180,0,0], degrees=True)
    target_pose = np.matmul(target_pose, Rx.as_matrix())

    blank_transform = [0,0,0]

    for i in range(3):

        pos_wrt_objframe = obj_pose[0:3].copy() 
        pos_wrt_objframe[2-i] += bounding_box[-1-2*i] + offset
        pos_wrt_global = np.matmul(rot_mat, pos_wrt_objframe)

        r_ = R.from_matrix(target_pose)

        grasps.append(np.append(pos_wrt_global, r_.as_quat()))

        target_transform = blank_transform.copy()
        target_transform[i] = -90
        print(target_transform)

        target_transform = R.from_euler('xyz', target_transform, degrees=True)
        target_pose = np.matmul(target_pose, target_transform.as_matrix())

    target_pose = rot_mat.copy()

    blank_transform = [0,0,0]
    
    for i in range(3):

        pos_wrt_objframe = obj_pose[0:3].copy() 
        pos_wrt_objframe[2-i] -= bounding_box[-1-2*i] + offset
        pos_wrt_global = np.matmul(rot_mat, pos_wrt_objframe)

        r_ = R.from_matrix(target_pose)

        grasps.append(np.append(pos_wrt_global, r_.as_quat()))

        target_transform = blank_transform.copy()
        if i == 0:
            target_transform[i] = -90
        else:
            target_transform[i] = 90

        target_transform = R.from_euler('xyz', target_transform, degrees=True)
        target_pose = np.matmul(target_pose, target_transform.as_matrix())

    return grasps


def get_approach_pose2(obj_name, obj_pose, bounding_box, initial_grasp_pose):

    grasps = []

    offset = 0.035

    quart_obj = obj_pose[-4:]
    r = R.from_quat(quart_obj)
    rot_mat = r.as_matrix()
    r = r.as_euler('xyz', degrees=True)

    pos_wrt_objframe = obj_pose[0:3]

    initial_grasp_pose = initial_grasp_pose[-4:]
    initial_grasp_pose = R.from_quat(initial_grasp_pose)
    initial_grasp_pose = initial_grasp_pose.as_matrix()

    target_pose = initial_grasp_pose.copy()

    blank_transform = [0,0,0]

    for i in range(3):

        pos_wrt_objframe = obj_pose[0:3].copy() 
        pos_wrt_objframe[2-i] += bounding_box[-1-2*i] + offset
        pos_wrt_global = np.matmul(rot_mat, pos_wrt_objframe)

        r_ = R.from_matrix(target_pose)

        grasps.append(np.append(pos_wrt_global, r_.as_quat()))

        target_transform = blank_transform.copy()
        target_transform[i] = -90
        print(target_transform)

        target_transform = R.from_euler('xyz', target_transform, degrees=True)
        target_pose = np.matmul(target_pose, target_transform.as_matrix())

    target_pose = initial_grasp_pose.copy()
    Rx = R.from_euler('xyz', [-180,0,0], degrees=True)
    target_pose = np.matmul(target_pose, Rx.as_matrix())

    blank_transform = [0,0,0]
    
    for i in range(3):

        pos_wrt_objframe = obj_pose[0:3].copy() 
        pos_wrt_objframe[2-i] -= bounding_box[-1-2*i] + offset
        pos_wrt_global = np.matmul(rot_mat, pos_wrt_objframe)

        r_ = R.from_matrix(target_pose)

        grasps.append(np.append(pos_wrt_global, r_.as_quat()))

        target_transform = blank_transform.copy()
        if i == 0:
            target_transform[i] = -90
        else:
            target_transform[i] = 90

        target_transform = R.from_euler('xyz', target_transform, degrees=True)
        target_pose = np.matmul(target_pose, target_transform.as_matrix())

    return grasps