from scipy.spatial.transform import Rotation as R
import numpy as np
import ipdb

def get_approach_pose(obj_name, obj_pose, bounding_box, pose, incupboard = False):

    grasps = []
    pre_grasps = []

    offset = 0.02

    quart_obj = obj_pose[-4:]
    r = R.from_quat(quart_obj)
    rot_mat = r.as_matrix()

    initial_grasp_pose = pose.copy()

    initial_grasp_pose = initial_grasp_pose[-4:]
    initial_grasp_pose = R.from_quat(initial_grasp_pose)
    initial_grasp_pose = initial_grasp_pose.as_matrix()

    target_pose = initial_grasp_pose.copy()

    blank_transform = [0,0,0]

    for i in range(3):

        pos_wrt_objframe = np.array([0,0,0]).astype('float')
        pos_wrt_objframe[2-i] += (bounding_box[-1-2*i] + offset)
        pos_wrt_global = np.matmul(rot_mat, pos_wrt_objframe) + obj_pose[0:3]

        r_ = R.from_matrix(target_pose)

        grasps.append(np.append(pos_wrt_global, r_.as_quat()))

        if incupboard:
            pos_wrt_objframe2 = pos_wrt_objframe.copy()
            pos_wrt_objframe2[2-i] += 0.3
            pos_wrt_global2 = np.matmul(rot_mat, pos_wrt_objframe2) + obj_pose[0:3]

            pre_grasps.append(np.append(pos_wrt_global2, r_.as_quat()))

        # ipdb.set_trace()
        target_transform = blank_transform.copy()
        target_transform[i] = -90
        # print(target_transform)

        target_transform = R.from_euler('xyz', target_transform, degrees=True)
        target_pose = np.matmul(target_pose, target_transform.as_matrix())

    target_pose = initial_grasp_pose.copy()
    Rx = R.from_euler('xyz', [-180,0,0], degrees=True)
    target_pose = np.matmul(target_pose, Rx.as_matrix())

    blank_transform = [0,0,0]
    
    for i in range(3):

        pos_wrt_objframe = np.array([0,0,0])
        pos_wrt_objframe[2-i] -= bounding_box[-1-2*i] + offset
        pos_wrt_global = np.matmul(rot_mat, pos_wrt_objframe) + obj_pose[0:3]

        r_ = R.from_matrix(target_pose)

        grasps.append(np.append(pos_wrt_global, r_.as_quat()))

        if incupboard:
            pos_wrt_objframe2 = pos_wrt_objframe.copy()
            pos_wrt_objframe2[2-i] -= 0.3
            pos_wrt_global2 = np.matmul(rot_mat, pos_wrt_objframe2) + obj_pose[0:3]

            pre_grasps.append(np.append(pos_wrt_global2, r_.as_quat()))

        target_transform = blank_transform.copy()
        if i == 0:
            target_transform[i] = -90
        else:
            target_transform[i] = 90

        target_transform = R.from_euler('xyz', target_transform, degrees=True)
        target_pose = np.matmul(target_pose, target_transform.as_matrix())

    return grasps, pre_grasps