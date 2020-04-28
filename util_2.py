from scipy.spatial.transform import Rotation as R
import numpy as np
import ipdb


def get_approach_pose(obj_name, obj_pose, bounding_box, pose, incupboard = False):

    print(obj_name)
    grasps = []
    pre_grasps = []

    offset = 0

    quart_obj = obj_pose[-4:]
    r = R.from_quat(quart_obj)
    rot_mat = r.as_matrix()
    r = r.as_euler('xyz', degrees=True)

    pos_wrt_objframe = obj_pose[0:3]

    initial_grasp_pose = pose.copy()

    initial_grasp_pose = initial_grasp_pose[-4:]
    initial_grasp_pose = R.from_quat(initial_grasp_pose)
    initial_grasp_pose = initial_grasp_pose.as_matrix()

    target_pose = initial_grasp_pose.copy()

    blank_transform = [0,0,0]

    for i in range(3):

        # ipdb.set_trace()

        pos_wrt_objframe = np.array([0,0,0]).astype('float')
        pos_wrt_objframe[2-i] += (bounding_box[-1-2*i] + offset)
        pos_wrt_global = np.matmul(rot_mat, pos_wrt_objframe) + obj_pose[0:3]

        r_ = R.from_matrix(target_pose)

        # print(np.append(pos_wrt_global, r_.as_quat()))

        grasps.append(np.append(pos_wrt_global, r_.as_quat()))

        # ipdb.set_trace()

        if incupboard:
            pos_wrt_objframe2 = np.array([0,0,0]).astype('float')
            pos_wrt_objframe2[2-i] += (bounding_box[-1-2*i] + offset + 0.3)
            pos_wrt_global2 = np.matmul(rot_mat, pos_wrt_objframe2) + obj_pose[0:3]

            pre_grasps.append(np.append(pos_wrt_global2, r_.as_quat()))

            # ipdb.set_trace()

        target_transform = blank_transform.copy()
        if i ==2 : break

        elif i == 0:
            target_transform[1-i] = -90
        else:
            target_transform[1-i] = 90
        # print(target_transform)

        target_transform = R.from_euler('xyz', target_transform, degrees=True)
        target_pose = np.matmul(target_pose, target_transform.as_matrix())

    target_pose = initial_grasp_pose.copy()
    Rx = R.from_euler('xyz', [-180,0,0], degrees=True)
    target_pose = np.matmul(target_pose, Rx.as_matrix())

    blank_transform = [0,0,0]
    
    for i in range(3):

        pos_wrt_objframe = np.array([0,0,0])
        pos_wrt_objframe[2-i] -= (bounding_box[-1-2*i] + offset)
        pos_wrt_global = np.matmul(rot_mat, pos_wrt_objframe) + obj_pose[0:3]

        r_ = R.from_matrix(target_pose)

        grasps.append(np.append(pos_wrt_global, r_.as_quat()))

        if incupboard:
            pos_wrt_objframe2 = np.array([0,0,0]).astype('float')
            pos_wrt_objframe2[2-i] -= (bounding_box[-1-2*i] + offset + 0.3)
            pos_wrt_global2 = np.matmul(rot_mat, pos_wrt_objframe2) + obj_pose[0:3]

            pre_grasps.append(np.append(pos_wrt_global2, r_.as_quat()))

        target_transform = blank_transform.copy()


        if i ==2 : break

        elif i == 0:
            target_transform[1-i] = 90
        else:
            target_transform[1-i] = 90

        target_transform = R.from_euler('xyz', target_transform, degrees=True)
        target_pose = np.matmul(target_pose, target_transform.as_matrix())

    if not incupboard:
        for s in grasps:
            pos_wrt_global2 = obj_pose[0:3].copy()
            pos_wrt_global2[2] += 0.3
            Rx = R.from_euler('xyz', [-180,0,90], degrees=True)

            pre_grasps.append(np.append(pos_wrt_global2, Rx.as_quat()))
    
    return grasps, pre_grasps
