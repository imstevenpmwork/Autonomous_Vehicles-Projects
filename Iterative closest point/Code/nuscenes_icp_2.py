import numpy as np
import open3d as o3d
from pyquaternion import Quaternion
import matplotlib.pyplot as plt

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

import copy
import os.path as osp
from tqdm import tqdm
from typing import List


def draw_registration_result(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud, transformation: np.ndarray):
    # transformation maps source to target
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0, 0])  # rgb
    target_temp.paint_uniform_color([0, 1, 0])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def get_pointcloud(sample_token: str, nusc: NuScenes) -> np.ndarray:
    """
    Get pointcloud, then transform it to ego vehicle frame and perform some filtering
    :param sample_token:
    :param nusc:
    :return: points_3d <float: num_points, 3>  # xyz
    """
    # get pointcloud from file
    sample = nusc.get('sample', sample_token)
    pointsensor = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    pc = LidarPointCloud.from_file(nusc.get_sample_data_path(sample['data']['LIDAR_TOP']))

    # eleminate points too close
    mask_x = np.logical_and(pc.points[0, :] < 1, pc.points[0, :] > -1)
    mask_y = np.logical_and(pc.points[1, :] < 1.5, pc.points[1, :] > -1.5)
    mask = np.logical_and(mask_x, mask_y)
    pc.points = pc.points[:, np.logical_not(mask)]

    # Transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
    cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc.translate(np.array(cs_record['translation']))

    # # eleminate points on the ground
    mask_z = pc.points[2, :] < 0.1
    pc.points = pc.points[:, np.logical_not(mask_z)]

    return pc.points[:3, :].T


def render_merged_pointcloud(scene_token: str, nusc: NuScenes, all_ego_pose: List[np.ndarray]):
    scene = nusc.get('scene', scene_token)
    print('rendering merged cloud')
    sample_token = scene['first_sample_token']
    merged_cloud = o3d.geometry.PointCloud()

    for i in tqdm(range(scene['nbr_samples'])):
        # get information of this sample
        sample = nusc.get('sample', sample_token)

        if i == 0:
            # initialize merged_cloud with the first pointcloud
            merged_cloud.points = o3d.utility.Vector3dVector(get_pointcloud(sample_token, nusc))
            # move on the next frame
            sample_token = sample['next']
            continue

        # get the source pointcloud
        src_cloud = o3d.geometry.PointCloud()
        src_cloud.points = o3d.utility.Vector3dVector(get_pointcloud(sample_token, nusc))

        # map source cloud to world frame
        src_cloud.transform(all_ego_pose[i])

        # merge
        merged_points_3d = np.vstack((np.asarray(merged_cloud.points), np.asarray(src_cloud.points)))
        merged_cloud.points = o3d.utility.Vector3dVector(merged_points_3d)

        # down sample merge_cloud if it get too big
        num_points = np.asarray(merged_cloud.points).shape[0]
        if num_points > 3e5:
            merged_cloud = merged_cloud.voxel_down_sample(voxel_size=0.05)

        # move on the next frame
        sample_token = sample['next']

    # render
    o3d.visualization.draw_geometries([merged_cloud])


def render_trajectory(all_ego_pose: List[np.ndarray]):
    def draw_rect(selected_corners, color: str = 'b'):
        prev = selected_corners[-1]
        for corner in selected_corners:
            ax.plot([prev[0], corner[0]], [prev[1], corner[1]], color=color, linewidth=2.0)
            prev = corner

    fig, ax = plt.subplots()
    l1 = 3.427
    l2 = 0.657
    w = 1.73
    src_corners = np.array([
        [l1,        w / 2.0,        0,      1],
        [-l2,       w / 2.0,        0,      1],
        [-l2,       -w / 2.0,       0,      1],
        [l1,        -w / 2.0,       0,      1],
        [0.5*l1,         0,              0,      1],
        [l1,        0,              0,      1]
    ])
    for pose in all_ego_pose:
        w_corners = (pose @ src_corners.T).T
        draw_rect(w_corners[:-2, :2])
        ax.plot(w_corners[-2:, 0], w_corners[-2:, 1], color='r', linewidth=2)
    ax.set_title('ego vehicle trajectory')
    plt.show()


def main():
    # load a scene
    dataroot = osp.relpath("../../../Dataset")
    nusc = NuScenes(version='v1.0-mini', dataroot=osp.join(dataroot, 'v1.0-mini'), verbose=False)
    scene = nusc.scene[0]

    # initialize target cloud
    target_cloud = o3d.geometry.PointCloud()
    # initialize the guess of ICP solution
    trans_init = np.eye(4)
    trans_init[0, 3] = 4.0

    icp_correspondences_threshold = 0.15  # maximum distance between correspondences

    w_M_target = np.eye(4)
    all_ego_pose = [w_M_target]  # for rendering merged point cloud


    sample_token = scene['first_sample_token']
    for i in range(scene['nbr_samples']):
        # get information of this sample
        sample = nusc.get('sample', sample_token)

        if i == 0:
            # get the first pointcloud and assign to target_cloud then move on to the next frame
            points_3d = get_pointcloud(sample_token, nusc)
            target_cloud.points = o3d.utility.Vector3dVector(points_3d)
            # move on the next frame
            sample_token = sample['next']
            continue

        # get the source pointcloud
        src_cloud = o3d.geometry.PointCloud()  # TODO: create an instance of open3d's PointCloud class
        points_3d = get_pointcloud(sample_token,nusc)  # TODO: get pointcloud using get_pointcloud
        src_cloud.points = o3d.utility.Vector3dVector(points_3d) # TODO: assign points_3d to src_cloud.points

        # invoke ICP
        reg_p2p = o3d.pipelines.registration.registration_icp(
            src_cloud, target_cloud, icp_correspondences_threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        print(reg_p2p)
        print('target_M_src: \n', reg_p2p.transformation)

        # compute the pose of the current ego frame w.r.t to the world frame 
        w_M_src = w_M_target.dot(reg_p2p.transformation) # TODO: hint using target_M_src and w_M_target
        all_ego_pose.append(w_M_src)

        #
        # update
        #
        w_M_target = w_M_src  # TODO: update the target frame with the src frame
        trans_init = reg_p2p.transformation   # update guess of ICP solution
        target_cloud = src_cloud  # update target_cloud with src_cloud

        # move on the next frame
        sample_token = sample['next']
    
    render_trajectory(all_ego_pose)

    render_merged_pointcloud(scene['token'], nusc, all_ego_pose)

    nusc.render_scene(scene['token'])

    print("DONE")


if __name__ == '__main__':
    main()

