import numpy as np
from pyquaternion import Quaternion
import os.path as osp
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

from occupancy_grid import Map
from bresenhan_nd import bresenhamline


def homogenize(x):
    """
    Convert a matrix of points into homogeneous coordiante
    """
    return np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)


def get_pointcloud_slice(sample_token: str, nusc: NuScenes, chosen_height: float = 0.5) -> np.ndarray:
    """
    Get a slice (at a certain height above the ground) of pointcloud in ego vehicle frame
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

    # get point at a certain height
    mask_z = np.logical_and(pc.points[2, :] > 0.9 * chosen_height, pc.points[2, :] < 1.1 * chosen_height)
    pc.points = pc.points[:, mask_z]

    return pc.points[:3, :].T


if __name__ == '__main__':
    # load a scene
    dataroot = osp.relpath("../../../Dataset") 
    nusc = NuScenes(version='v1.0-mini', dataroot=osp.join(dataroot, "v1.0-mini"), verbose=False)
    scene = nusc.scene[1]

    env_map = Map(500, 500, resolution=1)

    sample_token = scene['first_sample_token']
    w_M_0 = np.array([])  # pose of the first ego frame with respect to the world frame
    f0_M_i = np.array([])  # pose of i-th ego frame with respect to the first ego frame

    for i in tqdm(range(scene['nbr_samples'])):
        sample = nusc.get('sample', sample_token)
        lidar_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        ego_pose = nusc.get('ego_pose', lidar_rec['ego_pose_token'])

        if i == 0:
            # compute pose of first ego frame in world frame
            w_M_0 = np.eye(4)
            w_M_0[:3, :3] = Quaternion(ego_pose['rotation']).rotation_matrix
            w_M_0[:3, 3] = ego_pose['translation']
        else:
            # compute pose of i-th ego frame in world frame
            w_M_i = np.eye(4)
            w_M_i[:3, :3] = Quaternion(ego_pose['rotation']).rotation_matrix
            w_M_i[:3, 3] = ego_pose['translation']
            # compute pose of i-th ego frame in the first ego frame
            f0_M_i = np.linalg.inv(w_M_0) @ w_M_i  # TODO: note to inverse a matrix use np.linalg.inv [w_M_i = f0_M_i*w_M_0 ]

        # get point cloud in i-th ego frame
        points_3d = get_pointcloud_slice(sample_token, nusc, chosen_height=0.5)  # <float: num_points, 3>

        # map point cloud from i-th ego frame to the first ego frame
        if i > 0:
            points_3d = f0_M_i.dot(homogenize(points_3d).T).T # TODO

        if i == 0:
            vehicle_xy = np.zeros(2)
        else:
            vehicle_xy = np.array([f0_M_i[0,-1],f0_M_i[1,-1]])  # TODO: get xy position of i-th ego frame in the first ego frame
        vehicle_xy += 200  # for displaying purpose

        for j in range(points_3d.shape[0]):
            point = points_3d[j, :2] + 200  # for displaying purpose
            if np.all(point < 300):
                env_map.update_log_odds(point[1], point[0], occupied=True)
                start_pt = np.array([[int(vehicle_xy[0]), int(vehicle_xy[1])]])
                end_pt = np.array([[int(point[0]), int(point[1])]])
                bresenham_path = bresenhamline(start_pt, end_pt, max_iter=-1)

                for bres_pt in bresenham_path:
                    env_map.update_log_odds(bres_pt[1], bres_pt[0], occupied=False)

        # move on the next frame
        sample_token = sample['next']

    env_map.visualize()

