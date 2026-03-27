import os
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import re
from torchvision.datasets import VisionDataset

intrinsic_camera_matrix_filenames = ['intr_1.xml', 'intr_2.xml', 'intr_3.xml']
extrinsic_camera_matrix_filenames = ['extr_1.xml', 'extr_2.xml', 'extr_3.xml']


class Citystreet(VisionDataset):
    def __init__(self, root):
        super().__init__(root)
        # image of shape C,H,W (C,N_row,N_col); xy indexging; x,y (w,h) (n_col,n_row)
        # WILDTRACK has ij-indexing: H*W=480*1440, thus x (i) is \in [0,480), y (j) is \in [0,1440)
        # WILDTRACK has in-consistent unit: centi-meter (cm) for calibration & pos annotation
        self.__name__ = 'Wildtrack'
        self.img_shape, self.worldgrid_shape = [1520, 2704], [768,640]  # H,W; N_row,N_col
        self.num_cam, self.num_frame = 3, 2558
        # world x,y actually means i,j in Wildtrack, which correspond to h,w
        self.worldgrid2worldcoord_mat = np.linalg.inv(np.array([
    [1/76.25, 0,                   352 * 0.8],
    [0,                   1/76.25, 522 * 0.8],
    [0,                   0,                   1       ]
]))
        self.intrinsic_matrices, self.distorts,self.extrinsic_matrices = zip(
            *[self.get_intrinsic_extrinsic_matrix(cam) for cam in range(self.num_cam)])

    def get_image_fpaths(self, frame_range):
        img_fpaths = {cam: {} for cam in range(self.num_cam)}
        for camera_folder in sorted(os.listdir(os.path.join(self.root, 'Image_subsets'))):
            cam = int(camera_folder[-1]) - 1
            if cam >= self.num_cam:
                continue
            for fname in sorted(os.listdir(os.path.join(self.root, 'Image_subsets', camera_folder))):
                frame = int(fname.split('.')[0])
                if frame in frame_range:
                    img_fpaths[cam][frame] = os.path.join(self.root, 'Image_subsets', camera_folder, fname)
        return img_fpaths

    def get_worldgrid_from_pos(self, pos):
        grid_y = pos % 768
        grid_x = pos // 768
        return np.array([grid_x, grid_y], dtype=int)

    def get_intrinsic_extrinsic_matrix(self, camera_i):



        intrinsic_camera_path = os.path.join(self.root, 'calibrations', 'intrinsic')
        intrinsic_file = os.path.join(intrinsic_camera_path, intrinsic_camera_matrix_filenames[camera_i])
        fs = cv2.FileStorage(intrinsic_file, cv2.FILE_STORAGE_READ)


        intrinsic_matrix = fs.getNode('camera_matrix').mat()
        distortion_node = fs.getNode('distortion_coefficients')

        if distortion_node.empty():
            distortion_coeffs = np.zeros((5, 1), dtype=np.float32)
        else:
            distortion_coeffs = distortion_node.mat()

        fs.release()
        extrinsic_file_root = ET.parse(os.path.join(self.root, 'calibrations', 'extrinsic',
                                         extrinsic_camera_matrix_filenames[camera_i])).getroot()
        rvec_text = extrinsic_file_root.find('rvec').find('data').text
        rvec = np.array(list(map(float, rvec_text.replace('\n', ' ').split())), dtype=np.float32)
        tvec_text = extrinsic_file_root.find('tvec').find('data').text
        tvec = np.array(list(map(float, tvec_text.replace('\n', ' ').split())), dtype=np.float32)

        rotation_matrix, _ = cv2.Rodrigues(rvec)
        translation_matrix = tvec.reshape(3, 1)
        extrinsic_matrix = np.hstack((rotation_matrix, translation_matrix))

        return intrinsic_matrix, distortion_coeffs, extrinsic_matrix

    def get_camera_position_in_grid(self, camera_i):
        """返回相机在世界坐标和 grid 坐标下的位置"""
        intrinsic, dist, extrinsic = self.get_intrinsic_extrinsic_matrix(camera_i)

        # 分解外参
        R = extrinsic[:, :3]
        t = extrinsic[:, 3].reshape(3, 1)

        # 相机中心在 world 坐标系下
        camera_center_world = -R.T @ t  # (3,1)
        x, y, z = camera_center_world.ravel()

        # worldcoord -> grid 的矩阵
        worldcoord2grid_mat = np.array([
            [1/76.25, 0, 352 * 0.8],
            [0, 1/76.25, 522 * 0.8],
            [0, 0, 1]
        ])

        # 转到 grid 坐标
        coord_world = np.array([x, y, 1.0])
        coord_grid = worldcoord2grid_mat @ coord_world
        coord_grid /= coord_grid[2]

        return camera_center_world.ravel(), coord_grid[:2]


def test():
    from multiview_detector.utils.projection import get_imagecoord_from_worldcoord
    dataset = Citystreet(os.path.expanduser('D:/cvpr25_tracking/Citystreet_rectify') )
    for i in range(dataset.num_cam):
        cam_world, cam_grid = dataset.get_camera_position_in_grid(i)
        print(f"Camera {i + 1} world pos: {cam_world}, grid pos: {cam_grid}")



if __name__ == '__main__':
    test()