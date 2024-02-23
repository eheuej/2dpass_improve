import json
import os
import numpy as np
import struct
import cv2
from deocclusion import Deocclusion


def init_deocc_factory(clip_path):
    calibration = load_calib(clip_path)
    target_camera_calibration = calibration['camera_front_json']
    # 对齐内参格式
    target_camera_calibration['distort'] = {'param': target_camera_calibration['distort']}
    proj_config = {"calib_all": target_camera_calibration}
    deocc = Deocclusion(
        src_w=3840,
        src_h=2160,
        proj_config=proj_config,
        lidar2cam=np.array(calibration['lidar_top_2_camera_front'])
    )
    deocc.create_filters_from_yaml("/home/users/kaining.cui/workspace/hdflow/examples/configs/pilot_3d_data_aug/config/deocc_config.yaml", 
                                   version="multi_v2")
    return deocc


def load_calib(clip_path):
    with open(os.path.join(clip_path, "calibration.json"), 'r') as f:
        calibration = json.load(f)
    return calibration


def vcs2lidar_trans(points, clip_path):
    lidar2vcs = np.array(load_calib(clip_path)['lidar_top_2_vcs']['T'])
    vcs2lidar = np.linalg.inv(lidar2vcs)
    rot = vcs2lidar[:3, :3]
    trans = vcs2lidar[:3, 3].reshape(-1, 1)
    t_points = ((rot @ points[:, :3].T) + trans).T
    t_points = np.concatenate((t_points, points[:, 3].reshape(-1, 1)), axis=1)
    return t_points


def save_numpy_to_pcd_binary_xyz_intensity(filename, array):
    """
    将包含 x, y, z, i 的 numpy 数组保存为二进制格式的 PCD 文件。

    参数:
        filename (str): 要保存的 PCD 文件的名称。
        array (np.ndarray): 包含点云数据的 numpy 数组，应该有 4 列，分别对应 x, y, z, 和 i。

    返回:
        None
    """
    with open(filename, 'wb') as file:
        # 写入 PCD 文件头部
        file.write(b"VERSION .7\n")
        file.write(b"FIELDS x y z intensity\n")
        file.write(b"SIZE 4 4 4 4\n")
        file.write(b"TYPE F F F F\n")  # 注意：所有字段都是浮点型
        file.write(b"COUNT 1 1 1 1\n")
        file.write(f"WIDTH {len(array)}\n".encode())
        file.write(b"HEIGHT 1\n")
        file.write(b"VIEWPOINT 0 0 0 1 0 0 0\n")
        file.write(f"POINTS {len(array)}\n".encode())
        file.write(b"DATA binary\n")

        # 将点云数据以二进制格式写入文件
        for point in array:
            x, y, z, i = point
            # 将 x, y, z 和 i 打包成二进制格式
            packed_data = struct.pack('ffff', x, y, z, i)
            file.write(packed_data)
            


def example(): 
    # example_path = "/horizon-bucket/4DLABEL/users/fzeyang.zhao/multimod_test_0119/clips/UTHS6_20230904/UTHS6_20230904_1693811315136-1693811373931"
    example_path = './'
    print("init deocc factory...")
    deocc = init_deocc_factory(example_path)  # TODO: 包含读取内外参，需要修改
    print("load points..")
    points = np.fromfile(os.path.join(example_path, 'occupancy3d', "1693811315800.bin"), dtype=np.float32).reshape(-1, 4)
    points = vcs2lidar_trans(points, example_path)  # TODO: 点云坐标系转换
    
    print("load image...")
    image = cv2.imread(os.path.join(example_path, 'camera_front', '1693811315800.jpg'))
    print("deocc...")
    deocc_points, _, filter_ids = deocc.call_xyz(points)
    print(f'filter points: {deocc_points.shape[0]}/{points.shape[0]}')
    render_image = deocc.utils.project_render(image, deocc_points[:, :3])
    cv2.imwrite("./example.png", render_image)
    points[filter_ids, 3] = 255
    points[~filter_ids, 3] = 0
    print("Saving image in ./example.png")
    save_numpy_to_pcd_binary_xyz_intensity("./example.pcd", points)
    
if __name__ == "__main__":
    example()