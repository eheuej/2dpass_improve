import os
from matplotlib import pyplot as plt
import cv2
import numpy as np
import numba.cuda as cuda
import torch
import yaml
from typing import Any, Union, List
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from skimage.morphology import dilation, disk
from utils import get_labels_color, img_seg_map, lidar_seg_map, draw_text_with_outline


class ProjectUtils:
    def __init__(self, 
                 src_h, src_w, 
                 save_path=None, 
                 device=None,
                 proj_config=None,
                 lidar2cam=None,
                 **kwargs):
        self.src_w = src_w
        self.src_h = src_h
        self.device = device
        self.save_path = save_path
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)

        self.intrinsic = None
        self.dist_coeffs = None
        self.rotation_mat3x3 = np.eye(3)
        self.rotation_vec, _ = cv2.Rodrigues(self.rotation_mat3x3)
        self.transform_vec = np.array(
            [[0.0], [0.0], [0.0]], dtype=np.float32)
        if proj_config is not None:
            self.read_proj_config(proj_config)
        if lidar2cam is not None:
            self.load_lidar2cam(lidar2cam)
        self.color_map = self.init_color_map()
        
    def get_dst_intrinsic_by_shape(self, dst_h: int, dst_w: int):
        k = np.copy(self.intrinsic)
        k[0, :] = k[0, :] * (dst_w/self.src_w)
        k[1, :] = k[1, :] * (dst_h/self.src_h)
        return k
    
    def get_dst_intrinsic_by_scale(self, fy: float, fx: float):
        """通过缩放比例获取内参矩阵"""
        k = np.copy(self.intrinsic)
        k[0, :] = k[0, :] * (fx)
        k[1, :] = k[1, :] * (fy)
        return k

    def to_cam(self, cloud_xyz: np.ndarray) -> np.ndarray:
        """从Lidar坐标系投影到相机坐标系"""
        return (np.dot(self.rotation_mat3x3, cloud_xyz.T) + self.transform_vec).T

    def project_uv(self, cloud_xyz: np.ndarray, dst_h: int=None, dst_w: int=None):
        dst_h = self.src_h if dst_h is None else dst_h
        dst_w = self.src_w if dst_w is None else dst_w
        k = self.get_dst_intrinsic_by_shape(dst_w=dst_w, dst_h=dst_h)
        cloud = np.ascontiguousarray(cloud_xyz[:, :3])
        imagepoints, _ = cv2.projectPoints(objectPoints=cloud,
                                           rvec=self.rotation_vec,
                                           tvec=self.transform_vec,
                                           cameraMatrix=k,
                                           distCoeffs=self.dist_coeffs)
        return imagepoints

    def project_uv_from_cam(self, cloud_cam_xyz: np.ndarray, dst_h: int=None, dst_w: int=None):
        dst_h = self.src_h if dst_h is None else dst_h
        dst_w = self.src_w if dst_w is None else dst_w
        k = self.get_dst_intrinsic_by_shape(dst_w=dst_w, dst_h=dst_h)
        cloud = np.ascontiguousarray(cloud_cam_xyz)
        rvec = np.array([0.0, 0.0, 0.0], dtype=np.float32).reshape(3, 1)  # No rotation
        tvec = np.array([0.0, 0.0, 0.0], dtype=np.float32).reshape(3, 1)  # No translation
        imagepoints, _ = cv2.projectPoints(objectPoints=cloud,
                                           rvec=rvec,
                                           tvec=tvec,
                                           cameraMatrix=k,
                                           distCoeffs=self.dist_coeffs)
        return imagepoints

    def project_uv_xyz(self, xyz: np.ndarray, dst_h: int=None, dst_w: int=None):
        """投影至图像坐标"""
        cam_xyz = self.to_cam(xyz[:, :3])
        uv = self.project_uv(xyz[:, :3], dst_w=dst_w, dst_h=dst_h)
        if self.device is not None:
            uv = torch.tensor(uv, device=self.device)
            cam_xyz = torch.tensor(cam_xyz, device=self.device)
        return uv, cam_xyz
    
    def get_depth_image(self, cloud_xyz: np.ndarray, cloud_uv: np.ndarray=None, cloud_cam_xyz: np.ndarray=None, 
                        dst_h: int=None, dst_w: int=None):
        dst_h = self.src_h if dst_h is None else dst_h
        dst_w = self.src_w if dst_w is None else dst_w
        depth_image = np.zeros((dst_h, dst_w), np.float32)
        if cloud_cam_xyz is None:
            cloud_cam_xyz = self.to_cam(cloud_xyz[:, :3])
        if cloud_uv is None:
            cloud_uv = self.project_uv_from_cam(cloud_cam_xyz, dst_h, dst_w)

        for idx in reversed(np.argsort(cloud_cam_xyz[:, 2])):  # 远到近
            if int(cloud_cam_xyz[idx, 2] / 2 )< 1:
                continue
            distance = cloud_cam_xyz[idx, 2]
            cx = int(cloud_uv[idx, 0, 0])
            cy = int(cloud_uv[idx, 0, 1])
            if dst_w > cx >= 0 and dst_h > cy > 0:
                depth_image[cy, cx] = distance 
        return depth_image
    
    def get_lidarseg_image(self, 
                           cloud_xyz: np.ndarray, 
                           cloud_labels: np.ndarray,
                           dst_h: Union[int, List[int]], 
                           dst_w:  Union[int, List[int]],
                           cloud_cam_xyz: np.ndarray=None,
                           ) -> List[np.ndarray]:
        if isinstance(dst_h, int):
            dst_h = [dst_h]
        if isinstance(dst_w, int):
            dst_w = [dst_w]
        if cloud_cam_xyz is None:
            cloud_cam_xyz = self.to_cam(cloud_xyz)
        seg_img_list = []
        raw_cloud_uv = self.project_uv(cloud_xyz[:, :3], dst_h[0], dst_w[0])
        for h, w in zip(dst_h, dst_w):
            cloud_uv = raw_cloud_uv * (h / dst_h[0])
            seg_img = np.zeros((h, w), dtype=np.uint8)
            for idx in reversed(np.argsort(cloud_cam_xyz[:, 2])):  # 远到近
                if int(cloud_cam_xyz[idx, 2] / 2 )< 1:
                    continue
                cx = int(cloud_uv[idx, 0, 0])
                cy = int(cloud_uv[idx, 0, 1])
                if w > cx >= 0 and h > cy > 0:
                    seg_img[cy, cx] = cloud_labels[idx]
            seg_img_list.append(seg_img)
        return seg_img_list

    
    def get_multiresolution_depth_image(self, cloud_xyz: np.ndarray, 
                                        dst_h_list: List[int], dst_w_list: List[int],
                                        cloud_cam_xyz: np.ndarray=None):
        if cloud_cam_xyz is None:
            cloud_cam_xyz = self.to_cam(cloud_xyz[:, :3])
        depth_image_list = []
        for dst_h, dst_w in zip(dst_h_list, dst_w_list):
            depth_image_list.append(self.get_depth_image(cloud_xyz, 
                                                         cloud_cam_xyz=cloud_cam_xyz, 
                                                         dst_h=dst_h, dst_w=dst_w))
        return depth_image_list
    
    def get_depth_image_tensor(self, cloud_cam_xyz: torch.Tensor, cloud_uv: torch.Tensor=None,
                               dst_h: int=None, dst_w: int=None):
        """基于torch.Tensor的深度图像投影"""
        dst_h = self.src_h if dst_h is None else dst_h
        dst_w = self.src_w if dst_w is None else dst_w
        if cloud_uv is None:
            cloud_uv = self.project_uv_from_cam(cloud_cam_xyz, dst_h, dst_w)
        depth_image = torch.zeros((dst_h, dst_w), dtype=torch.float32, device=self.device)
        cloud_uv = cloud_uv.to(torch.int64).squeeze()
        depth_image[cloud_uv[:, 1], cloud_uv[:, 0]] = cloud_cam_xyz[:, 2]
        
        return depth_image
    
    def project_render(self, 
                       src_img: np.ndarray, 
                       cloud_xyz: np.ndarray=None, 
                       cloud_cam_xyz: np.ndarray=None,
                       cloud_uv: np.ndarray=None,
                       render_radius: int=1, render_thickness: int=-1):
        """
        :param cloud_xyz:  N * 3
        :param cloud_cam_xyz: N * 3
        :param src_img:  H * W * 3
        :return:
        """
        if cloud_cam_xyz is None:
            cloud_cam_xyz = self.to_cam(cloud_xyz)
        if cloud_uv is not None:
            imagepoints = cloud_uv
        else:
            dst_h = src_img.shape[0]
            dst_w = src_img.shape[1]
            imagepoints = self.project_uv_from_cam(cloud_cam_xyz, dst_h, dst_w)

        render_img = src_img.copy()
        min_v, max_v = 2, 100 # world_pts[:, color_axis].min(), world_pts[:, color_axis].max()
        step = (max_v - min_v) / 256.
        color_idx = (np.minimum((cloud_cam_xyz[:, 2] - 100) / step, 255)).astype('uint8')
        for j in np.argsort(cloud_cam_xyz[:, 2]):
            color_tuple = self.color_map[color_idx[j]]
            cx = int(imagepoints[j, 0, 0])
            cy = int(imagepoints[j, 0, 1])
            if dst_w > cx >= 0 and dst_h > cy > 1:
                cv2.circle(render_img, (cx, cy), render_radius, color_tuple, render_thickness)

        return render_img
    
    def project_multiresolution_render(self, 
                                       src_img: np.ndarray, 
                                       cloud_xyz: np.ndarray, 
                                       dst_h_list: List[int],
                                       dst_w_list: List[int],
                                       cloud_cam_xyz: np.ndarray=None,
                                       render_radius: int=1, 
                                       render_thickness: int=-1):
        render_list = []
        if cloud_cam_xyz is None:
            cloud_cam_xyz = self.to_cam(cloud_xyz[:, :3])
        for dst_h, dst_w in zip(dst_h_list, dst_w_list):
            dst_img = cv2.resize(src_img, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)
            render_list.append(self.project_render(dst_img, cloud_xyz, 
                                                   cloud_cam_xyz=cloud_cam_xyz, 
                                                   render_radius=render_radius,
                                                   render_thickness=render_thickness))
        return render_list
    
    def visualize(self, xyz, src_image, save_name=None, write=True, radius=1, thickness=-1):
        if self.save_path is None and write:
            return
        if not isinstance(xyz, list):
            xyz = [xyz]
            assert save_name is not None or not write
        if not isinstance(save_name, list):
            save_name = [save_name]
        if write:
            assert save_name[0] is not None and len(save_name) == len(xyz)
            
        vis_list = []
        for i, xyz_i in enumerate(xyz):
            # 画图
            if isinstance(xyz_i, torch.Tensor):
                xyz_i = xyz_i.cpu().numpy()
            vis_render = self.project_render(
                src_image, xyz_i[:, :3], None, radius, thickness)
            # 保存
            if write:
                save_name_i = save_name[i]
                if not save_name_i.endswith(".png"):
                    save_name_i += ".png"
                save_path_i = os.path.join(self.save_path, save_name_i)
                cv2.imwrite(save_path_i, vis_render)
                # print(f"save: {save_path_i}")
            vis_list.append(vis_render)
        return vis_list
    
    def get_in_img_mask(self, 
                        uv: Union[np.ndarray, torch.Tensor], 
                        dst_h: int=None, 
                        dst_w: int=None, 
                        margin: int=0,
                        ) -> Union[np.ndarray, torch.Tensor]:
        """获取一个mask, 保留在图像中有映射的点, 过滤掉图像外的点云
        如果margin不为0,那么去掉margin长度的边缘点
        """
        dst_h = self.src_h if dst_h is None else dst_h
        dst_w = self.src_w if dst_w is None else dst_w
        if len(uv.shape) == 3:
            uv = uv.squeeze()
        return (uv[:, 0] >= margin) & (uv[:, 0] < dst_w-margin) \
             & (uv[:, 1] >= margin) & (uv[:, 1] < dst_h-margin)

    @staticmethod
    def index2mask(length: int, ids: Union[np.ndarray, torch.Tensor], device) -> torch.Tensor:
        """索引转换成mask"""
        bool_mask = torch.full((length,), False, dtype=bool, device=device)
        bool_mask[ids] = True
        return bool_mask
    
    def calc_maskbox_gap(self, angle: torch.Tensor, f_pixel: float, k_ary: torch.Tensor = None) -> torch.Tensor:
        """
        需要对夹角进行 非线性矫正，
            参考 coeffs
            gap =  f*tan(\thate)

        :param angle: # [line1,line2,line3,line4]
        :param f_pixel: # 用像素单位描述的焦距
        :param k_ary : 是opencv distCoeffs 参数中的 k1 k2 k3 k4 ..
        :return: 与前后的gap [[gap_12,gap_12],[gap_12,gap_23],[gap_23,gap_34],.....[gap_(n-1)n,gap_(n-1)n] ]
        """
        dst_angle = angle.clone()
        # dst_angle = angle + angle * k_i * 2^(i)  i \in [1,2,3,...]
        if k_ary is not None:
            for i, k in enumerate(k_ary):
                dst_angle = dst_angle + k * angle ** (2 * (i + 1))

        gap = torch.floor(f_pixel * torch.tan(dst_angle[:-1] - dst_angle[1:])).to(self.device)  # Ai - A(i+1)
        gap_res = torch.zeros((dst_angle.shape[0], 2), dtype=torch.int, device=self.device)
        
        gap_res[0, 0] = gap[0]
        gap_res[-1, -1] = gap[-1]
        gap_res[1:, 0] = gap
        gap_res[:-1, 1] = gap
        
        return torch.abs(gap_res) + 2
    
    def mask_all(self, cloud_xyz, cloud_uv, cloud_cam, mask):
        return cloud_xyz[mask], cloud_uv[mask], cloud_cam[mask]
    
    def project_rectify(self, PL, RL):
        self.dist_coeffs[:] = 0.0
        self.intrinsic = PL[:3, :3]
        self.rotation_mat3x3 = RL @ self.rotation_mat3x3
        self.rotation_vec, _ = cv2.Rodrigues(self.rotation_mat3x3)
    
    def read_proj_config(self, config):
        cam_info = config["calib_all"]
        K = np.identity(3, np.float32)
        K[0,0] = cam_info['focal_u']
        K[1,1] = cam_info['focal_v']
        K[0,2] = cam_info['center_u']
        K[1,2] = cam_info['center_v']
        self.intrinsic = K
        self.dist_coeffs = np.array(cam_info["distort"]["param"])
    
    def load_lidar2cam(self, lidar2cam):
        self.rotation_mat3x3 = lidar2cam[:3, :3]
        self.rotation_vec, _ = cv2.Rodrigues(self.rotation_mat3x3)
        self.transform_vec = lidar2cam[:3, 3].reshape(-1, 1)
    
    @staticmethod
    def init_color_map():
        cm = plt.get_cmap("gist_ncar")
        v = 0
        colors = {}
        for i in range(256):
            c = cm(v)
            colors[i] = (int(c[2]*255), int(c[1]*255), int(c[0]*255))
            v += 1/256
        return colors


class DeocclusionBase:
    def __init__(self, utils: ProjectUtils, **kwargs):
        self.src_w = utils.src_w
        self.src_h = utils.src_h
        self.utils = utils
        self.device = utils.device
    
    def __call__(self, *args, **kwargs) -> Any:
        raise NotImplementedError
    

class FilterPreprocessor(DeocclusionBase):
    """用于去除相机中看不见的点。"""
    def __init__(self, utils, 
                 fov=120, 
                 dist_min=2, dist_max=255, 
                 filter_below_ground=1.5, 
                 ground_threshold=0.2,
                 **kwargs):
        super(FilterPreprocessor, self).__init__(utils)
        self.limit_angle = np.deg2rad(fov) / 2.0
        self.dist_min = dist_min
        self.dist_max = dist_max
        self.below_ground = filter_below_ground
        self.debug = kwargs.get('debug', False)
        self.ground_threshold = ground_threshold
        if self.debug:
            print(kwargs.keys())
    
    def __call__(self,
                 cloud_xyzilt: torch.Tensor,
                 cloud_uv: torch.Tensor,
                 cloud_cam_xyz: torch.Tensor,
                 dst_h: int,
                 dst_w: int,
                 filter_ids: torch.Tensor=None,
                 **kwargs):
        assert filter_ids is None
        total_mask = torch.ones_like(cloud_cam_xyz[:, 0], dtype=torch.bool)
        
        total_mask &= cloud_cam_xyz[:, 2] > self.dist_min  # 过滤相机背后的点云
        cloud_yaw = torch.arctan(cloud_cam_xyz[:, 0] / cloud_cam_xyz[:, 2])
        total_mask &= torch.abs(cloud_yaw) < self.limit_angle  # 过滤视场角之外的点
        total_mask &= cloud_cam_xyz[:, 2] <= self.dist_max  # 过滤距离过远的点
        total_mask &= self.utils.get_in_img_mask(cloud_uv, dst_h, dst_w, margin=0)  # 过滤相机图像外的点
        
        cloud_xyzilt, cloud_uv, cloud_cam_xyz = self.utils.mask_all(
            cloud_xyzilt, cloud_uv, cloud_cam_xyz, total_mask)
        
        if "cloud_labels" in kwargs.keys():
            cloud_labels = kwargs.get("cloud_labels")[total_mask]
            ground_points = cloud_xyzilt[cloud_labels == 1]
            below_ground_mask = self.get_mask_below_plane(cloud_xyzilt, ground_points)
        else:
            below_ground_mask = cloud_xyzilt[:, 2] > self.below_ground  # 过滤地面以下的点
                
        return_ids = torch.nonzero(total_mask, as_tuple=False).squeeze()[below_ground_mask]
        return return_ids
    
    def get_mask_below_plane(self, cloud_points: torch.Tensor, ground_points: torch.Tensor
                             ) -> torch.Tensor:
        ransac, poly = self.estimate_ground_plane(ground_points)
        distances = self.calculate_distance_to_plane(cloud_points, ransac, poly)
        return distances > -self.ground_threshold
    
    @staticmethod
    def estimate_ground_plane(ground_points: Union[np.ndarray, torch.Tensor]):
        """估计地面平面"""
        # 保证 ground_points 是 NumPy 数组
        ground_points_np = ground_points.cpu().numpy() if isinstance(ground_points, torch.Tensor) else ground_points

        X = ground_points_np[:, :2]  # 取 x 和 y 坐标
        y = ground_points_np[:, 2]   # 取 z 坐标

        poly = PolynomialFeatures(degree=1, include_bias=False)
        X_poly = poly.fit_transform(X)

        ransac = RANSACRegressor()
        ransac.fit(X_poly, y)

        return ransac, poly

    @staticmethod
    def calculate_distance_to_plane(cloud_points: Union[np.ndarray, torch.Tensor],
                                    ransac: RANSACRegressor, 
                                    poly: PolynomialFeatures
                                    ) -> Union[np.ndarray, torch.Tensor]:
        """计算点云到地面平面距离"""
        if isinstance(cloud_points, torch.Tensor):
            device = cloud_points.device
            cloud_points_np = cloud_points.cpu().numpy() 
        else:
            cloud_points_np = cloud_points

        X_poly = poly.transform(cloud_points_np[:, :2])
        plane_z = ransac.predict(X_poly)

        if isinstance(cloud_points, torch.Tensor):
            plane_z = torch.from_numpy(plane_z).to(device)

        distances = cloud_points[:, 2] - plane_z
        return distances


@cuda.jit
def dynamic_mask_kernel_v2(cloud_uv,
                           depth,
                           layer_len,
                           max_gap_size, max_kernel_size,
                           img_w, img_h,
                           index_map,
                           mask_maps):
    """ 点云投影图像深度 gt 的类 ray casting 的 cuda 核函数。
    
    通过每个点的距离计算一个遮挡范围和遮挡等级, 并且取每个像素遮挡等级的min值(越小等级越高)，从而得
    到一张遮挡图。实际点云投影到图像上时，遮挡图用于检验：点云深度和对应的遮挡等级相差多少，是否在可
    接受范围内。
    该算法以 layer_len 为组分层, 同1m内的点认为具有相同遮挡等级。如果需要更高的精度, 需要修改index_map和
    cur_depth修改映射等级。
    Args:
        cloud_uv (Tensor): 点云的图像坐标 N 2
        depth (Tensor): 点云的深度, 也就是相机坐标系的Z
        max_gap_size (int): 滤波器kernel_size-1/2
        max_kernel_size (int): 遮盖区域大小
        img_w (int): 图像的宽度, 用于判断是否越界
        img_h (int): 图像的高度, 用于判断是否越界
        index_map (Tensor): 遮挡等级的映射表, 根据距离、偏移量来确定
        mask_maps (Tensor): (max_index, H, W) 输出的结果 mask, 后续会对 axis=0 做min处理
    """    
    point_idx, offset = cuda.grid(2)
    if point_idx >= cloud_uv.shape[0] or offset >= max_kernel_size**2: return
    
    base_u = cloud_uv[point_idx, 0] - max_gap_size # 左上角的点, 可能越界
    base_v = cloud_uv[point_idx, 1] - max_gap_size  
    
    offset_u = offset % max_kernel_size  # 根据线程 id 计算偏移量
    offset_v = offset // max_kernel_size
    
    cur_u = base_u + offset_u  # 根据偏移量计算当前处理图像的像素坐标
    cur_v = base_v + offset_v
    if cur_u < 0 or cur_u >= img_w or cur_v < 0 or cur_v >= img_h: return
    
    # 计算偏移距离，用于计算遮挡区域等级，距离中心越远，其覆盖的等级应该越低（在深度变化较快区域有用）
    offset_dist = max(abs(offset_u - max_gap_size), abs(offset_v - max_gap_size))
    
    cur_depth = depth[point_idx] # 获取深度
    if cur_depth > 255 or cur_depth < 0 or offset_dist > index_map.shape[1]: return
    index = index_map[int(cur_depth//layer_len), offset_dist]  # 根据深度和便宜距离共同计算覆盖等级
    mask_maps[index, cur_v, cur_u] = index  # 赋值给遮盖mask
    
    
class MaskFilter(DeocclusionBase):
    def __init__(self, 
                 utils: ProjectUtils, 
                 threshold: int=0,
                 layer_len: float=0.5,
                 mask_size: int=9,
                 base_decay: float=0,
                 power: float=2,
                 temperatures: int=130,
                 max_cover_dist: int=150,
                 max_deocc_dist: int=40,
                 **kwargs):
        super(MaskFilter, self).__init__(utils)
        self.debug = kwargs.get('debug', False)

        # 超参
        self.threshold = threshold
        self.kernel_size = mask_size
        self.max_cover_dist = max_cover_dist
        self.temperature = temperatures
        self.base_decay = base_decay
        self.pow = power
        self.layer_len = layer_len
        self.max_deocc_dist = max_deocc_dist
        
        self.max_index = int(self.max_cover_dist / self.layer_len)
        self.index_map = self.gen_index_map(self._theta_v1)
        if self.debug:
            print(kwargs.keys())

    def __call__(self,
                 cloud_xyzilt: torch.Tensor,
                 cloud_uv: torch.Tensor,
                 cloud_cam_xyz: torch.Tensor,
                 filter_ids: torch.Tensor=None,
                 **kwargs):
        """主调函数

        Args:
            cloud_xyzilt (torch.Tensor): 原始点云
            cloud_uv (torch.Tensor): 原始点云的图像投影坐标
            cloud_cam_xyz (torch.Tensor): 相机坐标系下点云坐标
            filter_ids (torch.Tensor, optional): 前面Filter的过滤索引. Defaults to None.

        Returns:
            _type_: _description_
        """        
        # print(f"before mask: {cloud_uv.shape[0]}")
        if filter_ids is None:
            filter_ids = ((cloud_cam_xyz[:, 2] > 2) & (cloud_cam_xyz[:, 2] <= 255)).nonzero().squeeze()

        sub_cloud_xyzilt = cloud_xyzilt[filter_ids]
        sub_cloud_uv = cloud_uv[filter_ids]
        sub_cloud_cam_xyz = cloud_cam_xyz[filter_ids]
        mask_filter_ids = self.mask_filter(sub_cloud_xyzilt, sub_cloud_uv, sub_cloud_cam_xyz, **kwargs)
        # print(f"after mask: {filter_ids[mask_filter_ids].shape[0]}")
        return filter_ids[mask_filter_ids].squeeze()

    def mask_filter(self, 
                    cloud_xyzilt: torch.Tensor, 
                    cloud_uv: torch.Tensor, 
                    cloud_cam_xyz: torch.Tensor,
                    **kwargs) -> torch.Tensor:
        """_summary_

        Args:
            cloud_xyzilt (torch.Tensor): 原始点云
            cloud_uv (torch.Tensor): 原始点云的图像投影坐标
            cloud_cam_xyz (torch.Tensor): 相机坐标系下点云坐标

        Returns:
            torch.Tensor: 过滤后的索引向量
        """
        savename = kwargs.get('savename', 'default')

        cloud_uv = cloud_uv.squeeze().to(torch.int64)
        
        # 以距离为优先级计算遮挡mask和每个点的index等级
        mask_map = self.dynamic_mask(cloud_uv, cloud_cam_xyz)
        dist = self.index_map[torch.div(cloud_cam_xyz[:, 2], self.layer_len, rounding_mode='floor').to(torch.int64), 0]
        dist_map = (dist - mask_map[cloud_uv[:, 1], cloud_uv[:, 0]])
        
        mask_filter_ids = (dist_map <= self.threshold).nonzero(as_tuple=True)[0].to(torch.int64)
        return mask_filter_ids
    
    def dynamic_mask(self, cloud_uv, cloud_cam_xyz) -> torch.Tensor:
        mask_map = torch.full((self.max_index, self.src_h, self.src_w),   # 初始化遮盖图
                              fill_value=self.max_index, 
                              dtype=torch.int32, device=self.device)

        threads_per_block = (16, 16)
        blocks_per_grid_x = int((cloud_uv.shape[0] + threads_per_block[0] - 1) // threads_per_block[0])
        blocks_per_grid_y = int((self.kernel_size**2 + threads_per_block[1] - 1) // threads_per_block[1])
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
        depth = cloud_cam_xyz[:, 2]
        dynamic_mask_kernel_v2[blocks_per_grid, threads_per_block](
            cloud_uv,
            depth,
            self.layer_len,
            (self.kernel_size-1)//2,
            self.kernel_size,
            self.src_w, self.src_h,
            self.index_map,
            mask_map
        )
        mask_map = mask_map.min(dim=0)[0]
        return mask_map

    def gen_index_map(self, theta):
        index_map = torch.arange(int(256//self.layer_len), dtype=torch.int32, device=self.device)
        index_map[self.max_index:] = self.max_index-1
        max_offset = ((self.kernel_size - 1)/ 2)
        offset_index_map = self._offset_index_cal(max_offset, index_map, theta)
        offset_index_map[int(self.max_deocc_dist//self.layer_len):, 1:] = self.max_index-1
        return offset_index_map
    
    def _offset_index_cal(self, max_offset, index_map, theta):
        # y = \theta {i} * pow(o, power) + i,   其中, i为遮盖等级, o为偏移量, m为最大偏移量
        offset_arange = torch.arange(max_offset+1, device=self.device)
        index_map = index_map.reshape(-1, 1)
        offset_arange = offset_arange.reshape(1, -1)
        offset_index_map = theta(index_map) * torch.pow(offset_arange/max_offset, self.pow) + index_map
        # print(offset_index_map.shape)
        offset_index_map = torch.clip(offset_index_map.to(torch.int32), 0, self.max_index-1)
        return offset_index_map
    
    def _theta_v1(self, index_map):
        return (index_map / self.max_index) * self.temperature + self.base_decay
        

class ColumnFilter(DeocclusionBase):
    def __init__(self, 
                 utils: ProjectUtils, 
                 use_old_ver: bool = True, 
                 search_length: int = 30, 
                 threshold_drop: int = 2, 
                 cover_radius: int = 3,
                 cover_factor: int = 4,
                 search_radius: int = 5, 
                 threshold_depth: int = 3, 
                 threshold_depth_find_again: int = 5, 
                 threshold_depth_filter: int = 1, 
                 max_filter_dist: int=40,
                 filter_method: int = 0, 
                 **kwargs):
        super(ColumnFilter, self).__init__(utils)
        
        # 参数
        self.old = use_old_ver
        self.search_length = search_length
        self.threshold_drop = threshold_drop
        self.cover_radius = cover_radius
        self.cover_factor = cover_factor
        self.max_filter_dist = max_filter_dist
        self.search_radius = search_radius
        self.threshold_depth = threshold_depth
        self.threshold_depth_find_again = threshold_depth_find_again
        self.threshold_depth_filter = threshold_depth_filter
        self.filter_method = filter_method
        
        self.debug = kwargs.get('debug', False)
        if self.debug:
            print(kwargs.keys())
        
    def __call__(self,
                 cloud_xyzilt: torch.Tensor,
                 cloud_uv: torch.Tensor,
                 cloud_cam_xyz: torch.Tensor,
                 filter_ids: torch.Tensor=None,
                 **kwargs) -> Any:
        if filter_ids is None:
            filter_ids = (cloud_cam_xyz[:, 2] > 2).nonzero().squeeze()
        sub_cloud_xyzilt = cloud_xyzilt[filter_ids].cpu()
        sub_cloud_uv = cloud_uv[filter_ids].cpu()
        sub_cloud_cam_xyz = cloud_cam_xyz[filter_ids].cpu()
        if self.old:
            column_filter_ids = self.column_filter_old(sub_cloud_xyzilt, sub_cloud_uv, sub_cloud_cam_xyz)
        return filter_ids[column_filter_ids]
    
    def column_filter_old(self, cloud_xyzilt, cloud_uv, cloud_cam_xyz):
        filter_mask = cloud_cam_xyz[:, 2] < self.max_filter_dist
        cloud_xyz = cloud_xyzilt[:, :3][filter_mask]
        point_pos = cloud_uv[filter_mask]
        cam_xyz = cloud_cam_xyz[filter_mask]
        depth_image = np.zeros((self.src_h, self.src_w), dtype=np.double) 
        index_map = np.zeros((self.src_h, self.src_w), dtype=np.int64) * -1
        # step 1 ,生成 深度image
        for idx in np.argsort(cam_xyz[:, 2]):  # 近到远
            distance = cam_xyz[idx, 2]
            if int(distance) < 2:
                continue
            elif distance > 40:
                break
            cx = int(point_pos[idx, 0, 0])
            cy = int(point_pos[idx, 0, 1])

            if self.src_w > cx and cx >= 0 and self.src_h > cy and cy > 0:
                dist_norm = (self.max_filter_dist - distance) / self.max_filter_dist
                w_min = max(0, cx - int(self.cover_radius + dist_norm * self.cover_factor))
                w_max = min(self.src_w - 1, cx + int(self.cover_radius + dist_norm * self.cover_factor))

                depth_image[cy, w_min:w_max] = distance
                index_map[cy, w_min:w_max] = idx

        # step 2 ,开始处理
        blacklist = list()
        for x in range(0, self.src_w):
            for y in range(self.src_h - 1, 0, -1):
                cur_depth = depth_image[y, x]
                if cur_depth < 0.01 or cur_depth > 255.0:
                    continue
                vec = []
                search_length = int((1 - cur_depth / 255) * self.search_length)
                for search_idx in range(1, search_length):
                    next_y = y - search_idx
                    if next_y <= 0:
                        break

                    next_depth = depth_image[next_y, x]
                    if next_depth > 0.01 and next_depth > cur_depth + self.threshold_depth:
                        vec.append([next_y, x, next_depth])  # y,x,depth
                        continue

                    if 0.01 < next_depth < cur_depth + self.threshold_depth_find_again:
                        for ry, rx, rdepth in vec:
                            # blacklist.add(index_map[ry, rx])
                            depth_image[ry, rx] = 0
                            x_min, x_max = max(0, rx - self.search_radius), min(self.src_w - 1, rx + self.search_radius + 1)
                            y_min, y_max = max(0, ry - self.search_radius), min(self.src_h - 1, ry + self.search_radius + 1)
                            drop_y, drop_x = (depth_image[x_min:x_max, y_min:y_max] > (rdepth - self.threshold_depth_filter)).nonzero()
                            drop_y += y_min
                            drop_x += x_min
                            drop_index = index_map[drop_y, drop_x]
                            blacklist.append(drop_index)                            
                            depth_image[drop_y, drop_x] = 0
                        break
        try:
            blacklist = np.concatenate(blacklist)
            if self.filter_method == 0:
                white_list = set()
                for x in range(0, self.src_w):
                    for y in range(0, self.src_h):
                        if depth_image[y, x] > 0.01 and index_map[y, x] not in blacklist and index_map[y, x] != -1:
                            white_list.add(index_map[y, x])
                white_tensor = torch.tensor(list(white_list), dtype=torch.int64, device=self.device)
                results = (filter_mask.nonzero().squeeze()[white_tensor], (~filter_mask).nonzero().squeeze())
                return torch.concat(results, dim=0)
            elif self.filter_method == 1:
                select_disc = torch.ones((filter_mask.sum()), dtype=torch.bool, device=self.device)
                select_disc[blacklist] = False
                results = (filter_mask.nonzero().squeeze()[select_disc], (~filter_mask).nonzero().squeeze())
                return torch.concat(results, dim=0)
        except Exception as e:
            print(e)
            return torch.arange(start=0, end=cloud_cam_xyz.shape[0])
    
    
class KernelRecover(DeocclusionBase):
    def __init__(self, 
                 utils: ProjectUtils,
                 max_iter: int = 10,
                 max_size: int = 13,
                 base_threshold: int = 0,
                 min_size: int = 7,
                 recover_pow: float = 2,
                 scale: int = 2,
                 norm: int = 20,
                 boundary_thickness: int = 1,
                 **kwargs):
        """搜索邻域恢复丢弃点。
        
        用法: 实例化对象 recover = KernelRecover(**kwargs)后, 调用__call__ recover(**params)

        Args:
            utils (FilterUtils): 辅助工具类, 多个method共用。
            max_iter (int): KernelRecover的最大循环次数。
            max_size (Union[int, List[int]]): 开始搜索的最大核大小，必须是奇数，如果是偶数将会加1调整。
            base_threshold (int): 基础阈值。
            min_size (int): 最小核大小。
            recover_pow (float): 恢复力度指数。
            scale (int): 缩放系数。
            norm (int): 标准化值。
            boundary_thickness (int): 缩小地面seg区域的参数(不恢复边界的gt)
            kwargs: 其他超参通过字典传入（如果有额外的参数，它们将在此处被接收）。
        """
        super().__init__(utils)
        
        if isinstance(max_size, list):
            max_size = max_size[0]
        if max_size % 2 == 0:
            max_size += 1
            print("[WARNING]: KernelRecover's max_size must be odd.")
        
        # 参数
        self.max_iter = max_iter
        self.kernel_size = max_size
        self.base_threshold = base_threshold
        self.min_size = min_size
        self.pow = recover_pow
        self.scale = scale
        self.norm = norm
        self.thickness = boundary_thickness
        self.debug = kwargs.get('debug', False)
        if self.debug:
            print(kwargs.keys())
        
        img_max_key = max(img_seg_map.keys())
        img_map_tensor = torch.full((img_max_key + 1,), fill_value=-1, device=self.device) 
        for k, v in img_seg_map.items():
            img_map_tensor[k] = v
        self.img_seg_map = img_map_tensor
    
    def __call__(
            self,                  
            cloud_xyzilt: torch.Tensor, 
            cloud_uv: torch.Tensor, 
            cloud_cam_xyz: torch.Tensor,
            dst_h: int,
            dst_w: int,
            filter_ids: torch.Tensor,
            seg_img: Union[torch.Tensor, np.ndarray]=None,
            **kwargs,
        ):
        """主调用函数
        
        Args:
            cloud_xyzilt (torch.Tensor): 原始点云坐标
            cloud_uv (torch.Tensor): 点云图像坐标
            cloud_cam_xyz (torch.Tensor): 点云相机坐标系坐标
            filter_ids (torch.Tensor): 去遮挡算法过滤后点云索引
            seg_img (np.ndarray, optional): 若传入, 仅恢复地面点
            
        Returns
            torch.Tensor: 恢复的点云索引, 包括原过滤点云
        
        """
        cloud_xyz = cloud_xyzilt[:, :3]
        kernel_size = self.kernel_size
        if seg_img is not None:
            seg_img = torch.from_numpy(seg_img).to(self.device)
            seg_img = self.img_seg_map[seg_img.to(torch.int64)] != 1
            seg_img = ~self.dilate_seg_mask(seg_img)
            scale = (int(seg_img.shape[0] / dst_h), int(seg_img.shape[1] / dst_w))
        else:
            scale = None
            
        for i in range(self.max_iter):
            # 生成 kernel_recover所需数据
            filter_mask = self.utils.index2mask(cloud_xyz.shape[0], filter_ids, self.device)
            drop_mask = ~filter_mask
            filter_xyz, filter_uv, filter_cam_xyz = cloud_xyz[filter_mask], cloud_uv[filter_mask], cloud_cam_xyz[filter_mask]
            drop_xyz, drop_uv, drop_cam_xyz = cloud_xyz[drop_mask], cloud_uv[drop_mask], cloud_cam_xyz[drop_mask]
            drop_limit = (drop_cam_xyz[:, 2] > 0) & (drop_cam_xyz[:, 1] < 3)
            drop_xyz, drop_uv, drop_cam_xyz = drop_xyz[drop_limit], drop_uv[drop_limit], drop_cam_xyz[drop_limit]
            
            recover_ids_in_drop = self.kernel_recover(kernel_size,
                                                      drop_uv, drop_cam_xyz, 
                                                      filter_uv, filter_cam_xyz,
                                                      dst_h, dst_w, 
                                                      seg_img, scale)
            
            # 返回的恢复索引是基于 drop 点云的，需要恢复为相对于原始点云的索引
            recover_ids = drop_mask.nonzero()[drop_limit][recover_ids_in_drop]
            if recover_ids.shape[0] <= 1:
                if kernel_size == self.min_size:
                    break
                else:
                    kernel_size -= 2
                    continue
            filter_ids = torch.cat((filter_ids, recover_ids.squeeze()), dim=0)
        return filter_ids
    
    def kernel_recover(self, 
                       kernel_size: int,
                       drop_uv: torch.Tensor, drop_cam_xyz: torch.Tensor, 
                       filter_uv: torch.Tensor, filter_cam_xyz: torch.Tensor,
                       dst_h: int, dst_w: int, 
                       seg_mask: torch.Tensor=None, scale=None) -> torch.Tensor:
        """使用GPU执行邻近点恢复来恢复稀疏投影点云。
        
        Args:
            kernel_size (int): 搜索的邻域大小
            threshold (float): 深度阈值
            drop_xyz (torch.Tensor): 经过过滤后丢弃的点云的坐标, 大小为(M, 3)
            filter_xyz (torch.Tensor): 经过过滤后保留的点云的坐标, 大小为(N, 3)
            drop_uv, drop_cam_xyz (torch.Tensor): 丢弃点云的投影坐标
            filter_uv, filter_cam_xyz (torch.Tensor): 过滤点云的投影坐标
            dst_h , dst_w (int): 用于判断点云投影是否越界
            seg_mask (torch.Tensor): 用于限制仅恢复地面点
            device (str): 指定使用的CUDA设备
        
        Returns:
            torch.Tensor: 恢复的点云
        
        """
        # 初始化
        filter_uv = filter_uv.squeeze().to(torch.int64)  # 过滤点的图像坐标
        drop_uv = drop_uv.squeeze().to(torch.int64)  # 丢弃点的图像坐标
        dist = filter_cam_xyz[:, 2]  # 过滤点的深度值
        filter_in_img_mask = self.utils.get_in_img_mask(filter_uv, dst_h, dst_w)  # 在图像范围内的过滤点的mask
        filter_img_uv = filter_uv[filter_in_img_mask]  # 在图像范围内的过滤点的图像坐标uv
        
        # 初始化深度图
        depth_img = torch.zeros((self.src_h, self.src_w), dtype=torch.double, device=self.device)
        depth_img[filter_img_uv[:, 1], filter_img_uv[:, 0]] = dist[filter_in_img_mask]  # 深度图中的点被赋予相应的深度值
        
        # 计算丢弃点在图像中的坐标
        kernel_radius = kernel_size // 2  # kernel的半径
        drop_in_img_mask = self.utils.get_in_img_mask(drop_uv, dst_h, dst_w, kernel_radius)
        drop_img_uv = drop_uv[drop_in_img_mask]  # 在图像范围内的丢弃点的图像坐标
        drop_u, drop_v = drop_img_uv[:, 0], drop_img_uv[:, 1]  # 获取u和v坐标
        unexist_mask = (depth_img[drop_v, drop_u] == 0)  # 用于找出深度图中未被遮挡的丢弃点mask。
        if seg_mask is not None:
            ground_mask = seg_mask[drop_v * scale[0], drop_u * scale[1]]
            unexist_mask &= ground_mask  # 为了方便，将两个mask合并
        
        # 搜索邻域
        search_du, search_dv = torch.meshgrid(torch.arange(-kernel_radius, kernel_radius + 1).to(self.device),
                                              torch.arange(-kernel_radius, kernel_radius + 1).to(self.device),
                                              indexing='ij')
        search_u = search_du + drop_u[unexist_mask].view(-1, 1, 1)
        search_v = search_dv + drop_v[unexist_mask].view(-1, 1, 1)
        search_neighbor = depth_img[search_v, search_u]  # 获取邻域中各点的深度值
        valid_mask = search_neighbor != 0  # 创建有效邻域的mask
        
        # 验证邻域深度值的一致性
        drop_dist = drop_cam_xyz[:, 2][drop_in_img_mask][unexist_mask].view(-1, 1, 1)
        dist_threshold = self.scale * torch.pow(drop_dist/self.norm, self.pow) + self.base_threshold
        
        neighbor_mask = torch.abs(search_neighbor - drop_dist) < dist_threshold  # 与邻域的深度值进行比较
        
        # 计算恢复mask
        recover_mask = valid_mask.sum(dim=(1, 2)) == neighbor_mask.sum(dim=(1, 2))  # 邻域内所有点深度值均匹配的情况
        recover_mask &= valid_mask.sum(dim=(1, 2)) != 0  # 至少有一个匹配的邻域点
        
        # 返回恢复的点云索引 (相对于drop_xyz)
        recover_ids = drop_in_img_mask.nonzero()[unexist_mask][recover_mask]
        
        return recover_ids.squeeze()
    
    def dilate_seg_mask(self, seg_mask):
        if isinstance(seg_mask, torch.Tensor):
            seg_mask = seg_mask.cpu().numpy()
        
        if self.thickness > 1:
            selem = disk(self.thickness)
            seg_mask = dilation(seg_mask, selem)
        return torch.from_numpy(seg_mask).to(self.device)


class DenoiseFilter(DeocclusionBase):
    def __init__(self, utils: ProjectUtils, 
                 max_kernel_size: int=11,
                 min_kernel_size: int=5,
                 noise_threshold: float=3,
                 count_threshold: int=1,
                 **kwargs):
        super().__init__(utils, **kwargs)
        self.kernel_size = max_kernel_size
        self.min_kernel_size = min_kernel_size
        self.noise_threshold = noise_threshold
        self.count_threshold = count_threshold
        self.debug = kwargs.get('debug', False)
        if self.debug:
            print(kwargs.keys())

    
    def __call__(
            self,                  
            cloud_uv: torch.Tensor, 
            cloud_cam_xyz: torch.Tensor,
            filter_ids: torch.Tensor,
            dst_h: int, 
            dst_w: int,
            **kwargs,
        ):
        kernel_size = self.kernel_size
        while kernel_size >= self.min_kernel_size:
            denoise_mask = self.denoise_filter(cloud_uv[filter_ids], cloud_cam_xyz[filter_ids], kernel_size, dst_h, dst_w)
            filter_ids = filter_ids[denoise_mask]
            kernel_size -= 2
        return filter_ids
    
    def denoise_filter(self, cloud_uv, cloud_cam_xyz, kernel_size, dst_h, dst_w):
        """类似于KernelRecover, 但作为去噪声算法"""
        kernel_radius = kernel_size // 2
        cloud_uv = cloud_uv.squeeze().to(torch.long)
        depth_img = torch.zeros((self.src_h, self.src_w), dtype=torch.double, device=self.device)
        depth_img[cloud_uv[:, 1], cloud_uv[:, 0]] = cloud_cam_xyz[:, 2]
        
        in_img_mask = self.utils.get_in_img_mask(cloud_uv, dst_h, dst_w, margin=kernel_radius)
        sub_cloud_uv = cloud_uv[in_img_mask]
        sub_cloud_cam_xyz = cloud_cam_xyz[in_img_mask]
        
        search_du, search_dv = torch.meshgrid(
            torch.arange(-kernel_radius, kernel_radius + 1).to(self.device),
            torch.arange(-kernel_radius, kernel_radius + 1).to(self.device),
        )
        search_u = search_du + sub_cloud_uv[:, 0].view(-1, 1, 1)
        search_v = search_dv + sub_cloud_uv[:, 1].view(-1, 1, 1)
        search_neighbor = depth_img[search_v, search_u]
        valid_mask = search_neighbor != 0 
        
        neighbor_mask = torch.abs(search_neighbor - sub_cloud_cam_xyz[:, 2].reshape(-1, 1, 1)) < self.noise_threshold
        denoise_mask = ((valid_mask & neighbor_mask).sum(dim=(1, 2)) > self.count_threshold) 
        denoise_mask |= (valid_mask.sum(dim=(1, 2)) <= self.count_threshold)
        return in_img_mask.nonzero().squeeze()[denoise_mask]


class SegParsing(DeocclusionBase):
    def __init__(self, utils, **kwargs):
        super().__init__(utils, **kwargs)
        lidar_max_key = max(lidar_seg_map.keys())
        lidar_map_tensor = torch.full((lidar_max_key + 1,), fill_value=-1, device=self.device)  # -1为错误映射
        for k, v in lidar_seg_map.items():
            lidar_map_tensor[k] = v
        self.lidar_seg_map = lidar_map_tensor
        img_max_key = max(img_seg_map.keys())
        img_map_tensor = torch.full((img_max_key + 1,), fill_value=-1, device=self.device)  # -1为错误映射
        for k, v in img_seg_map.items():
            img_map_tensor[k] = v
        self.img_seg_map = img_map_tensor
        
    def __call__(self, 
                 cloud_uv: torch.Tensor,
                 filter_ids: torch.Tensor,
                 cloud_labels: Union[torch.Tensor, np.ndarray]=None,
                 seg_img: Union[torch.Tensor, np.ndarray]=None,
                 dst_h=None,
                 dst_w=None,
                 **kwargs):
        """Seg Label 经过映射后, 判断 Lidar label 和 img label是否相同。

        Args:
            cloud_uv (torch.Tensor): 图像坐标
            filter_ids (torch.Tensor): 前面得到的过滤点索引
            cloud_labels (torch.Tensor, optional): 0-17的标签, 默认为None
            seg_img (torch.Tensor, optional): 图像标签, 40cls的输出, 默认为None

        Returns:
            _type_: _description_
        """
        if cloud_labels is None or seg_img is None:
            return filter_ids
        if isinstance(cloud_labels, np.ndarray):
            cloud_labels = torch.tensor(cloud_labels, device=self.device)
        if isinstance(seg_img, np.ndarray):
            seg_img = torch.tensor(seg_img, device=self.device)
        debug_vis_path = kwargs.get("debug_vis_path")
        src_img = kwargs.get("src_img")
        
        sub_cloud_uv = cloud_uv[filter_ids]
        sub_cloud_uv[..., 1] *= seg_img.shape[0] / dst_h
        sub_cloud_uv[..., 0] *= seg_img.shape[1] / dst_w
        sub_cloud_uv = sub_cloud_uv.to(torch.int64)
        sub_cloud_labels = cloud_labels[filter_ids]
        sub_cloud_labels_map = self.lidar_seg_map[sub_cloud_labels]
        
        seg_img_map = self.img_seg_map[seg_img.to(torch.int64)]
        sub_img_map_in_uv = seg_img_map[sub_cloud_uv[:, 0, 1], sub_cloud_uv[:, 0, 0]]
        
        label_parsing = (sub_cloud_labels_map == sub_img_map_in_uv)
        
        if kwargs.get("debug", False):
            # err_parsing = (~label_parsing) * 12
            vis_cloud_seg = cv2.resize(
                self.vis_cloud_seg_img(sub_cloud_labels_map, sub_cloud_uv, src_img), 
                (dst_w, dst_h), interpolation=cv2.INTER_LINEAR
            )
            # vis_img_seg = self.vis_cloud_seg_img(sub_img_map_in_uv, sub_cloud_uv, src_img)
            seg_img_color = cv2.resize(
                get_labels_color(seg_img_map.flatten().cpu()).reshape(seg_img_map.shape[0], -1, 3).astype(np.uint8),
                (dst_w, dst_h), interpolation=cv2.INTER_LINEAR
            )
            # vis_err = self.vis_cloud_seg_img(err_parsing, sub_cloud_uv, src_img)
            
            vis_cloud_seg = draw_text_with_outline(vis_cloud_seg, "LiDAR Seg", 2)
            # vis_img_seg = draw_text_with_outline(vis_img_seg, "Image Seg (points only)")
            seg_img_color = draw_text_with_outline(seg_img_color, "Image Seg", 2)
            # vis_err = draw_text_with_outline(vis_err, "parsing result")
            
            vis = np.concatenate((vis_cloud_seg, seg_img_color), axis=0)
            cv2.imwrite(debug_vis_path, vis)
        
        return filter_ids[label_parsing.nonzero().squeeze()]
    
    def vis_cloud_seg_img(self, cloud_labels, cloud_uv, img):
        cloud_uv = cloud_uv.squeeze().cpu()
        vis_cloud_seg = img.copy()
        cloud_colors = get_labels_color(cloud_labels.cpu())
        vis_cloud_seg[cloud_uv[:, 1], cloud_uv[:, 0]] = cloud_colors
        return vis_cloud_seg        

class Deocclusion:
    def __init__(self, 
                 src_w: int, src_h: int, 
                 utils: ProjectUtils=None,
                 proj_config: dict=None,
                 lidar2cam: np.ndarray=None,
                 **kwargs):
        """对点云投影到图像进行去遮挡。先通过create_filters初始化pipeline, 再通过__call__或call_xyz调用

        Args:
            src_w (int): 原始图像大小
            src_h (int): 原始图像大小
            utils (ProjectUtils, optional): ProjectUtils对象
            proj_config (dict, optional): 如果没有传入utils, 使用 proj_config 构建 utils. 需要为hdflow 支持的字典参数结构
            lidar2cam (np.ndarray, optional): 如果没有传入utils, 使用 lidar2cam 构建 utils. 需要为4*4 lidar到相机坐标转换矩阵
        """     
        self.src_h = src_h  
        self.src_w = src_w
        self.dst_h = src_h
        self.dst_w = src_w
        
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = kwargs.get('device', device)
        self.debug = kwargs.get('debug', False)
        self.save_path = kwargs.get('save_path', None)
        
        self.utils = utils
        if self.utils is None:
            self.utils = ProjectUtils(
                src_h, src_w, 
                device=self.device, save_path=self.save_path,
                proj_config=proj_config, lidar2cam=lidar2cam
            )
        self.filter_map = {
            'preprocessor': FilterPreprocessor, 
            'mask': MaskFilter, 
            'column': ColumnFilter, 
            'recover': KernelRecover,
            'denoise': DenoiseFilter,
            'segparsing': SegParsing,
        }
        self.filter_pipeline: List[DeocclusionBase] = []
        self.pipeline_str: List[str] = []
        
    def __call__(self, cloud_xyzilt, cloud_uv, cloud_cam_xyz, **kwargs):
        if isinstance(cloud_uv, np.ndarray):
            cloud_uv = torch.tensor(cloud_uv, device=self.device)
            cloud_cam_xyz = torch.tensor(cloud_cam_xyz, device=self.device)
        cloud_xyzilt_tensor = torch.tensor(cloud_xyzilt, device=self.device)
        filter_ids = None
        for i, filter in enumerate(self.filter_pipeline):
            filter_ids = filter.__call__(cloud_xyzilt=cloud_xyzilt_tensor, 
                                         cloud_uv=cloud_uv, 
                                         cloud_cam_xyz=cloud_cam_xyz, 
                                         filter_ids=filter_ids, 
                                         dst_h=self.dst_h,
                                         dst_w=self.dst_w,
                                         **kwargs)
        filtered_point = cloud_xyzilt[filter_ids.cpu()]
        filtered_cam_xyz = cloud_cam_xyz[filter_ids].cpu().numpy()
        return filtered_point, filtered_cam_xyz, filter_ids.cpu().numpy()
    
    def call_xyz(self, cloud_xyzilt, **kwargs):
        """在内部完成坐标转换"""
        cloud_uv, cloud_cam_xyz = self.utils.project_uv_xyz(cloud_xyzilt[:, :3], self.dst_h, self.dst_w)
        return self.__call__(cloud_xyzilt, cloud_uv, cloud_cam_xyz, **kwargs)
    
    def call_debug(self, cloud_xyzilt, **kwargs):
        if isinstance(cloud_uv, np.ndarray):
            cloud_uv = torch.tensor(cloud_uv, device=self.device)
            cloud_cam_xyz = torch.tensor(cloud_cam_xyz, device=self.device)
        cloud_xyzilt_tensor = torch.tensor(cloud_xyzilt, device=self.device)
        filter_ids = None
        filtered_point_list = []
        for i, filter in enumerate(self.filter_pipeline):
            filter_ids = filter.__call__(cloud_xyzilt=cloud_xyzilt_tensor, 
                                         cloud_uv=cloud_uv, 
                                         cloud_cam_xyz=cloud_cam_xyz, 
                                         filter_ids=filter_ids, 
                                         dst_h=self.dst_h,
                                         dst_w=self.dst_w,
                                         **kwargs)
            filtered_point_list.append(cloud_xyzilt[filter_ids.cpu()])
        return filtered_point_list
    
    def create_filters_from_yaml(self, config_path: str, version: str):
        with open(config_path, 'r') as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                return None
        self.create_filters_from_dict(config[version])
    
    def create_filters_from_dict(self, config: dict):
        self.filter_pipeline = []
        filter_order = config.get('order', [])
        self.pipeline_str = filter_order
        self.dst_w = config.get('dst_w', 960)
        self.dst_h = config.get('dst_h', 540)
        
        # 按照定义的顺序创建过滤器实例
        for filter_name in filter_order:
            filter_config = config.get(filter_name)
            filter_class = self.filter_map.get(filter_name)  # 从映射中获取相应的类
            filter_config['debug'] = self.debug
            filter_config['utils'] = self.utils

            if filter_class and filter_config:  # 如果找到了相应的类并且有配置信息
                try:
                    filter_instance = filter_class(**filter_config)  # 创建过滤器实例
                    self.filter_pipeline.append(filter_instance)
                except TypeError as e:
                    print(f"Error initializing filter {filter_name}: {e}")
    
    def set_dst(self, dst_h: int, dst_w: int):
        self.dst_h = dst_h
        self.dst_w = dst_w
