import numpy as np
import cv2
"""文档: https://horizonrobotics.feishu.cn/wiki/LEFSw8QXkiFBYxk8PvUcdxldnAB?from=from_copylink"""


hook_map = { 
    0: 0,    # ignore 模型不会输出这个类别
    1: 1,    # road
    2: 2,    # sidewalk
    3: 7,    # curb
    4: 9,    # building
    5: 11,   # vegetation
    6: 12,   # terrain
    7: 14,   # fence
    8: 15,   # pole
    9: 16,   # trafficlight
    10: 17,  # trafficsign
    11: 22,  # vehicles
    12: 28,  # pedestrian
    13: 19,  # obstacle
    14: 8,   # bump
    15: 41,  # cone
    16: 10,  # trunk
    17: 42,  # ground lock
}


LEARNING_MAP_BEISAI_17_v5 = {
    0: 20,   # ignore
    1: 1,   # "road"
    2: 2,   # "sidewalk"
    3: 1,   # "laneline" mapped to "road"
    4: 1,   # "stopline" mapped to "road"
    5: 1,   # "crossline" mapped to "road"
    6: 1,   # "lanearrow" mapped to "road"
    7: 3,   # "curb"
    8: 14,  # "bump"
    9: 4,   # "building" 
    10: 16, # "trunk" mapped to "trunk", different with 15 class
    11: 5,  # "vegetation"
    12: 6,  # "terrain"
    13: 6,  # "separation" mapped to "terrain"
    14: 7,  # "fence"
    15: 8,  # "pole"
    16: 9,  # "trafficlight"
    17: 10, # "trafficsign"
    18: 13, # "electricity_box" mapped to "obstacle"
    19: 7,  # "Flowerbed mapped to "fence"
    20: 15, # "Cone" 
    21: 13, # "Water-filled_barrier" mapped to "obstacle"
    22: 13, # "Parking_reserve_barrier" mapped to "obstacle"
    23: 17, # "Ground_lock" mapped to "Ground_lock", different with v16
    24: 1,  # "Parking_line" mapped to "road"
    25: 13, # "Limitator" mapped to "obstacle"
    26: 13, # "blocking_objects" mapped to "obstacle"
    27: 11, # "bus" mapped to "vehicles"
    28: 11, # "truck" mapped to "vehicles"
    29: 11, # "car" mapped to "vehicles"
    30: 11, # "construction_vehicle" mapped to "vehicles"
    31: 11, # "bicycle" mapped to "vehicles"
    32: 11, # "motorcycle" mapped to "vehicles"
    33: 12, # "bicyclist" mapped to "pedestrian"
    34: 12, # "motorcyclist" mapped to "pedestrian"
    35: 12, # "pedestrian"
    36: 11, # "cart" mapped to "vehicles"
    37: 13, # "animal" mapped to "obstacle"
    38: 11, # "pedicab" mapped to "vehicles"
    39: 13, # "moving_object" mapped to "obstacle"
    40: 20, # "shadow" mapped to "ignore"
    41: 20, # Ignore mapped to "ignore"
}


color_map = {
    0: [0, 0, 0],        # ignore
    1: [255, 0, 255],    # "road"
    2: [75, 0, 75],      # "sidewalk"
    3: [75, 0, 175],     # "curb"
    4: [0, 200, 255],    # "building"
    5: [0, 175, 0],      # "vegetation"
    6: [80, 240, 150],   # "terrain"
    7: [50, 120, 255],   # "fence"
    8: [150, 240, 255],  # "pole"
    9: [0, 0, 255],      # "trafficlight"
    10: [255, 0, 0],     # "trafficsign"
    11: [245, 150, 100], # "vehicles" 
    12: [30, 30, 255],   # "pedestrian"
    13: [255, 150, 255], # "obstacle"
    14: [75, 0, 125],    # "bump"
    15: [170, 255, 150], # "Cone"
    16: [0, 60, 135],    # "trunk"
    17: [255, 255, 50],  # "Ground_lock"
}


labels_str = {
    0: "ignore",
    1: "road",
    2: "sidewalk",
    3: "curb",
    4: "building",
    5: "vegetation",
    6: "terrain",
    7: "fence",
    8: "pole",
    9: "trafficlight",
    10: "trafficsign",
    11: "vehicles",
    12: "pedestrian",
    13: "obstacle",
    14: "bump",
    15: "cone",
    16: "trunk",
    17: "ground lock",
}


def get_labels_color(labels: np.ndarray) -> np.ndarray:
    """将语义标签(0-17)映射为RGB颜色"""
    colors = np.array([color_map[label] for label in np.unique(labels)])
    return colors[np.searchsorted(np.unique(labels), labels)]


def swap_keys_values(d: dict) -> dict:
    """反转字典key value"""
    return {v: k for k, v in d.items()}


def inv_hook_map(labels: np.ndarray) -> np.ndarray:
    """用 hook_map 反向映射标签至 0-17 """
    inv_hook_map = swap_keys_values(hook_map)
    inv_labels = np.vectorize(inv_hook_map.__getitem__)(labels)
    return inv_labels


def gen_color_example(example_save_path):
    """生成色卡示例"""
    image = np.ones((150, 1200, 3), dtype=np.uint8) * 255
    for idx, (label, color) in enumerate(color_map.items()):
        row, col = divmod(idx, 6)
        top_left = (col * 200, row * 50)
        bottom_right = ((col + 1) * 200, (row + 1) * 50)
        
        # 填充颜色
        cv2.rectangle(image, top_left, bottom_right, color, -1)
        
        # 添加文本
        text = f"{label} {labels_str[label]}"
        position = (top_left[0] + 5, top_left[1] + 20)  # 调整位置
        cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA) # 轮廓
        cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

    # 保存图像
    cv2.imwrite(example_save_path, image)
    

def combine_point_cloud_with_color(points: np.ndarray, labels: np.ndarray, labels_type: int=43) -> np.ndarray:
    """
    Generate an array containing point cloud data with associated BGR color information based on labels.

    This function processes point cloud data, which may include not only XYZ coordinates but also additional 
    features such as reflectance intensity, scan time, and beam positions. It combines this data with color 
    information derived from the provided labels. The color assignment depends on the label of each point. 
    The function supports two types of labels, 43 and 17, and applies different mappings or assertions based 
    on the selected label type.

    Args:
        points (np.ndarray): A numpy array of shape (N, C) representing the point cloud data,
                             where N is the number of points and C is the number of channels (at least 3 for XYZ).
        labels (np.ndarray): A numpy array of shape (N,) containing the labels for each point in the point cloud.
        labels_type (int, optional): An integer indicating the label type. Supported types are 43 and 17. 
                                     Defaults to 43.

    Returns:
        np.ndarray: An augmented numpy array of shape (N, C+3), where each row contains the original point cloud data
                    (first C elements) and the corresponding BGR color values (last three elements) for each point.

    Raises:
        AssertionError: If the `labels_type` is not one of the supported types (43 or 17).
                        If `labels_type` is 17, it also asserts that the maximum label value does not exceed 17.
                        If the number of points does not match the number of labels.

    Note:
        The `inv_hook_map` function is used to transform labels when `labels_type` is 43.
        The `get_labels_color` function should provide the mapping from labels to BGR colors.
    """
    assert labels_type in [43, 17]
    assert points.shape[0] == labels.shape[0]
    if labels_type == 43:
        labels = inv_hook_map(labels)
    elif labels_type == 17:
        assert labels.max() <= 17
    colors = get_labels_color(labels)
    return np.concatenate((points, colors), axis=-1)
    

import numpy as np


ELE_LIST = [
    14.436,
    13.535,
    13.082,
    12.624,
    12.165,
    11.702,
    11.239,
    10.771,
    10.305,
    9.830,
    9.356,
    8.880,
    8.401,
    7.921,
    7.438,
    6.953,
    6.467,
    5.978,
    5.487,
    4.996,
    4.501,
    4.007,
    3.509,
    3.013,
    2.512,
    2.013,
    1.885,
    1.761,
    1.637,
    1.511,
    1.386,
    1.258,
    1.13,
    1.008,
    0.88,
    0.756,
    0.63,
    0.505,
    0.379,
    0.251,
    0.124,
    0.000,
    -0.129,
    -0.254,
    -0.380,
    -0.506,
    -0.632,
    -0.760,
    -0.887,
    -1.012,
    -1.141,
    -1.266,
    -1.393,
    -1.519,
    -1.646,
    -1.773,
    -1.901,
    -2.027,
    -2.155,
    -2.282,
    -2.409,
    -2.535,
    -2.663,
    -2.789,
    -2.916,
    -3.044,
    -3.172,
    -3.299,
    -3.425,
    -3.552,
    -3.680,
    -3.806,
    -3.933,
    -4.062,
    -4.190,
    -4.318,
    -4.444,
    -4.571,
    -4.699,
    -4.824,
    -4.951,
    -5.081,
    -5.209,
    -5.336,
    -5.463,
    -5.589,
    -5.718,
    -5.843,
    -5.968,
    -6.100,
    -6.607,
    -7.117,
    -7.624,
    -8.134,
    -8.640,
    -9.149,
    -9.652,
    -10.160,
    -10.665,
    -11.170,
    -11.672,
    -12.174,
    -12.673,
    -13.173,
    -13.67,
    -14.166,
    -14.66,
    -15.154,
    -15.645,
    -16.135,
    -16.622,
    -17.106,
    -17.592,
    -18.072,
    -18.548,
    -19.030,
    -19.501,
    -19.978,
    -20.445,
    -20.918,
    -21.379,
    -21.848,
    -22.304,
    -22.768,
    -23.219,
    -23.678,
    -24.123,
    -25.016,
] + list(np.linspace(-90, 90, 400))


"""
0: 忽略
1: 所有地面
4: 建筑
5: 植被
7: fence
8: 杆子+标志牌+交通灯
11: 车
12: 人
13: 障碍物
16: 树干
"""
img_seg_map = {
    0: 1,     # road
    1: 1,     # sidewalk
    2: 5,     # vegetation
    3: 1,     # terrain
    4: 8,
    5: 8,
    6: 8,
    7: 1,
    8: 1,
    9: 12,
    10: 12,
    11: 11,
    12: 11,
    13: 11,
    14: 11,
    15: 11,
    16: 11,
    17: 11,
    18: 4,
    19: 7,
    20: 0,
    21: 13,
    22: 13,
    23: 8,
    24: 1,
    25: 1,
    26: 1,
    27: 1,
    28: 1,
    29: 1,
    30: 1,
    31: 1,
    32: 1,
    33: 13,
    34: 13,
    35: 13,
    36: 13,
    37: 13,
    38: 0,
    39: 0,
    40: 0,
    41: 0,
    42: 0,
}

lidar_seg_map = {
    0: 0,
    1: 1,
    2: 1,
    3: 1,
    4: 4,
    5: 5,
    6: 1,
    7: 7,
    8: 8,
    9: 8,
    10: 8,
    11: 11,
    12: 12,
    13: 13,
    14: 1,
    15: 13,
    16: 16,
    17: 13,
}


import cv2


def draw_text_with_outline(img, text, font_scale=1.0, text_x=10, text_y=10):
    """会改变传入的img"""
    # 设置字体，大小和两个颜色（一种用于文本，一种用于轮廓）
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2  # 可根据需要调整字体的粗细
    outline_thickness = 3  # 描边的粗细
    text_color = (255, 255, 255)  # 白色文本
    outline_color = (0, 0, 0)  # 黑色轮廓

    # 获取文本框的大小
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

    # 计算文本的起始位置（左上角）
    text_x = text_x
    text_y = text_size[1] + text_y  # 从上边界文本高度+10像素的地方开始

    # 首先绘制文本的轮廓
    cv2.putText(img, text, (text_x, text_y), font, font_scale, outline_color, outline_thickness, lineType=cv2.LINE_AA)

    # 然后在轮廓上面绘制文本
    cv2.putText(img, text, (text_x, text_y), font, font_scale, text_color, thickness, lineType=cv2.LINE_AA)

    return img