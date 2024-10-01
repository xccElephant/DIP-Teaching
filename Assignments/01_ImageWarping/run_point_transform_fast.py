import os
import time

import cv2
import gradio as gr
import numpy as np

# 初始化全局变量, 存储控制点和目标点
points_src = []
points_dst = []
image = None


# 上传图像时清空控制点和目标点
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()  # 清空控制点
    points_dst.clear()  # 清空目标点
    image = img
    return img


# 记录点击点事件, 并标记点在图像上, 同时在成对的点间画箭头
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]  # 获取点击的坐标

    # 判断奇偶次来分别记录控制点和目标点
    if len(points_src) == len(points_dst):
        points_src.append([x, y])  # 奇数次点击为控制点
    else:
        points_dst.append([x, y])  # 偶数次点击为目标点

    # 在图像上标记点（蓝色：控制点, 红色：目标点）, 并画箭头
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(
            marked_image, tuple(pt), 5, (255, 0, 0), -1
        )  # 蓝色表示控制点, "-1" 表示实心圆
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 5, (0, 0, 255), -1)  # 红色表示目标点

    # 画出箭头, 表示从控制点到目标点的映射
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(
            marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 2
        )  # 绿色箭头表示映射

    return marked_image


# 执行仿射变换
def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    """
    Image deformation method based on linear Moving Least Squares

    Parameters
    ----------
    image : numpy.ndarray
        The input image.
    source_pts : numpy.ndarray
        The source points, shape (n, 2).
    target_pts : numpy.ndarray
        The target points, shape (n, 2).
    alpha : float, optional
        The weight of the target points, by default 1.0.
    eps :float, optional
        A small number to avoid division by zero, by default 1e-8.

    Returns
    -------
        A deformed image : numpy.ndarray
    """

    h, w = image.shape[:2]
    image = image.astype(np.float64)

    # 交换 source_pts 和 target_pts，实现向后映射
    source_pts, target_pts = target_pts.copy(), source_pts.copy()

    # 创建像素网格坐标
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    grid = np.stack((grid_x, grid_y), axis=-1).reshape(-1, 2)  # 形状为 (h*w, 2)

    # 计算权重 w_i
    diff = grid[:, np.newaxis, :] - source_pts[np.newaxis, :, :]  # 形状为 (h*w, n, 2)
    dist2 = np.sum(diff**2, axis=2)  # 形状为 (h*w, n)
    w_i = 1.0 / (dist2**alpha + eps)  # 形状为 (h*w, n)

    # 计算加权中心 p_star 和 q_star
    w_sum = np.sum(w_i, axis=1, keepdims=True)  # 形状为 (h*w, 1)
    p_star = (
        np.sum(w_i[:, :, np.newaxis] * source_pts[np.newaxis, :, :], axis=1) / w_sum
    )  # (h*w, 2)
    q_star = (
        np.sum(w_i[:, :, np.newaxis] * target_pts[np.newaxis, :, :], axis=1) / w_sum
    )  # (h*w, 2)

    # 计算局部坐标系下的坐标差值
    p_hat = source_pts[np.newaxis, :, :] - p_star[:, np.newaxis, :]  # (h*w, n, 2)
    q_hat = target_pts[np.newaxis, :, :] - q_star[:, np.newaxis, :]  # (h*w, n, 2)

    # 计算矩阵 mu (h*w, 2, 2)
    w_i_expand = w_i[:, :, np.newaxis, np.newaxis]  # (h*w, n, 1, 1)
    p_hat_expand = p_hat[:, :, :, np.newaxis]  # (h*w, n, 2, 1)
    p_hat_T = p_hat[:, :, np.newaxis, :]  # (h*w, n, 1, 2)
    mu = np.sum(w_i_expand * np.matmul(p_hat_expand, p_hat_T), axis=1)  # (h*w, 2, 2)

    # 计算矩阵 B (h*w, 2, 2)
    q_hat_expand = q_hat[:, :, :, np.newaxis]  # (h*w, n, 2, 1)
    q_hat_T = q_hat[:, :, np.newaxis, :]  # (h*w, n, 1, 2)
    B = np.sum(w_i_expand * np.matmul(p_hat_expand, q_hat_T), axis=1)  # (h*w, 2, 2)

    # 计算 mu 的逆矩阵
    det_mu = np.linalg.det(mu)  # (h*w,)
    det_mu[det_mu == 0] = eps  # 避免除零
    mu_inv = np.linalg.inv(mu)  # (h*w, 2, 2)

    # 计算 A_v (h*w, 1, 2)
    v_minus_p_star = grid[:, np.newaxis, :] - p_star[:, np.newaxis, :]  # (h*w, 1, 2)
    A_v = np.matmul(v_minus_p_star, mu_inv)  # (h*w, 1, 2)

    # 计算映射后的位移 (h*w, 2)
    mapped_disp = np.squeeze(np.matmul(A_v, B), axis=1)  # (h*w, 2)

    # 计算映射后的坐标 (h*w, 2)
    mapped_v = mapped_disp + q_star  # (h*w, 2)

    # 对映射后的坐标进行双线性插值
    x_src = mapped_v[:, 0]
    y_src = mapped_v[:, 1]

    # 判断坐标是否在源图像范围内
    valid = (x_src >= 0) & (x_src < w - 1) & (y_src >= 0) & (y_src < h - 1)

    x0 = np.floor(x_src).astype(np.int32)
    y0 = np.floor(y_src).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    dx = x_src - x0
    dy = y_src - y0

   # 获取源图像像素值
    def get_pixel_values(x, y):
        indices = y * w + x
        valid_indices = (indices >= 0) & (indices < image.size // 3)  # 检查索引是否有效
        valid_indices = np.where(valid_indices)[0]  # 获取有效索引的位置
        pixel_values = np.full((len(indices), 3), [255, 255, 255], dtype=np.float64)  # 默认白色
        pixel_values[valid_indices] = image.reshape(-1, 3)[indices[valid_indices]]  # 仅获取有效索引的像素值
        return pixel_values

    I00 = get_pixel_values(x0, y0)
    I10 = get_pixel_values(x1, y0)
    I01 = get_pixel_values(x0, y1)
    I11 = get_pixel_values(x1, y1)

    # 进行双线性插值
    warped_pixel = (
        (1 - dx)[:, np.newaxis] * (1 - dy)[:, np.newaxis] * I00
        + dx[:, np.newaxis] * (1 - dy)[:, np.newaxis] * I10
        + (1 - dx)[:, np.newaxis] * dy[:, np.newaxis] * I01
        + dx[:, np.newaxis] * dy[:, np.newaxis] * I11
    )

    # 构建变形后的图像
    warped_image = np.full((h * w, 3), [255, 255, 255], dtype=np.float64)
    warped_image[valid] = warped_pixel[valid]
    warped_image = warped_image.reshape((h, w, 3)).astype(np.uint8)

    # 将变换后的坐标输出到文件
    transformed_points = mapped_v.reshape((h, w, 2))
    np.savetxt(
        os.path.join(os.path.dirname(__file__), "transformed_points.csv"),
        transformed_points.reshape(-1, 2),
        delimiter=",",
    )

    return warped_image


def run_warping():
    global points_src, points_dst, image  # 获取全局变量

    start_time = time.time()  # 记录开始时间
    warped_image = point_guided_deformation(
        image,
        np.array(points_src, dtype=np.float64),
        np.array(points_dst, dtype=np.float64),
    )
    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算耗时

    return warped_image, f"变形耗时: {elapsed_time:.2f} 秒"


# 清除选中点
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image  # 返回未标记的原图


# 使用 Gradio 构建界面
with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align: center'>Image Deformation Playground</h1>")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                label="上传图片", interactive=True, width=800, height=200
            )
            point_select = gr.Image(
                label="点击选择控制点和目标点", interactive=True, width=800, height=600
            )

        with gr.Column():
            result_image = gr.Image(label="变换结果", width=800, height=600)
            time_text = gr.Text(label="耗时")  # 添加显示耗时的文本框

            # 按钮
            run_button = gr.Button("Run Warping")
            clear_button = gr.Button("Clear Points")  # 添加清除按钮

    # 上传图像的交互
    input_image.upload(upload_image, input_image, point_select)
    # 选择点的交互, 点选后刷新图像
    point_select.select(record_points, None, point_select)
    # 点击运行 warping 按钮, 计算并显示变换后的图像
    run_button.click(run_warping, None, [result_image, time_text])
    # 点击清除按钮, 清空所有已选择的点
    clear_button.click(clear_points, None, point_select)

    # Launch the Gradio interface
    demo.launch(show_error=True)
