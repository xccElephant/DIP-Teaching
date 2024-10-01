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

    # 在图像上标记点(蓝色：控制点, 红色：目标点), 并画箭头
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
    warped_image = np.zeros_like(image, dtype=np.float64)  # 初始化变形后的图像

    # 交换 source_pts 和 target_pts, 实现向后映射
    source_pts, target_pts = target_pts.copy(), source_pts.copy()

    transformed_points = []  # 用于存储变换后的坐标

    for y in range(h):
        for x in range(w):
            v = np.array([x, y], dtype=np.float64)  # 当前像素位置

            # 计算权重 w_i
            dist2 = np.sum((source_pts - v) ** 2, axis=1)  # 源点到当前像素的平方距离
            w_i = 1.0 / (dist2**alpha + eps)  # 权重

            # 计算加权中心 p_star 和 q_star
            w_sum = np.sum(w_i)
            p_star = np.sum(w_i[:, np.newaxis] * source_pts, axis=0) / w_sum
            q_star = np.sum(w_i[:, np.newaxis] * target_pts, axis=0) / w_sum

            # 计算局部坐标系下的坐标差值
            p_hat = source_pts - p_star  # 形状为 (n, 2)
            q_hat = target_pts - q_star  # 形状为 (n, 2)

            # 计算矩阵 μ
            mu = np.zeros((2, 2), dtype=np.float64)
            for i in range(source_pts.shape[0]):
                p_hi = p_hat[i].reshape(2, 1)
                mu += w_i[i] * p_hi @ p_hi.T

            # 判断 μ 是否可逆
            if np.abs(np.linalg.det(mu)) < eps:
                mapped_v = v  # 若不可逆, 映射为原始坐标
            else:
                mu_inv = np.linalg.inv(mu)  # 计算 μ 的逆矩阵
                A_v = (v - p_star) @ mu_inv  # 计算中间变量 A_v

                # 计算矩阵 B
                B = np.einsum("i,ij,ik->jk", w_i, p_hat, q_hat)  # 形状为 (2, 2)

                # 计算映射后的位移
                mapped_disp = A_v @ B

                # 计算映射后的坐标
                mapped_v = mapped_disp + q_star

            transformed_points.append(mapped_v)  # 存储变换后的坐标
            x_src, y_src = mapped_v

            # 对映射后的坐标进行双线性插值
            if 0 <= x_src < w - 1 and 0 <= y_src < h - 1:
                x0 = int(np.floor(x_src))
                y0 = int(np.floor(y_src))
                x1 = x0 + 1
                y1 = y0 + 1

                dx = x_src - x0
                dy = y_src - y0

                # 获取四个邻近像素值
                I00 = image[y0, x0]  # 左上
                I10 = image[y0, x1]  # 右上
                I01 = image[y1, x0]  # 左下
                I11 = image[y1, x1]  # 右下

                # 进行双线性插值
                warped_pixel = (
                    (1 - dx) * (1 - dy) * I00
                    + dx * (1 - dy) * I10
                    + (1 - dx) * dy * I01
                    + dx * dy * I11
                )

                warped_image[y, x] = warped_pixel
            else:
                # 映射坐标超出范围, 填充黑色
                warped_image[y, x] = [255, 255, 255]

    # 将变换后的坐标输出到文件
    np.savetxt(
        os.path.join(os.path.dirname(__file__), "transformed_points.csv"),
        np.array(transformed_points).reshape(-1, 2),
        delimiter=",",
    )

    return warped_image.astype(np.uint8)


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
