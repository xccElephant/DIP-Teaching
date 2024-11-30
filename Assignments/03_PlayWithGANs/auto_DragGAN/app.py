"""
自动使用 DragGAN 库进行人脸变形:
预设几种常见的变形, 如 "微笑"、"皱眉"、"张嘴"、"闭眼"、"闭嘴"等, 自动使用 'face_alignment' 库检测面部特征点, 构建好特征点的变形(输入和输出的 pair), 然后调用 DragGAN 进行变形.
Gradio 库用于构建一个简单的 Web 界面, 用于选择变形类型和输入图像, 并显示变形后的图像.
"""

import os
import sys
import face_alignment
import gradio as gr
import numpy as np
from PIL import Image
import torch

# 添加 DragGAN 目录到 Python 路径
draggan_path = os.path.join(os.path.dirname(__file__), 'DragGAN')
if draggan_path not in sys.path:
    sys.path.append(draggan_path)

# 导入 DragGAN 所需的模块
import dnnlib
sys.path.append(os.path.join(draggan_path, 'viz'))
from renderer import Renderer

# 预定义变形类型及其对应的特征点变化
TRANSFORMS = {
    "微笑": "smile",
    "皱眉": "frown",
    "张嘴": "open_mouth",
    "闭眼": "close_eyes",
    "闭嘴": "close_mouth",
}

class AutoDragGAN:
    def __init__(self):
        # 指定模型路径
        self.model_path = os.path.join(draggan_path, 'checkpoints', 'stylegan2-ffhq-512x512.pkl')
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件 {self.model_path} 不存在!")
        print(f"成功加载模型：{self.model_path}")

        # 初始化 face_alignment
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)

        # 初始化 DragGAN 的渲染器
        self.renderer = Renderer()

    def detect_landmarks(self, image):
        """检测图像中的人脸特征点"""
        if isinstance(image, str):
            image = Image.open(image)
        image_np = np.array(image)
        landmarks = self.fa.get_landmarks(image_np)
        if landmarks is None or len(landmarks) == 0:
            raise ValueError("未检测到人脸!")
        return landmarks[0]

    def get_transform_points(self, landmarks, transform_type):
        """根据变形类型，获取需要变形的特征点和目标位置"""
        points = []
        targets = []

        if transform_type == "微笑":
            # 提升嘴角
            left_mouth_corner = landmarks[48]  # 左嘴角 (x, y)
            right_mouth_corner = landmarks[54]  # 右嘴角 (x, y)
            offset = -5  # 向上移动
            points.append([left_mouth_corner[1], left_mouth_corner[0]])  # [y, x]
            targets.append([left_mouth_corner[1] + offset, left_mouth_corner[0]])
            points.append([right_mouth_corner[1], right_mouth_corner[0]])  # [y, x]
            targets.append([right_mouth_corner[1] + offset, right_mouth_corner[0]])

        elif transform_type == "皱眉":
            # 拉近眉毛之间的距离
            left_brow = landmarks[21]  # 左眉尾
            right_brow = landmarks[22]  # 右眉头
            offset = -5  # 向内移动
            points.append([left_brow[1], left_brow[0]])  # [y, x]
            targets.append([left_brow[1], left_brow[0] + offset])
            points.append([right_brow[1], right_brow[0]])  # [y, x]
            targets.append([right_brow[1], right_brow[0] - offset])

        elif transform_type == "张嘴":
            # 下拉下巴，打开嘴巴
            chin = landmarks[8]  # 下巴
            offset = 10  # 向下移动
            points.append([chin[1], chin[0]])  # [y, x]
            targets.append([chin[1], chin[0] + offset])

        elif transform_type == "闭眼":
            # 上移下眼睑，下移上眼睑
            left_upper_eyelid = landmarks[37]
            left_lower_eyelid = landmarks[41]
            right_upper_eyelid = landmarks[44]
            right_lower_eyelid = landmarks[46]
            offset = -3
            points.extend([
                [left_upper_eyelid[1], left_upper_eyelid[0]],
                [left_lower_eyelid[1], left_lower_eyelid[0]],
                [right_upper_eyelid[1], right_upper_eyelid[0]],
                [right_lower_eyelid[1], right_lower_eyelid[0]]
            ])
            targets.extend([
                [left_upper_eyelid[1], left_upper_eyelid[0] + offset],
                [left_lower_eyelid[1], left_lower_eyelid[0] - offset],
                [right_upper_eyelid[1], right_upper_eyelid[0] + offset],
                [right_lower_eyelid[1], right_lower_eyelid[0] - offset]
            ])

        elif transform_type == "闭嘴":
            # 上移下唇，下移上唇
            upper_lip = landmarks[51]
            lower_lip = landmarks[57]
            offset = -3
            points.extend([
                [upper_lip[1], upper_lip[0]],
                [lower_lip[1], lower_lip[0]]
            ])
            targets.extend([
                [upper_lip[1], upper_lip[0] + offset],
                [lower_lip[1], lower_lip[0] - offset]
            ])

        else:
            raise ValueError(f"未知的变形类型: {transform_type}")

        # 转换为 numpy 数组并转换为整数
        points = np.round(points).astype(int)
        targets = np.round(targets).astype(int)

        return points, targets

    def draggan_deform(self, points, targets):
        """使用 DragGAN 进行图像变形"""
        # 设置渲染器的参数
        render_args = {
            'pkl': self.model_path,
            'w0_seed': 0,  # 可以根据需要调整
            'noise_mode': 'const',
            'trunc_psi': 0.7,
            'is_drag': True,
            'points': points,
            'targets': targets,
            'mask': None,
            'lambda_mask': 0,
            'feature_idx': 5,
            'reset': True,
            'reset_w': True,
        }
        # 执行渲染
        res = self.renderer.render(**render_args)
        if 'image' not in res:
            if 'error' in res:
                print(f"渲染错误：{res.error}")
            raise RuntimeError("渲染失败！")
        image = res.image
        return Image.fromarray(image)

    def apply_transform(self, image, transform_type):
        """应用预定义的变形"""
        print(f"应用变形: {transform_type}")
        # 将图像调整为 512x512
        image = image.resize((512, 512))
        # 检测特征点
        landmarks = self.detect_landmarks(image)
        # 获取变形的点和目标点
        points, targets = self.get_transform_points(landmarks, transform_type)
        # 调用 DragGAN 进行变形
        result_image = self.draggan_deform(points, targets)
        return result_image

def create_interface():
    draggan = AutoDragGAN()

    with gr.Blocks() as app:
        gr.Markdown("# 自动人脸变形 DragGAN")

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="输入图像", type="pil")
                transform_type = gr.Dropdown(
                    choices=list(TRANSFORMS.keys()),
                    label="变形类型",
                    value="微笑"
                )
                transform_btn = gr.Button("应用变形")

            with gr.Column():
                output_image = gr.Image(label="变形结果")

        def process(image, transform):
            return draggan.apply_transform(image, transform)

        transform_btn.click(
            fn=process,
            inputs=[input_image, transform_type],
            outputs=output_image,
        )

    return app

if __name__ == "__main__":
    app = create_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        debug=True,
        show_error=True,
    )
