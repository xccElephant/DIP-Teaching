import unittest
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from app import AutoDragGAN

class TestAutoDragGAN(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """测试开始前的设置"""
        cls.draggan = AutoDragGAN()
        # 测试图片路径 - 请确保这个路径存在一张人脸图片
        cls.test_image_path = "girl.jpg"  
        
        if not os.path.exists(cls.test_image_path):
            raise FileNotFoundError(f"请在 {cls.test_image_path} 放置一张测试用的人脸图片")
            
    def test_face_landmarks(self):
        """测试人脸关键点检测"""
        print("\n测试1: 人脸关键点检测")
        try:
            # 加载测试图片
            image = Image.open(self.test_image_path)
            
            # 检测关键点
            landmarks = self.draggan.detect_landmarks(image)
            
            # 验证关键点数量 (face-alignment 应该返回68个关键点)
            self.assertEqual(len(landmarks), 68, "关键点数量应该为68个")
            
            # 可视化关键点
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            plt.scatter(landmarks[:, 0], landmarks[:, 1], c='r', s=5)
            plt.savefig('landmarks_visualization.png')
            plt.close()
            
            print("✓ 人脸关键点检测成功")
            print(f"- 检测到 {len(landmarks)} 个关键点")
            print("- 关键点可视化已保存到 landmarks_visualization.png")
            
        except Exception as e:
            self.fail(f"人脸关键点检测失败: {str(e)}")
    
    def test_draggan_transform(self):
        """测试DragGAN变形功能"""
        print("\n测试2: DragGAN变形")
        try:
            # 加载测试图片
            image = Image.open(self.test_image_path)
            
            # 测试"闭眼"变形
            transform_type = "闭眼"
            print(f"正在测试 {transform_type} 变形...")
            
            # 执行变形
            result_image = self.draggan.apply_transform(image, transform_type)
            
            # 验证返回的图片是否有效
            self.assertIsNotNone(result_image, "变形结果不应为空")
            
            # 保存变形结果
            if isinstance(result_image, np.ndarray):
                result_image = Image.fromarray(result_image)
            result_image.save(f'transform_result_{transform_type}.png')
            
            print("✓ DragGAN变形成功")
            print(f"- 变形结果已保存到 transform_result_{transform_type}.png")
            
        except Exception as e:
            self.fail(f"DragGAN变形失败: {str(e)}")

if __name__ == '__main__':
    unittest.main(verbosity=2)
