from setuptools import setup, find_packages

setup(
    name="SMITE",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "numpy>=1.19.2",
        "opencv-python>=4.5.3",
        "scipy>=1.7.1",
        "pillow>=8.3.1",
        "tqdm>=4.62.2",
        "transformers>=4.11.0",
        "diffusers>=0.11.0",
        "accelerate>=0.12.0"
    ],
    author="Your Name",
    description="SMITE: Segment Me In Time - Video Segmentation with Diffusion Models",
    python_requires=">=3.8",
) 