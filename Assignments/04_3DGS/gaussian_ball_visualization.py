import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal

def plot_gaussian_1d(mu=0, sigma=1):
    """绘制一维高斯分布"""
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
    y = 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-(x-mu)**2/(2*sigma**2))
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.title('1D Gaussian Ball')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.grid(True)
    plt.show()

def plot_gaussian_2d(mu=[0, 0], sigma=[[1, 0], [0, 1]], points=100):
    """绘制二维高斯分布"""
    x = np.linspace(mu[0] - 3*np.sqrt(sigma[0][0]), mu[0] + 3*np.sqrt(sigma[0][0]), points)
    y = np.linspace(mu[1] - 3*np.sqrt(sigma[1][1]), mu[1] + 3*np.sqrt(sigma[1][1]), points)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    
    rv = multivariate_normal(mu, sigma)
    Z = rv.pdf(pos)

    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, Z, levels=20, cmap='viridis')
    plt.colorbar(label='Density')
    plt.title('2D Gaussian Ball')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def plot_gaussian_3d(mu=[0, 0, 0], sigma=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], points=30):
    """绘制三维高斯分布"""
    x = np.linspace(mu[0] - 3*np.sqrt(sigma[0][0]), mu[0] + 3*np.sqrt(sigma[0][0]), points)
    y = np.linspace(mu[1] - 3*np.sqrt(sigma[1][1]), mu[1] + 3*np.sqrt(sigma[1][1]), points)
    z = np.linspace(mu[2] - 3*np.sqrt(sigma[2][2]), mu[2] + 3*np.sqrt(sigma[2][2]), points)
    
    X, Y, Z = np.meshgrid(x, y, z)
    pos = np.stack([X, Y, Z], axis=-1)
    
    rv = multivariate_normal(mu, sigma)
    density = rv.pdf(pos)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 选择几个等值面进行绘制
    levels = np.linspace(density.min(), density.max(), 10)
    for level in levels[3:6]:  # 只绘制中间几个等值面以避免过于拥挤
        ax.contour3D(X[:,:,0], Y[:,:,0], density[:,:,points//2], 
                     levels=[level], cmap='viridis')
    
    ax.set_title('3D Gaussian Ball')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

# 测试代码
if __name__ == "__main__":
    # 1D示例
    plot_gaussian_1d(mu=0, sigma=1)
    
    # 2D示例
    mu_2d = [0, 0]
    sigma_2d = [[1, 0.5], [0.5, 2]]
    plot_gaussian_2d(mu_2d, sigma_2d)
    
    # 3D示例
    mu_3d = [0, 0, 0]
    sigma_3d = [[1, 0.2, 0.2], [0.2, 2, 0.3], [0.2, 0.3, 1.5]]
    plot_gaussian_3d(mu_3d, sigma_3d)
