# 基于MLS的图像变形算法

## 仿射变换

给定控制点对 $(p_i, q_i)$ (行向量), 对于变换点 $v$, 找到 $l_v(x)$, 使得下式最小化:

$$
\sum_{i} w_i |l_v(p_i) - q_i|^2
$$

其中, $w_i = \frac{1}{|p_i-v|^{2\alpha}}$

$$
l_v(x) = xM + T
$$

通过令梯度为 0, 容易得到最优的 $l_v(x)$ 的表达式为:

$$
l_v(x) = (x - p^*)(\sum_i \hat{p_i}^T w_i \hat{p_i})^{-1} \sum_j w_j\hat{p_j}^T\hat{q_j} + q^*
$$

其中, $p^* = \frac{\sum_i w_i p_i}{\sum_i w_i}, q^* = \frac{\sum_j w_j q_j}{\sum_j w_j}, \hat{p_i} = p_i - p^*, \hat{q_j} = q_j - q^*$.

所以, 
$$
l_v(v) = \sum_j A_j \hat{q_j} + q^*.$$
其中, 
$$
A_j = (v-p^*)(\sum_i \hat{p_i}^T w_i \hat{p_i})^{-1}w_j\hat{p_j}^T.
$$

当 $p_j$ 固定, $q_j$ 在变化时, $A_j$ 是固定的, 可以预先计算好, 加快变形时间.

对于图像的每个点 $v$, 都执行以上的变换, 得到新图像
