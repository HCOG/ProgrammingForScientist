"""
Lesson 02: Monte Carlo Integration — 用随机性计算确定性
========================================================
目标：理解如何用随机采样估算定积分
知识点：Monte Carlo方法、概率论、大数定律、数值积分

学习目标（反向阅读提示）：
- 先运行，观察输出的积分估计值
- 倒推：为什么增加采样点数会提高精度？
- 倒推：随机撒点和规则网格相比，优势在哪？
"""

import numpy as np
import math


def monte_carlo_integrate_2d(func, x_min, x_max, y_min, y_max, n_samples=100000):
    """
    二维Monte Carlo积分
    
    核心思想：在积分区域内随机撒点，统计落入函数曲线下方的比例
    
    数学推导：
        I = ∫∫ f(x,y) dA
          ≈ (A / N) * Σ f(x_i, y_i)  [对随机点求平均]
    
    其中 A = (x_max - x_min) * (y_max - y_min) 是区域面积
    """
    # 随机生成采样点
    x = np.random.uniform(x_min, x_max, n_samples)
    y = np.random.uniform(y_min, y_max, n_samples)
    
    # 计算函数值
    f_values = func(x, y)
    
    # 统计正值和负值（处理可能有负值的函数）
    area = (x_max - x_min) * (y_max - y_min)
    
    # 方法：统计 f(x,y) > 0 的概率
    # E[f] = (1/N) * Σ f(x_i, y_i)
    # I = area * E[f]
    integral_estimate = area * np.mean(f_values)
    
    # 计算标准误差
    std_error = area * np.std(f_values) / math.sqrt(n_samples)
    
    return integral_estimate, std_error


def example_function_2d(x, y):
    """
    示例函数：f(x,y) = sin(x) * cos(y) + 1
    定义域：x ∈ [0, π], y ∈ [0, π]
    """
    return np.sin(x) * np.cos(y) + 1


def example_circle(x, y):
    """
    示例函数：单位圆面积
    f(x,y) = 1 if x² + y² <= 1, else 0
    积分结果应为 π
    """
    return (x**2 + y**2) <= 1


if __name__ == "__main__":
    print("=" * 55)
    print("Monte Carlo 积分方法演示")
    print("=" * 55)
    
    # ===== 示例1：sin(x)cos(y) 在 [0,π]×[0,π] 上的积分 =====
    print("\n【示例1】∫∫ [sin(x)·cos(y) + 1] dxdy, x∈[0,π], y∈[0,π]")
    print("解析解: π² (因为∫sin(x)dx = 2, ∫cos(y)dy = 0, 实际为π·π=π²)")
    
    for n in [1000, 10000, 100000, 1000000]:
        result, error = monte_carlo_integrate_2d(
            example_function_2d, 0, math.pi, 0, math.pi, n_samples=n
        )
        print(f"  N = {n:>8}: 估计值 = {result:.6f}, 误差 = ±{error:.6f}")
    
    # ===== 示例2：单位圆面积 =====
    print("\n【示例2】单位圆面积（验证方法正确性）")
    print("解析解: π ≈ 3.1415926535...")
    
    for n in [1000, 10000, 100000, 1000000]:
        # 圆的外接正方形: x ∈ [-1,1], y ∈ [-1,1]
        result, error = monte_carlo_integrate_2d(
            example_circle, -1, 1, -1, 1, n_samples=n
        )
        print(f"  N = {n:>8}: 估计值 = {result:.6f}, 误差 = ±{error:.6f}")
    
    # ===== 收敛性可视化数据 =====
    print("\n【收敛性测试】单位圆面积，随机种子固定")
    np.random.seed(12345)
    print("  N        估计值         相对误差")
    print("  " + "-" * 35)
    cumulative_mean = 0
    for k, n in enumerate([10, 50, 100, 500, 1000, 5000, 10000], 1):
        result, _ = monte_carlo_integrate_2d(example_circle, -1, 1, -1, 1, n_samples=n)
        relative_error = abs(result - math.pi) / math.pi * 100
        print(f"  {n:>5}:  {result:.6f}    {relative_error:.3f}%")
