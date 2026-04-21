"""
Lesson 04: Gradient Descent — 找到山谷的最快路径
===================================================
目标：理解梯度下降法的原理和变种
知识点：优化算法、梯度、牛顿法、学习率、收敛性

核心思想：
    山坡陡峭的方向就是函数下降最快的方向
    沿着梯度的负方向更新参数

学习目标（反向阅读提示）：
- 倒推：为什么梯度指向函数上升最快的方向？
- 倒推：学习率太大或太小会导致什么问题？
- 比较不同的优化器：SGD, Momentum, Adam
"""

import numpy as np
import matplotlib.pyplot as plt


def rosenbrock(x, y):
    """
    Rosenbrock 函数（经典的测试优化算法的问题）
    全局最小值在 (1, 1)，函数值为 0
    峡谷形状使得普通梯度下降很难收敛
    f(x,y) = (1-x)² + 100(y-x²)²
    """
    return (1 - x)**2 + 100 * (y - x**2)**2


def rosenbrock_gradient(x, y):
    """
    Rosenbrock 函数的梯度（解析计算）
    ∂f/∂x = -2(1-x) - 400x(y-x²)
    ∂f/∂y = 200(y-x²)
    """
    df_dx = -2 * (1 - x) - 400 * x * (y - x**2)
    df_dy = 200 * (y - x**2)
    return np.array([df_dx, df_dy])


def gradient_descent(grad_func, initial_point, learning_rate, n_iterations, tolerance=1e-8):
    """
    标准梯度下降法
    
    更新规则：θ_{t+1} = θ_t - α * ∇f(θ_t)
    
    参数:
        grad_func: 梯度函数
        initial_point: 初始点
        learning_rate: 学习率 α
        n_iterations: 最大迭代次数
        tolerance: 提前停止的梯度范数阈值
    """
    point = np.array(initial_point, dtype=float)
    history = [point.copy()]
    
    for i in range(n_iterations):
        gradient = grad_func(point[0], point[1])
        point = point - learning_rate * gradient
        history.append(point.copy())
        
        # 提前停止条件：梯度足够小
        if np.linalg.norm(gradient) < tolerance:
            print(f"提前收敛于第 {i} 次迭代")
            break
    
    return np.array(history)


def momentum_gradient_descent(grad_func, initial_point, learning_rate, momentum=0.9, n_iterations=10000):
    """
    动量梯度下降法
    
    核心思想：积累历史的梯度方向，像滚下山坡的惯性
    更新规则：
        v_{t+1} = β * v_t + (1-β) * ∇f(θ_t)    # 速度更新
        θ_{t+1} = θ_t - α * v_{t+1}              # 位置更新
    
    参数:
        momentum: 动量系数 β，控制历史梯度的影响
    """
    point = np.array(initial_point, dtype=float)
    velocity = np.zeros(2)
    history = [point.copy()]
    
    for i in range(n_iterations):
        gradient = grad_func(point[0], point[1])
        velocity = momentum * velocity + (1 - momentum) * gradient
        point = point - learning_rate * velocity
        history.append(point.copy())
        
        if np.linalg.norm(gradient) < 1e-8:
            print(f"提前收敛于第 {i} 次迭代")
            break
    
    return np.array(history)


def adam_optimizer(grad_func, initial_point, learning_rate=0.001, 
                    beta1=0.9, beta2=0.999, epsilon=1e-8, n_iterations=10000):
    """
    Adam (Adaptive Moment Estimation) 优化器
    
    结合了：
    1. 动量法（一阶矩估计）
    2. RMSProp（二阶矩估计，自适应学习率）
    
    更新规则：
        m_t = β₁m_{t-1} + (1-β₁)g_t          # 一阶矩（均值）
        v_t = β₂v_{t-1} + (1-β₂)g_t²         # 二阶矩（方差）
        m̂_t = m_t / (1-β₁^t)                 # 偏差校正
        v̂_t = v_t / (1-β₂^t)                 # 偏差校正
        θ_{t+1} = θ_t - α * m̂_t / (√v̂_t + ε)
    """
    point = np.array(initial_point, dtype=float)
    m = np.zeros(2)  # 一阶矩估计
    v = np.zeros(2)  # 二阶矩估计
    history = [point.copy()]
    
    for t in range(1, n_iterations + 1):
        gradient = grad_func(point[0], point[1])
        
        # 一阶矩更新
        m = beta1 * m + (1 - beta1) * gradient
        # 二阶矩更新
        v = beta2 * v + (1 - beta2) * gradient**2
        
        # 偏差校正
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        
        # 参数更新
        point = point - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        history.append(point.copy())
        
        if np.linalg.norm(gradient) < 1e-8:
            print(f"提前收敛于第 {t} 次迭代")
            break
    
    return np.array(history)


def golden_section_search(f, a, b, tolerance=1e-6, max_iterations=100):
    """
    黄金分割搜索法：一维单峰函数的寻优方法
    
    特点：不需要导数信息，收敛稳定
    缺点：只适用于一维问题
    
    核心：利用黄金比例 0.618 来选择试探点
    """
    phi = (1 + np.sqrt(5)) / 2  # 黄金比例 ≈ 1.618
    
    c = b - (b - a) / phi
    d = a + (b - a) / phi
    
    for _ in range(max_iterations):
        if abs(b - a) < tolerance:
            break
        
        if f(c) < f(d):
            b = d
        else:
            a = c
        
        c = b - (b - a) / phi
        d = a + (b - a) / phi
    
    return (a + b) / 2, f((a + b) / 2)


if __name__ == "__main__":
    print("=" * 55)
    print("优化算法：梯度下降法及其变种")
    print("=" * 55)
    
    # ===== 测试函数 =====
    print("\n目标函数: Rosenbrock function f(x,y) = (1-x)² + 100(y-x²)²")
    print("全局最小值: (1, 1), 函数值 = 0")
    print("初始点: (-1.5, 0.5)")
    
    initial = [-1.5, 0.5]
    
    # ===== 标准梯度下降 =====
    print("\n【1】标准梯度下降 (learning_rate=0.001)")
    history_gd = gradient_descent(rosenbrock_gradient, initial, learning_rate=0.001, n_iterations=10000)
    final_gd = history_gd[-1]
    print(f"  最终点: ({final_gd[0]:.6f}, {final_gd[1]:.6f})")
    print(f"  函数值: {rosenbrock(final_gd[0], final_gd[1]):.6f}")
    print(f"  迭代次数: {len(history_gd)}")
    
    # ===== 动量梯度下降 =====
    print("\n【2】动量梯度下降 (learning_rate=0.001, momentum=0.9)")
    history_momentum = momentum_gradient_descent(rosenbrock_gradient, initial, 
                                                  learning_rate=0.001, momentum=0.9, n_iterations=10000)
    final_momentum = history_momentum[-1]
    print(f"  最终点: ({final_momentum[0]:.6f}, {final_momentum[1]:.6f})")
    print(f"  函数值: {rosenbrock(final_momentum[0], final_momentum[1]):.6f}")
    print(f"  迭代次数: {len(history_momentum)}")
    
    # ===== Adam 优化器 =====
    print("\n【3】Adam 优化器 (learning_rate=0.01)")
    history_adam = adam_optimizer(rosenbrock_gradient, initial, 
                                  learning_rate=0.01, n_iterations=10000)
    final_adam = history_adam[-1]
    print(f"  最终点: ({final_adam[0]:.6f}, {final_adam[1]:.6f})")
    print(f"  函数值: {rosenbrock(final_adam[0], final_adam[1]):.6f}")
    print(f"  迭代次数: {len(history_adam)}")
    
    # ===== 收敛路径对比 =====
    print("\n【收敛过程对比】")
    print(f"{'算法':<20} {'收敛稳定性':<15} {'最终精度':<12} {'备注'}")
    print("-" * 65)
    print(f"{'标准梯度下降':<18} {'震荡大':<15} {rosenbrock(final_gd[0], final_gd[1]):<12.2e} 容易陷入局部")
    print(f"{'动量法':<18} {'改善震荡':<15} {rosenbrock(final_momentum[0], final_momentum[1]):<12.2e} 加速收敛")
    print(f"{'Adam':<18} {'自适应学习率':<15} {rosenbrock(final_adam[0], final_adam[1]):<12.2e} 通常最优")
    
    # ===== 黄金分割搜索（一维示例） =====
    print("\n" + "=" * 55)
    print("【附加】黄金分割搜索法演示：求 f(x) = x² - 2x + 1 的最小值")
    print("=" * 55)
    
    def simple_func(x):
        return x**2 - 2*x + 1
    
    optimum, min_val = golden_section_search(simple_func, -5, 5)
    print(f"  搜索区间: [-5, 5]")
    print(f"  找到的最优点: x = {optimum:.6f}")
    print(f"  最小函数值: {min_val:.6f}")
    print(f"  解析解: x = 1, f(1) = 0")
