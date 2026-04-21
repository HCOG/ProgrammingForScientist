"""
Lesson 03: Heat Equation — 热传导的数学之美
============================================
目标：用有限差分法求解一维热传导方程
知识点：偏微分方程数值解、有限差分、稳定性条件、边界条件

热传导方程（偏微分方程）:
    ∂u/∂t = α * ∂²u/∂x²

学习目标（反向阅读提示）：
- 先理解边界条件：杆子两端温度是如何固定的？
- 倒推：为什么时间步长和空间步长要满足 CFL 条件？
- 倒推：显式格式和隐式格式各自的优势和劣势？
"""

import numpy as np
import matplotlib.pyplot as plt


def solve_heat_explicit(u0, alpha, dx, dt, n_steps):
    """
    显式有限差分法求解一维热传导方程
    
    离散化：
        ∂u/∂t ≈ (u_i^(n+1) - u_i^n) / dt
        ∂²u/∂x² ≈ (u_{i+1}^n - 2*u_i^n + u_{i-1}^n) / dx²
    
    代入方程得到显式迭代格式：
        u_i^(n+1) = u_i^n + r * (u_{i+1}^n - 2*u_i^n + u_{i-1}^n)
    
    其中 r = α * dt / dx²
    
    ⚠️ 稳定性条件：r <= 0.5（否则数值解会发散）
    """
    u = u0.copy()
    n_points = len(u)
    r = alpha * dt / dx**2
    
    # 稳定性检查
    if r > 0.5:
        print(f"警告：r = {r:.3f} > 0.5，显式格式不稳定！")
        print("建议：减小 dt 或增大 dx")
    
    history = [u.copy()]
    
    for step in range(n_steps):
        u_new = u.copy()
        for i in range(1, n_points - 1):
            u_new[i] = u[i] + r * (u[i+1] - 2*u[i] + u[i-1])
        u = u_new
        history.append(u.copy())
    
    return np.array(history)


def solve_heat_implicit(u0, alpha, dx, dt, n_steps):
    """
    隐式有限差分法（向后欧拉）求解一维热传导方程
    
    离散化（时间导数用t_{n+1}时刻的值近似）：
        ∂u/∂t ≈ (u_i^(n+1) - u_i^n) / dt
        ∂²u/∂x² ≈ (u_{i+1}^{n+1} - 2*u_i^{n+1} + u_{i-1}^{n+1}) / dx²
    
    得到三对角方程组：(-r)u_{i-1}^{n+1} + (1+2r)u_i^{n+1} + (-r)u_{i+1}^{n+1} = u_i^n
    
    ⚠️ 优势：隐式格式无条件稳定，可以取更大的时间步长
    ⚠️ 代价：每一步需要求解线性方程组
    """
    from scipy.linalg import solve_banded
    
    u = u0.copy()
    n_points = len(u)
    r = alpha * dt / dx**2
    
    history = [u.copy()]
    
    for step in range(n_steps):
        # 构建三对角矩阵 (banded format)
        # ab[0,:] = upper diagonal, ab[1,:] = main diagonal, ab[2,:] = lower diagonal
        ab = np.zeros((3, n_points))
        ab[1, :] = 1 + 2 * r  # 主对角
        ab[0, 1:] = -r         # 上对角
        ab[2, :-1] = -r        # 下对角
        
        # 边界条件：Dirichlet边界，固定端点温度
        ab[1, 0] = 1
        ab[1, -1] = 1
        
        # 右侧向量
        b = u.copy()
        
        # 求解三对角方程组
        u = solve_banded((1, 1), ab, b)
        history.append(u.copy())
    
    return np.array(history)


def initial_temperature_distribution(x, L):
    """
    初始温度分布：高斯热源位于杆中央
    u(x,0) = exp(-100 * (x - L/2)²)
    """
    return np.exp(-100 * (x - L / 2)**2)


if __name__ == "__main__":
    # ===== 参数设置 =====
    L = 1.0           # 杆长 [m]
    N = 100           # 空间点数
    dx = L / (N - 1)
    x = np.linspace(0, L, N)
    
    alpha = 0.01      # 热扩散系数 [m²/s]
    
    # 时间步长（显式格式需要满足 CFL 条件）
    dt_explicit = 0.0001    # 满足 r < 0.5
    dt_implicit = 0.01      # 可以取更大的时间步长
    
    # 初始条件
    u0 = initial_temperature_distribution(x, L)
    
    # 边界条件：两端温度固定为0
    u0[0] = 0
    u0[-1] = 0
    
    # ===== 模拟设置 =====
    n_steps = 500  # 总时间步数
    
    print("=" * 55)
    print("一维热传导方程数值求解")
    print("=" * 55)
    print(f"杆长 L = {L} m, 空间点数 N = {N}")
    print(f"热扩散系数 α = {alpha} m²/s")
    print(f"显式格式: dt = {dt_explicit}, r = {alpha * dt_explicit / dx**2:.4f}")
    print(f"隐式格式: dt = {dt_implicit}, r = {alpha * dt_implicit / dx**2:.4f}")
    print(f"模拟总时长: {n_steps * dt_explicit:.3f} s (显式)")
    
    # ===== 显式方法求解 =====
    print("\n正在运行显式有限差分法...")
    history_explicit = solve_heat_explicit(u0.copy(), alpha, dx, dt_explicit, n_steps)
    
    # ===== 隐式方法求解 =====
    print("正在运行隐式有限差分法...")
    history_implicit = solve_heat_implicit(u0.copy(), alpha, dx, dt_implicit, n_steps)
    
    # ===== 结果分析 =====
    final_temp_explicit = history_explicit[-1]
    final_temp_implicit = history_implicit[-1]
    
    # 找最高温度位置
    max_idx_explicit = np.argmax(final_temp_explicit)
    max_idx_implicit = np.argmax(final_temp_implicit)
    
    print("\n" + "=" * 55)
    print("结果对比")
    print("=" * 55)
    print(f"{'时刻':<10} {'显式最高温度':<18} {'隐式最高温度':<18}")
    print("-" * 50)
    for t_idx in [0, 100, 250, 500]:
        if t_idx < len(history_explicit):
            t_explicit = t_idx * dt_explicit
            max_temp_explicit = np.max(history_explicit[t_idx])
            max_temp_implicit = np.max(history_implicit[t_idx])
            print(f"{t_explicit:.4f}s   {max_temp_explicit:<18.6f} {max_temp_implicit:<18.6f}")
    
    print("\n温度分布统计（最终时刻）:")
    print(f"  显式方法: 最大={final_temp_explicit.max():.6f}, 最小={final_temp_explicit.min():.6f}")
    print(f"  隐式方法: 最大={final_temp_implicit.max():.6f}, 最小={final_temp_implicit.min():.6f}")
    print(f"  两者差异: 最大绝对误差 = {np.max(np.abs(final_temp_explicit - final_temp_implicit)):.6f}")
