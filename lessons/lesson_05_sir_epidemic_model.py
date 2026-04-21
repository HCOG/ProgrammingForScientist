"""
Lesson 05: SIR Epidemic Model — 传染病动力学
==============================================
目标：用微分方程建模传染病的传播过程
知识点：常微分方程数值解、疾病传播模型、参数估计、基本再生数 R₀

SIR 模型微分方程（Kermack-McKendrick, 1927）：
    dS/dt = -β * S * I / N      (易感者变化率)
    dI/dt = β * S * I / N - γ * I  (感染者变化率)
    dR/dt = γ * I                (康复者变化率)

其中：
    S: 易感者 (Susceptible)
    I: 感染者 (Infectious)
    R: 康复者 (Recovered/Removed)
    β: 传染率 (Transmission rate)
    γ: 康复率 (Recovery rate)
    R₀ = β/γ: 基本再生数 (衡量疾病传播能力)

学习目标（反向阅读提示）：
- 倒推：为什么 S+I+R 恒等于 N（总人口守恒）？
- 倒推：R₀ > 1 意味着什么？疫情会如何发展？
- 倒推：减少 β 或增加 γ 分别对应什么防控措施？
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def sir_model(y, t, beta, gamma, N):
    """
    SIR 模型的微分方程定义
    
    参数:
        y: 状态向量 [S, I, R]
        t: 时间点
        beta: 传染率
        gamma: 康复率
        N: 总人口
    
    返回:
        [dS/dt, dI/dt, dR/dt]
    """
    S, I, R = y
    
    # 传染力 (force of infection)
    lambda_ = beta * I / N
    
    dSdt = -lambda_ * S
    dIdt = lambda_ * S - gamma * I
    dRdt = gamma * I
    
    return [dSdt, dIdt, dRdt]


def compute_r0(beta, gamma):
    """计算基本再生数 R₀"""
    return beta / gamma


def simulate_sir(N, I0, beta, gamma, t_max, dt=0.1):
    """
    模拟 SIR 模型
    
    参数:
        N: 总人口
        I0: 初始感染人数
        beta: 传染率
        gamma: 康复率
        t_max: 模拟结束时间
        dt: 时间步长
    
    返回:
        t: 时间数组
        S, I, R: 各 compartment 的人数
    """
    # 初始条件
    S0 = N - I0
    R0 = 0
    y0 = [S0, I0, R0]
    
    # 时间点
    t = np.arange(0, t_max, dt)
    
    # 求解 ODE
    solution = odeint(sir_model, y0, t, args=(beta, gamma, N))
    S = solution[:, 0]
    I = solution[:, 1]
    R = solution[:, 2]
    
    return t, S, I, R


def find_peak_infection(t, I):
    """找出感染高峰"""
    peak_idx = np.argmax(I)
    return t[peak_idx], I[peak_idx]


def compute_final_size(I, R):
    """
    计算最终感染比例（感染结束后的累计康复者比例）
    这代表了疫情的最终规模
    """
    return R[-1]


def herd_immunity_threshold(R0):
    """
    计算群体免疫阈值
    
    当免疫人群比例 p > 1 - 1/R₀ 时，疫情开始消退
    即：如果有足够多的人免疫，病毒无法持续传播
    """
    return 1 - 1 / R0


if __name__ == "__main__":
    # ===== 基础参数设置 =====
    N = 10000          # 总人口
    I0 = 10            # 初始感染人数
    beta = 0.4         # 传染率
    gamma = 0.1        # 康复率 (平均感染期 = 1/gamma = 10 天)
    
    R0 = compute_r0(beta, gamma)
    print("=" * 55)
    print("SIR 传染病模型模拟")
    print("=" * 55)
    print(f"\n【模型参数】")
    print(f"  总人口 N = {N:,}")
    print(f"  初始感染 I₀ = {I0}")
    print(f"  传染率 β = {beta}")
    print(f"  康复率 γ = {gamma}")
    print(f"  平均感染期 = {1/gamma:.1f} 天")
    print(f"  基本再生数 R₀ = β/γ = {R0:.2f}")
    print(f"  群体免疫阈值 = 1 - 1/R₀ = {herd_immunity_threshold(R0):.1%}")
    
    # ===== 场景1：基准情景 =====
    print(f"\n【场景1：基准情景 (R₀ = {R0:.2f})】")
    t, S, I, R = simulate_sir(N, I0, beta, gamma, t_max=150, dt=0.5)
    
    peak_t, peak_I = find_peak_infection(t, I)
    final_infected_ratio = compute_final_size(I, R) / N
    
    print(f"  感染高峰时刻: 第 {peak_t:.1f} 天")
    print(f"  感染高峰人数: {peak_I:,.0f} (占总人口 {peak_I/N:.1%})")
    print(f"  最终感染比例: {final_infected_ratio:.1%}")
    
    # ===== 场景2：强防控措施（降低 β）=====
    print(f"\n【场景2：强防控 (β 降至 {beta*0.5:.2f})】")
    beta_reduced = beta * 0.5
    R0_reduced = compute_r0(beta_reduced, gamma)
    print(f"  新的 R₀ = {R0_reduced:.2f}")
    
    t2, S2, I2, R2 = simulate_sir(N, I0, beta_reduced, gamma, t_max=150, dt=0.5)
    peak_t2, peak_I2 = find_peak_infection(t2, I2)
    final_infected_ratio2 = compute_final_size(I2, R2) / N
    
    print(f"  感染高峰时刻: 第 {peak_t2:.1f} 天")
    print(f"  感染高峰人数: {peak_I2:,.0f} (占总人口 {peak_I2/N:.1%})")
    print(f"  最终感染比例: {final_infected_ratio2:.1%}")
    
    # ===== 场景3：疫苗接种（提高 γ 或直接免疫）=====
    print(f"\n【场景3：疫苗接种 (直接免疫 40% 人口)】")
    N_immune = int(N * 0.4)
    N_susceptible = N - N_immune
    I0_vaccinated = 10
    S0_vaccinated = N_susceptible - I0_vaccinated
    
    # 在疫苗接种人群中也保持相同的 R₀，但初始 S 减少了
    t3, S3, I3, R3 = simulate_sir(N_susceptible, I0_vaccinated, beta, gamma, t_max=150, dt=0.5)
    
    # 加上疫苗免疫的人群
    R3_vaccinated = R3 + N_immune
    
    peak_t3, peak_I3 = find_peak_infection(t3, I3)
    final_infected_ratio3 = R3_vaccinated[-1] / N
    
    print(f"  疫苗覆盖率: 40%")
    print(f"  有效总人口: {N_susceptible:,}")
    print(f"  感染高峰时刻: 第 {peak_t3:.1f} 天")
    print(f"  感染高峰人数: {peak_I3:,.0f}")
    print(f"  最终感染比例（含疫苗）: {final_infected_ratio3:.1%}")
    print(f"  直接因感染康复: {R3[-1]/N:.1%}")
    
    # ===== 敏感性分析 =====
    print(f"\n【敏感性分析：不同 R₀ 的影响】")
    print(f"{'R₀':<6} {'高峰时刻':<10} {'高峰感染%':<12} {'最终感染%':<12}")
    print("-" * 45)
    
    for r0_multiplier in [0.5, 0.75, 1.0, 1.25, 1.5]:
        beta_test = beta * r0_multiplier
        R0_test = compute_r0(beta_test, gamma)
        t_test, _, I_test, _ = simulate_sir(N, I0, beta_test, gamma, t_max=200, dt=0.5)
        peak_t_test, peak_I_test = find_peak_infection(t_test, I_test)
        _, _, _, R_test = simulate_sir(N, I0, beta_test, gamma, t_max=200, dt=0.5)
        final_ratio = R_test[-1] / N
        
        print(f"{R0_test:<6.2f} {peak_t_test:<10.1f} {peak_I_test/N:<12.1%} {final_ratio:<12.1%}")
    
    # ===== 关键洞察 =====
    print("\n" + "=" * 55)
    print("【关键洞察】")
    print("=" * 55)
    print(f"1. 当 R₀ > 1 时，疫情才会持续传播")
    print(f"2. R₀ = {R0:.2f} 意味着每个感染者平均传染 {R0:.2f} 个人")
    print(f"3. 降低 β（如戴口罩、减少接触）可直接降低 R₀")
    print(f"4. 提高 γ（如快速检测、隔离治疗）也可降低 R₀")
    print(f"5. 群体免疫阈值 = 1 - 1/R₀ = {herd_immunity_threshold(R0):.1%}")
    print(f"   当超过这个比例的人免疫后，疫情开始消退")
