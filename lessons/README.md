# 编程反向阅读课程

为科学家女友设计的 Python 编程教程，通过分析真实科学计算代码来学习编程思维。

## 课程设计理念

**反向阅读法**：不从头读到尾，而是先看输入输出，再倒推代码逻辑。

### 如何使用这些文件

1. **先运行**：`python lesson_XX_xxx.py`，观察输出结果
2. **再看代码**：从文件末尾的 `if __name__ == "__main__":` 开始阅读
3. **倒推逻辑**：为什么输入会产生这个输出？每个函数起什么作用？
4. **理解数学**：这些方程/算法在科学上代表什么？

## 课程目录

### Lesson 01: 傅里叶变换 (Fourier Transform)
**文件**: `lesson_01_fourier_transform.py`  
**主题**: 从频率视角看信号  
**科学背景**: 任何信号都可以分解成不同频率的正弦波叠加  
**知识点**: 离散傅里叶变换(DFT)、FFT、频谱分析、奈奎斯特频率  
**核心问题**: 如何从时域信号找出隐藏的频率成分？  
**依赖**: `numpy`, `numpy.fft`

---

### Lesson 02: Monte Carlo 积分 (Monte Carlo Integration)
**文件**: `lesson_02_monte_carlo_integration.py`  
**主题**: 用随机性计算确定性  
**科学背景**: 通过大量随机采样来估算积分值  
**知识点**: 大数定律、概率论、数值积分、方差缩减  
**核心问题**: 为什么随机撒点可以用来计算定积分？  
**依赖**: `numpy`, `math`

---

### Lesson 03: 热传导方程 (Heat Equation)
**文件**: `lesson_03_heat_equation.py`  
**主题**: 热传导的数学之美  
**科学背景**: 偏微分方程描述热量如何在物体中传播  
**知识点**: 有限差分法、显式/隐式格式、CFL稳定性条件、三对角矩阵求解  
**核心问题**: 如何用计算机求解偏微分方程？  
**依赖**: `numpy`, `scipy.linalg`

---

### Lesson 04: 梯度下降法 (Gradient Descent)
**文件**: `lesson_04_gradient_descent.py`  
**主题**: 找到山谷的最快路径  
**科学背景**: 优化算法是机器学习/科学计算的核心  
**知识点**: 梯度下降、动量法(Momentum)、Adam优化器、黄金分割搜索  
**核心问题**: 如何让计算机"学会"找到最优解？  
**依赖**: `numpy`

---

### Lesson 05: SIR 传染病模型 (Epidemic Model)
**文件**: `lesson_05_sir_epidemic_model.py`  
**主题**: 传染病动力学  
**科学背景**: 用微分方程建模疾病传播过程  
**知识点**: 常微分方程数值解(Runge-Kutta)、SIR模型、基本再生数R₀  
**核心问题**: 疫情如何发展？防控措施为什么有效？  
**依赖**: `numpy`, `scipy.integrate.odeint`

---

## 学习建议

### 适合的对象
- 有科学/工程背景（理解微积分、概率论基础概念）
- 有一定数学基础，想学习编程
- 不喜欢死板的语法教程，喜欢从问题出发学习

### 逆向阅读技巧

**技巧1：先看输入输出**
```python
# 在代码末尾找到这段
if __name__ == "__main__":
    # 这里是程序的起点
    # 先理解这里发生了什么
```

**技巧2：追踪数据流**
```python
# 问自己：
# 1. 原始数据是什么？
# 2. 经过哪些变换？
# 3. 最终输出是什么？
```

**技巧3：理解数学假设**
- 每个函数背后的数学原理是什么？
- 方程中的参数有什么物理意义？
- 结果是否符合直觉？

**技巧4：动手修改**
- 把参数改大会发生什么？
- 如果去掉某个步骤会怎样？
- 能不能换成另一种算法？

## 环境依赖

```bash
pip install numpy scipy matplotlib
```

## 文件结构

```
lessons/
├── README.md                          # 本文件
├── lesson_01_fourier_transform.py      # 傅里叶变换
├── lesson_02_monte_carlo_integration.py # Monte Carlo 积分
├── lesson_03_heat_equation.py         # 热传导方程
├── lesson_04_gradient_descent.py       # 梯度下降法
└── lesson_05_sir_epidemic_model.py     # SIR 传染病模型
```

## 课程目标

完成本课程后，你将：
- ✅ 理解数值计算的基本方法
- ✅ 掌握 Python 处理科学问题的思维模式
- ✅ 能够阅读和理解他人的代码
- ✅ 建立"代码即数学"的理解

---

*"Science is knowing; engineering is doing. Code is the bridge."*
