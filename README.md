# Humanoid Robot Balancing Simulation(MuJoCo)

## Table of Contents

1. [Overview](#1-overview)
2. [Math Basics](#2-math-basics)
3. [Project Structure](#3-project-structure)
4. [How to Run](#4-how-to-run)
5. [Requirements](#5-requirements)
6. [References](#6-References)


---

## 1. Overview
This project simulates a humanoid robot (Unitree G1) balancing in the **MuJoCo** physics engine. It focuses on keeping the robot standing safely on two feet (Double Support Phase) using two main methods:
1. Uses standard PID combined with Jacobian control for the Center of Mass (COM) and Zero Moment Point (ZMP).
2. Adds Model Predictive Control (MPC) based on the Linear Inverted Pendulum Model (LIPM) to plan safe movements using Quadratic Programming (QP).

---

## 2. Math Basics

### 1. Liner Inverted Pendulum Model(LIPM)
### 1.1 Dynamics Equation

The LIPM is a simplified model of a bipedal robot, assuming the entire mass is concentrated at the **center of mass (COM)** at a constant height $h$ above the ground.

The dynamics equation is given by:

$$\ddot{x} = \frac{g}{h}(x - p)$$

where:
- $x$ — horizontal position of the COM
- $p$ — position of the Zero Moment Point (ZMP)
- $h$ — height of the COM (constant)
- $g = 9.81 \, \text{m/s}^2$

Rearranging, the ZMP — COM relationship is:

$$p = x - \frac{h}{g}\ddot{x}$$

### 1.2 Discrete State-Space Representation

The state vector is chosen with **jerk** (the third derivative of position) as the control input:

$$\mathbf{x}_k = \begin{bmatrix} x_k \\ \dot{x}_k \\ \ddot{x}_k \end{bmatrix}, \qquad u_k = \dddot{x}_k$$

The discrete-time state equation with time step $\Delta t$ is:

$$\mathbf{x}_{k+1} = \mathbf{A}\,\mathbf{x}_k + \mathbf{B}\,u_k$$

$$\mathbf{A} = \begin{bmatrix} 1 & \Delta t & \dfrac{\Delta t^2}{2} \\[6pt] 0 & 1 & \Delta t \\[4pt] 0 & 0 & 1 \end{bmatrix}, \qquad \mathbf{B} = \begin{bmatrix} \dfrac{\Delta t^3}{6} \\[6pt] \dfrac{\Delta t^2}{2} \\[4pt] \Delta t \end{bmatrix}$$

The ZMP output is:

$$z_k = \mathbf{C}\,\mathbf{x}_k, \qquad \mathbf{C} = \begin{bmatrix} 1 & 0 & -\dfrac{h}{g} \end{bmatrix}$$

### 2. Model Predictive Control (MPC) 
MPC plans the optimal COM path by looking ahead $N$ steps ($N=20$). It ensures the ZMP stays safely inside the foot area. The robot is modeled as an inverted pendulum:
$$x_{k+1} = A x_k + B u_k$$
$$p_{ZMP} = C x_k = p - \frac{h}{g}a$$
*(Where $x$ is position, velocity, and acceleration of the COM, $h$ is height, and $g$ is gravity).*

**Cost Function:** Minimizes the ZMP error and prevents sudden jerky movements ($u_k$):
$$J = \sum_{i=1}^{N} \left( Q \| p_{ZMP, i} - p_{ZMP}^{ref} \|^2 + R \| u_i \|^2 \right)$$
**Constraints:** ZMP must stay inside the support polygon (the feet):
$$ZMP_{min} \le P_{ZMP, i} \le ZMP_{max}$$
The `OSQP` solver runs at 100Hz to find the best COM state, which is then sent to the Jacobian controller.

### 3. COM & ZMP Jacobian Control 
**Goal:** Pull the COM to the desired position $\mathbf{r}_{\text{COM}}^{\text{des}}$ by adjusting joint velocities.

Relative COM–foot Jacobian:

$$\mathbf{J}_{\text{rel}} = \mathbf{J}_{\text{COM}} - \mathbf{J}_{\text{foot}}$$

Joint velocity command derived from COM error:

$$\Delta\mathbf{q}_{\text{COM}} = \mathbf{J}_{\text{rel}}^{+} \cdot \frac{k_{\text{COM}}\,(\mathbf{r}_{\text{COM}}^{\text{des}} - \mathbf{r}_{\text{COM}})}{\Delta t}$$

Compensating torque:

$$\tau_{\text{COM},i} = K_{p,i}\,(\Delta q_{\text{COM},i} \cdot \Delta t) + K_{v,i}\,\Delta q_{\text{COM},i}$$

Actual ZMP measured from contact forces (see Section 5.2), error compensated by:

$$\Delta\mathbf{q}_{\text{ZMP}} = \mathbf{J}_{\text{rel}}^{+} \cdot \frac{k_{\text{ZMP}}\,(\mathbf{p}^{\text{des}} - \mathbf{p}_{\text{COP}})}{\Delta t}$$

$$\tau_{\text{ZMP},i} = K_{p,i}\,(\Delta q_{\text{ZMP},i} \cdot \Delta t) + K_{v,i}\,\Delta q_{\text{ZMP},i}$$

### 4. Joint-Space PID Control 
Calculates the final motor torque ($\tau$) needed to move the joints to the target positions and speeds:
$$\tau_{PID} = K_p (q_{des} - q) + K_v (\dot{q}_{des} - \dot{q}) + K_i \int (q_{des} - q) dt$$

### 5. Hierarchical Inverse Kinematics (IK)
Uses Null-space projection to set up the starting pose by prioritizing tasks:
* **Task 1 :** Keep the COM and feet stable ($J_1$).
* **Task 2 :** Adjust the upper body posture ($J_2$).

---

## 3. Project Structure

```
┌──────────────────────────────────────────────────────────────────────┐
│                          (PD + ZMP)                                  │
│                            run.py                                    │
│                                                                      │
└────────────────────────────┬─────────────────────────────────────────┘
                             │ import
┌────────────────────────────▼─────────────────────────────────────────┐
│                        datactrlpd.py                                 │
│  ┌──────────────┐    ┌──────────────────────────────────────────┐    │
│  │ selectRobot  │    │              myRobot                     │    │
│  │  - Load XML  │    │  controller() ──► PIDcontrol()           │    │
│  │  - Gain PD   │    │             ├──► COMcontrol() [Jacobian] │    │
│  │  - numik IK  │    │             └──► ZMPcontrol() [Jacobian] │    │
│  └──────────────┘    └──────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                             (MPC)                                    │
│                           runmpc.py                                  │
│                                                                      │
└────────────────────────────┬─────────────────────────────────────────┘
                             │ import
┌────────────────────────────▼─────────────────────────────────────────┐
│                           mpc.py                                     │
│  ┌──────────────┐    ┌──────────────────────────────────────────┐    │
│  │  LIPM_MPC    │    │              myRobot                     │    │
│  │  - Px,Pu,Pz  │◄───│  controller() ──► [LIPM_MPC @ 100Hz]     │    │
│  │  - QP (OSQP) │    │             ├──► PIDcontrol()            │    │
│  └──────────────┘    │             ├──► COMcontrol() [Jacobian] │    │
│                      │             └──► ZMPcontrol() [Jacobian] │    │
│  ┌──────────────┐    └──────────────────────────────────────────┘    │
│  │   numik()    │                        +MPC                        │
│  │  IK 2 level  │                                                    │
│  └──────────────┘                                                    │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                    MuJoCo Physics Engine                             │
│             Unitree G1  (g1.xml  +  scene_basic.xml)                 │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 4. How to Run
```bash
python run.py  
```
Or
```bash
python runmpc.py 
```

### 5. Requirements
1. Python
2. Mujoco
```bash
pip install mujoco numpy scipy matplotlib moviepy qpsolvers osqp
```
## 6. References

- Kajita, S. et al. (2003). *Biped Walking Pattern Generation by Using Preview Control of Zero-Moment Point.* ICRA.
- Wieber, P.-B. (2006). *Trajectory Free Linear Model Predictive Control for Stable Walking in the Presence of Strong Perturbations.* Humanoids.
- Nakamura, Y. & Hanafusa, H. (1987). *Optimal Redundancy Control of Robot Manipulators.* IJRR.
- Vukobratovic, M. & Borovac, B. (2004). *Zero-Moment Point — Thirty Five Years of Its Life.* International Journal of Humanoid Robotics.
- MuJoCo Documentation: https://mujoco.readthedocs.io
- OSQP Solver: https://osqp.org
- MuJoCo Menagerie (Unitree G1): https://github.com/google-deepmind/mujoco_menagerie
