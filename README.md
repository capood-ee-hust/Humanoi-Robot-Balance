# Humanoid Robot Balancing Simulation (MuJoCo)

## Table of Contents
- [Overview](#-overview)
- [Math Basics](#-math-basics)
- [Project Structure](#️-project-structure)
- [How to Run](#-how-to-run)
- [PID vs. MPC Comparison](#️-pid-vs-mpc-comparison)
- [Author](#-author)

---

## Overview
This project simulates a humanoid robot (Unitree G1) balancing in the **MuJoCo** physics engine. It focuses on keeping the robot standing safely on two feet (Double Support Phase) using two main methods:
1. Uses standard PID combined with Jacobian control for the Center of Mass (COM) and Zero Moment Point (ZMP).
2. Adds Model Predictive Control (MPC) based on the Linear Inverted Pendulum Model (LIPM) to plan safe movements using Quadratic Programming (QP).

---

## Math Basics

### 1. Model Predictive Control (MPC) 
MPC plans the optimal COM path by looking ahead $N$ steps ($N=20$). It ensures the ZMP stays safely inside the foot area. The robot is modeled as an inverted pendulum:
$$x_{k+1} = A x_k + B u_k$$
$$p_{ZMP} = C x_k = p - \frac{h}{g}a$$
*(Where $x$ is position, velocity, and acceleration of the COM, $h$ is height, and $g$ is gravity).*

**Cost Function:** Minimizes the ZMP error and prevents sudden jerky movements ($u_k$):
$$J = \sum_{i=1}^{N} \left( Q \| p_{ZMP, i} - p_{ZMP}^{ref} \|^2 + R \| u_i \|^2 \right)$$
**Constraints:** ZMP must stay inside the support polygon (the feet):
$$ZMP_{min} \le p_{ZMP, i} \le ZMP_{max}$$
The `OSQP` solver runs at 100Hz to find the best COM state, which is then sent to the Jacobian controller.

### 2. COM & ZMP Jacobian Control 
Converts the target COM movement from 3D space into joint angles using the Jacobian matrix ($J_{rel} = J_{COM} - J_{foot}$):
$$\Delta \dot{q}_{COM} = J_{rel}^{\dagger} \cdot \left( K_{COM} \frac{x_{COM}^{des} - x_{COM}^{act}}{\Delta t} \right)$$

### 3. Joint-Space PID Control - Low Level
Calculates the final motor torque ($\tau$) needed to move the joints to the target positions and speeds:
$$\tau_{PID} = K_p (q_{des} - q) + K_v (\dot{q}_{des} - \dot{q}) + K_i \int (q_{des} - q) dt$$

### 4. Hierarchical Inverse Kinematics (IK)
Uses Null-space projection to set up the starting pose by prioritizing tasks:
* **Task 1 (High Priority):** Keep the COM and feet stable ($J_1$).
* **Task 2 (Low Priority):** Adjust the upper body posture ($J_2$).

---

## Project Structure

The project is split into two independent versions for easy comparison:

**Version 1: Baseline PID / Jacobian Control**
* **`datactrlpd.py`**: Contains the `myRobot` class, MuJoCo data extraction, IK solver, and the purely reactive PID/COM/ZMP controllers.
* **`run.py`**: The main script to run the baseline simulation.

**Version 2: Advanced MPC Control**
* **`mpc.py`**: Upgrades `datactrlpd.py` by adding the `LIPM_MPC` class. It uses a 3-level structure: High (MPC plans path) -> Mid (Jacobian translates path) -> Low (PID moves motors).
* **`runmpc.py`**: The main script to run the MPC simulation.

---

## How to Run

### Requirements
Make sure you have Python 3.x and install these libraries:
```bash
pip install mujoco numpy scipy matplotlib moviepy qpsolvers osqp