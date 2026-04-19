import numpy as np
import os
import scipy.sparse as sparse
from qpsolvers import solve_qp
import xml.etree.ElementTree as ET
import mujoco as mj
import mujoco.viewer
from scipy.interpolate import CubicSpline
from scipy.linalg import solve_discrete_are
from copy import deepcopy
import time
from moviepy import ImageSequenceClip

#Find Euler angles from COM position and Contact position of inverted pendulum model
def findeulr(qcm,qcp,l):
    return np.array([np.arctan2(qcp[1] - qcm[1], qcm[2] - qcp[2]), np.arctan2(qcm[0] - qcp[0], np.linalg.norm(qcm[1:] - qcp[1:]) ), 0.0])

_FLOAT_EPS = np.finfo(np.float64).eps
_EPS4 = _FLOAT_EPS * 4.0

def euler2quat(euler):
    euler = np.asarray(euler, dtype=np.float64)
    ai, aj, ak = euler[..., 2] / 2, -euler[..., 1] / 2, euler[..., 0] / 2
    si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
    ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk
    quat = np.empty(euler.shape[:-1] + (4,), dtype=np.float64)
    quat[..., 0] = cj * cc + sj * ss
    quat[..., 3] = cj * sc - sj * cs
    quat[..., 2] = -(cj * ss + sj * cc)
    quat[..., 1] = cj * cs - sj * sc
    return quat

def mat2euler(mat):
    mat = np.asarray(mat, dtype=np.float64)
    cy = np.sqrt(mat[..., 2, 2] * mat[..., 2, 2] + mat[..., 1, 2] * mat[..., 1, 2])
    condition = cy > _EPS4
    euler = np.empty(mat.shape[:-1], dtype=np.float64)
    euler[..., 2] = np.where(condition, -np.arctan2(mat[..., 0, 1], mat[..., 0, 0]), -np.arctan2(-mat[..., 1, 0], mat[..., 1, 1]))
    euler[..., 1] = np.where(condition, -np.arctan2(-mat[..., 0, 2], cy), -np.arctan2(-mat[..., 0, 2], cy))
    euler[..., 0] = np.where(condition, -np.arctan2(mat[..., 1, 2], mat[..., 2, 2]), 0.0)
    return euler

def quat2mat(quat):
    quat = np.asarray(quat, dtype=np.float64)
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    Nq = np.sum(quat * quat, axis=-1)
    s = 2.0 / Nq
    X, Y, Z = x * s, y * s, z * s
    wX, wY, wZ = w * X, w * Y, w * Z
    xX, xY, xZ = x * X, x * Y, x * Z
    yY, yZ, zZ = y * Y, y * Z, z * Z
    mat = np.empty(quat.shape[:-1] + (3, 3), dtype=np.float64)
    mat[..., 0, 0] = 1.0 - (yY + zZ)
    mat[..., 0, 1] = xY - wZ
    mat[..., 0, 2] = xZ + wY
    mat[..., 1, 0] = xY + wZ
    mat[..., 1, 1] = 1.0 - (xX + zZ)
    mat[..., 1, 2] = yZ - wX
    mat[..., 2, 0] = xZ - wY
    mat[..., 2, 1] = yZ + wX
    mat[..., 2, 2] = 1.0 - (xX + yY)
    return np.where((Nq > _FLOAT_EPS)[..., np.newaxis, np.newaxis], mat, np.eye(3))

def quat2euler(quat):
    return mat2euler(quat2mat(quat))

def data2q(data):
    q = 0 * data.qvel.copy()
    qqt = data.qpos[3:7].copy()
    qeulr = quat2euler(qqt)
    for i in np.arange(0, 3): q[i] = data.qpos[i].copy()
    for i in np.arange(3, 6): q[i] = qeulr[i - 3].copy()
    for i in np.arange(6, len(data.qvel)): q[i] = data.qpos[i + 1].copy()
    return q

def q2data(data, q0):
    import copy
    qqt = euler2quat(q0[3:6])
    for i in np.arange(0, 3): data.qpos[i] = copy.copy(q0[i])
    for i in np.arange(3, 7): data.qpos[i] = copy.copy(qqt[i - 3])
    for i in np.arange(7, len(data.qpos)): data.qpos[i] = copy.copy(q0[i - 1])
    return data

# MPC Controller
class LIPM_MPC:
    def __init__(self, dt, h, N):
        self.dt = dt
        self.h = h
        self.N = N  #số steps)
        g = 9.81
        # State-Space Matrix
        self.A = np.array([
            [1.0,  dt, (dt**2)/2.0],
            [0.0, 1.0,          dt],
            [0.0, 0.0,         1.0]
        ])
        self.B = np.array([
            [(dt**3)/6.0],
            [(dt**2)/2.0],
            [         dt]
        ])
        self.C = np.array([[1.0, 0.0, -h/g]]) #tính ZMP
        
        #tính toán các ma trận dự đoán (Prediction Matrices)
        self.Px = np.zeros((3*N, 3))
        self.Pu = np.zeros((3*N, N))
        self.Pz = np.zeros((N, 3))
        self.Pzu = np.zeros((N, N))
        
        A_p = np.eye(3)
        for i in range(N):
            A_p = self.A @ A_p
            self.Px[i*3:(i+1)*3, :] = A_p
            self.Pz[i, :] = self.C @ A_p
            for j in range(i+1):
                A_pow = np.linalg.matrix_power(self.A, i-j)
                val = self.C @ A_pow @ self.B
                self.Pzu[i, j] = val[0, 0]
                self.Pu[i*3:(i+1)*3, j] = (A_pow @ self.B).flatten()           
        # Cost Function
        self.Q = np.eye(N) * 1000.0  # Ưu tiên bám sát điểm ZMP mục tiêu
        self.R = np.eye(N) * 1e-5    # Hạn chế giật cục (Jerk)
        # Ma trận Hessian cho QP
        self.H = 2.0 * (self.Pzu.T @ self.Q @ self.Pzu + self.R)
        self.H = sparse.csc_matrix(self.H) # OSQP yêu cầu ma trận thưa (sparse)

    def solve(self, x0, z_ref, z_min, z_max):
        # x0: Trạng thái hiện tại [pos, vel, acc]
        Z_ref_vec = np.ones(self.N) * z_ref
        # Gradient f
        f = 2.0 * self.Pzu.T @ self.Q @ (self.Pz @ x0 - Z_ref_vec)
        #  ràng buộc: z_min <= zmp <= z_max
        G = np.vstack([self.Pzu, -self.Pzu])
        h_max = z_max - self.Pz @ x0
        h_min = -(z_min - self.Pz @ x0)
        h_bound = np.concatenate([h_max, h_min])
        G = sparse.csc_matrix(G)
        # Gọi OSQP
        try:
            U = solve_qp(P=self.H, q=f, G=G, h=h_bound, solver="osqp")
        except:
            U = None
        if U is None:
            U = np.zeros(self.N) 
        x_next = self.A @ x0 + self.B @ np.array([U[0]])
        return x_next.flatten()
# --- SCENE & XML GENERATOR ---
def scenegen(xml_tree):
    root = xml_tree.getroot()
    return ET.tostring(root)
def addrobot2scene(xml_tree, robotpath):
    root = xml_tree.getroot()
    for tag1 in root.findall("include"):
        tag1.attrib['file'] = robotpath 
    return ET.tostring(root)
# --- THIẾT LẬP ROBOT UNITREE G1 ---
def selectRobot( vel, step_len, spno):
    
    dirname = os.path.dirname(__file__) 
    xml_path = os.path.join(dirname, 'scene_basic.xml')
    xml_tree = ET.parse(xml_path) 
    
    # Tạo mặt phẳng
    xml_str = scenegen(xml_tree)
    xml_tree = ET.ElementTree(ET.fromstring(xml_str))

    # model Unitree G1
    robotpath = os.path.join(dirname, 'Unitree/g1.xml')
    xml_str = addrobot2scene(xml_tree, robotpath)

    model = mj.MjModel.from_xml_string(xml_str)  
    data = mj.MjData(model)  
    model.opt.timestep = 0.0005

    ub_jnts = np.arange(18, model.nv)
    left_legjnts = np.arange(6, 12)
    right_legjnts = np.arange(12, 18)
    foot_size = np.array([0.050, 0.040]) 

    Kp = np.zeros(model.nu)
    Kv = np.zeros(model.nu)
    Ki = np.zeros(model.nu)
    
    # Gain G1
    Kp[0:12] = 5000 
    Kv[0:12] = 500  
    Kp[3] = 10000
    Kv[3] = 100
    Kp[9] = 10000
    Kv[9] = 100
    Kp[12:] = 50
    Kv[12:] = 1

    humn = myRobot(ub_jnts, left_legjnts, right_legjnts, foot_size, vel)
    humn.mj2humn(model, data)

    # Initial pose cho G1
    q0 = data2q(data)
    q0[[2, 9, 10, 15, 16]] = [0.9 * q0[2], 0.5, -0.5, 0.5, -0.5]

    humn.r_com[0] = -0.0
    humn.r_com[1] = (0 + humn.o_left[1]) / 3
    humn.r_com[2] = 0.9 * humn.r_com[2]

    humn.o_left[1] = 0.5 * humn.o_left[1]
    humn.o_right[1] = 0.5 * humn.o_right[1]

    humn.o_left[0] = 0.0
    humn.o_left[2] = 0
    humn.o_right[0] = 0.0
    humn.o_right[2] = -0.0

    data.qvel[0] = vel
    data.qvel[1] = 0.0
    humn.spno = spno  

    if humn.spno == 2:
        data.qvel[0] = 0
        humn.r_com[1] = 0.0
        humn.r_com[2] = 0.8 / 0.9 * humn.r_com[2]

    # Tính toán Inverse Kinematics cho tư thế đứng
    ti = 0
    while ti < 100:
        delt = 1 / 100
        q, gradH0 = numik(model, data, q0, delt, humn.r_com, humn.o_left, humn.o_right, humn.ub_jnts, np.zeros([model.nv]), False, 0, 0)
        q0 = q.copy()
        ti += delt

    data = q2data(data, q)
    humn.q0 = q.copy()
    humn.mj2humn(model, data)

    print('mass=', humn.m, ' rCOM=', humn.r_com)
    print('o_left=', humn.o_left, 'o_right=', humn.o_right)
    
    humn.xlimft = 0.5 
    humn.ylimft = abs(humn.o_left[1] - humn.o_right[1])   
    humn.Stlr = np.array([1, 0])
    humn.zSw = 0.05 
    humn.step_time = step_len / (2 * vel) 
    humn.step_len = step_len 
    humn.sspbydsp = 2
    humn.Tsip = 0  
    humn.cam_dist = 2
    # Active Balancing Weights
    humn.COMctrl = 0.4   
    humn.ZMPctrl = 0.05  
    humn.k_ub = 0.5      

    humn.init_controller(Kp, Kv, Ki)
    humn.posCTRL = True 

    return humn, model, data

# Điều khiển Robot
class myRobot:
    def __init__(self, ub_jnts, left_legjnts, right_legjnts, foot_size, vel):
        self.ub_jnts = ub_jnts
        self.left_legjnts = left_legjnts
        self.right_legjnts = right_legjnts
        self.foot_size = foot_size
        self.cam_dist = 1
        self.vel = vel
        self.Tsip = 0  
        self.WD = 0 
        self.posCTRL = False 
        self.KINctrl = False 
        self.ZMPctrl = 0
        self.AMctrl = 0.0
        self.k_ub = 0.1000 
        self.k_L = self.AMctrl * 1 / (1 + self.k_ub * 0 * np.linalg.norm(np.zeros([6])))
        self.FWctrl = 0
        self.ftctrl = 0
        self.xn = np.array([])
        self.dxn = np.array([])
        self.fn = np.array([])
        self.AM_CMspl = []
        for i in range(3):
            self.AM_CMspl.append(CubicSpline([0, 1], [0, 0], bc_type='clamped'))

    def mj2humn(self, model, data):
        mj.mj_fwdPosition(model, data)
        mj.mj_comVel(model, data)
        mj.mj_subtreeVel(model, data)
        
        self.ti = data.time
        self.m = mj.mj_getTotalmass(model) 
        self.r_com = data.subtree_com[0].copy() 
        self.dr_com = data.subtree_linvel[0].copy()  
        self.dq_com = data.cvel[1].copy() 
        self.o_left = data.site('left_foot_site').xpos.copy()  
        self.o_right = data.site('right_foot_site').xpos.copy()  
        
        self.fcl = np.zeros(3)
        self.fcr = np.zeros(3)
        self.fcn = 0.0
        self.rf = np.zeros(3)
        self.rcop = np.zeros(3)
        
        self.q = data2q(data)
        self.dq = 0 * data.qvel.copy()
        
        self.q_err = np.zeros([model.nv])
        self.Eqdt = 0 * data.qvel 
        self.E_init = 1 / 2 * self.m * np.linalg.norm(self.dr_com)**2 + self.m * 9.81 * self.r_com[2] 
        self.gradH = np.ones([model.nv]) 

        self.qTraj_des = np.array([self.q.copy()])
        self.dqtTraj_des = self.dq.copy()
        self.qTraj_act = self.q.copy()
        self.dqTraj_act = self.dq.copy()
        self.tTraj = data.time
        self.CAMTraj = np.zeros(3)
        self.torqueTraj = np.zeros(model.nu)
        self.fLTraj = np.zeros(6)
        self.fRTraj = np.zeros(6)

    def updateTrajData(self, model, data):
        self.qTraj_des = np.vstack((self.qTraj_des, self.q.copy()))
        self.dqtTraj_des = np.vstack((self.dqtTraj_des, self.dq.copy()))
        self.qTraj_act = np.vstack((self.qTraj_act, self.q.copy()))    
        self.dqTraj_act = np.vstack((self.dqTraj_act, self.dq.copy()))
        self.tTraj = np.vstack((self.tTraj, data.time))
        self.torqueTraj = np.vstack((self.torqueTraj, data.actuator_force.copy()))
        self.fLTraj = np.vstack((self.fLTraj, data.sensordata[0:6].copy()))
        self.fRTraj = np.vstack((self.fRTraj, data.sensordata[6:12].copy()))

    def mjfc(self, model, data):
        self.fcl = np.zeros([3])
        self.fcl[0] = data.xfrc_applied[model.site_bodyid[data.site("left_foot_site").id]][2]
        self.fcr = np.zeros([3])
        self.fcr[0] = data.xfrc_applied[model.site_bodyid[data.site("right_foot_site").id]][2]
        self.fcn = self.fcl[0] + self.fcr[0]
        self.rfl = np.zeros([3])
        self.rfr = np.zeros([3])
        self.rf = np.zeros([3])
        for i in np.arange(0, data.ncon):
            fci = np.zeros([6])
            try:
                mj.mj_contactForce(model, data, i, fci)
                self.fcn = self.fcn + abs(fci[0])
                self.rf = self.rf + np.array(data.contact[i].pos) * abs(fci[0])  
                if model.geom_bodyid[data.contact[i].geom2] == model.site_bodyid[data.site("left_foot_site").id] or model.geom_bodyid[data.contact[i].geom1] == model.site_bodyid[data.site("left_foot_site").id]:  
                    self.fcl = self.fcl + fci[0:3]
                    self.rfl = self.rfl + np.array(data.contact[i].pos) * abs(fci[0])
                elif model.geom_bodyid[data.contact[i].geom2] == model.site_bodyid[data.site("right_foot_site").id] or model.geom_bodyid[data.contact[i].geom1] == model.site_bodyid[data.site("right_foot_site").id]:  
                    self.fcr = self.fcr + fci[0:3]
            except:
                pass

    def init_controller(self, Kp, Kv, Ki):
        self.Kp = Kp
        self.Kv = Kv
        self.Ki = Ki
        # khởi tạo bộ đk mpc
        dt = 0.0005  
        h_com = self.r_com[2] # Lấy chiều cao Trọng tâm ban đầu
        N_horizon = 20        
        
        # 2 bộ giải độc lập cho trục x (tiến lùi) và trục y (trái phải)
        self.mpc_X = LIPM_MPC(dt, h_com, N_horizon)
        self.mpc_Y = LIPM_MPC(dt, h_com, N_horizon)
        
        # Bộ nhớ lưu trạng thái nội bộ của MPC [pos, vel, acc]
        self.state_x = np.array([self.r_com[0], 0.0, 0.0])
        self.state_y = np.array([self.r_com[1], 0.0, 0.0])

    def controller(self, model, data):
        self.tau = 0 * data.ctrl
        q = data2q(data)
        self.qdes = np.zeros([model.nv])
        self.dqdes = np.zeros([model.nv])
        self.ddqdes = np.zeros([model.nv])
        
        self.mjfc(model, data)
        # Lấy quỹ đạo khớp tĩnh 
        for i in np.arange(0, model.nv):
            self.qdes[i] = self.qspl[i](data.time)
            self.dqdes[i] = self.qspl[i](data.time, 1)
 
        if not hasattr(self, 'mpc_tick'):
            self.mpc_tick = 0

        if self.mpc_tick % 20 == 0:
           
            z_ref_x = self.oCPx(data.time)
            z_ref_y = self.oCPy(data.time)
            
            # Vùng an toàn của ZMP (Cho phép ZMP xê dịch tối đa 5cm quanh tâm)
            z_min_x, z_max_x = z_ref_x - 0.05, z_ref_x + 0.05
            z_min_y, z_max_y = z_ref_y - 0.05, z_ref_y + 0.05
            
            # solve QP problem
            self.state_x = self.mpc_X.solve(self.state_x, z_ref_x, z_min_x, z_max_x)
            self.state_y = self.mpc_Y.solve(self.state_y, z_ref_y, z_min_y, z_max_y)
            
        self.mpc_tick += 1
        
    
        self.ocm_des = np.array([self.state_x[0], self.state_y[0], self.oCMz(data.time)])
       
        self.tau_PID = self.PIDcontrol(model, data)

        if self.COMctrl > 0:
            self.tau_PID += self.COMcontrol(model, data)
            
        if self.ZMPctrl > 0:
            self.tau_PID += self.ZMPcontrol(model, data)

        # Xuất lệnh Torque ra động cơ
        if self.posCTRL:
            self.tau = self.qdes[6:model.nv].copy()
        else:
            for i in range(model.nu):
                self.tau[i] = self.tau_PID[i] / model.actuator_gear[i][0]

        return self.tau

    def PIDcontrol(self, model, data):
        q = data2q(data)
        self.tau_PID = 0 * data.ctrl
        for i in range(model.nu):
            self.tau_PID[i] = (self.Kp[i]) * (self.qdes[i + 6] - q[i + 6]) + self.Kv[i] * (self.dqdes[i + 6] - data.qvel[i + 6]) + self.Ki[i] * (self.Eqdt[i+6])
            self.Eqdt[i+6] = self.Eqdt[i+6] + (self.qdes[i + 6] - q[i + 6]) * model.opt.timestep
        return self.tau_PID

    def COMcontrol(self, model, data):  
        self.rcom = data.subtree_com[0].copy()  
        if self.COMctrl == 0:
            self.tau_COM = 0 * data.ctrl
        else:
            jnts = range(model.nv)
            Jcm = np.zeros((3, model.nv))  
            Jct = np.zeros((3, model.nv))  
            mj.mj_jacSubtreeCom(model, data, Jcm, 0)

            if abs(self.fcl[0]) > 0 and abs(self.fcr[0]) == 0:
                mj.mj_jacSite(model, data, Jct, None, data.site("left_foot_site").id)
            elif abs(self.fcr[0]) > 0 and abs(self.fcl[0]) == 0:
                mj.mj_jacSite(model, data, Jct, None, data.site("right_foot_site").id)
            else:
                if self.o_right[0] > self.o_left[0]:
                    mj.mj_jacSite(model, data, Jct, None, data.site("left_foot_site").id)
                else:
                    mj.mj_jacSite(model, data, Jct, None, data.site("right_foot_site").id)

            J2 = Jcm - Jct
            delx2 = 1 * self.COMctrl * (self.ocm_des - self.rcom) / model.opt.timestep
            dq_err = 1 * np.matmul(np.linalg.pinv(J2), delx2)
            self.q_err = self.q_err + dq_err * model.opt.timestep

            self.tau_COM = 0 * data.ctrl
            for i in np.arange(0, model.nu):
                self.tau_COM[i] = self.Kp[i] * (dq_err[i + 6] * model.opt.timestep) + self.Kv[i] * (dq_err[i + 6])  

            self.dqdes = self.dqdes + dq_err
            self.qdes = self.qdes + dq_err * model.opt.timestep

        return self.tau_COM

    def ZMPcontrol(self, model, data):
        self.ocp_des = np.array([self.oCPx(data.time), self.oCPy(data.time), self.oCPz(data.time)]) 
        self.mjfc(model, data)
        if abs(self.fcn) > 0:
            self.rcop = self.rf / abs(self.fcn)
        else: 
            self.rcop = self.ocp_des 

        if self.ocp_des[2] > self.rcom[2]:
            if abs(self.fcn) == 0:
                self.rcop[2] = 0
            self.ocp_des[0] = self.ocm_des[0] + (self.ocm_des[0] - self.ocp_des[0]) * abs(self.rcop[2] - self.ocm_des[2]) / (self.ocp_des[2] - self.ocm_des[2])
            self.ocp_des[1] = self.ocm_des[1] + (self.ocm_des[1] - self.ocp_des[1]) * abs(self.rcop[2] - self.ocm_des[2]) / (self.ocp_des[2] - self.ocm_des[2])
            self.ocp_des[2] = self.rcop[2]
        
        if abs(self.fcn) == 0:
            self.rcop = self.ocp_des 

        if self.ZMPctrl == 0:
            self.tau_ZMP = 0 * data.ctrl
        else:
            jnts = range(model.nv)
            Jcm = np.zeros((3, model.nv))  
            Jct = np.zeros((3, model.nv))  
            mj.mj_jacSubtreeCom(model, data, Jcm, 0)

            if abs(self.fcl[0]) > 0 and abs(self.fcr[0]) == 0:
                mj.mj_jacSite(model, data, Jct, None, data.site("left_foot_site").id)
            elif abs(self.fcr[0]) > 0 and abs(self.fcl[0]) == 0:
                mj.mj_jacSite(model, data, Jct, None, data.site("right_foot_site").id)
            else:
                if self.o_right[0] > self.o_left[0]:
                    mj.mj_jacSite(model, data, Jct, None, data.site("left_foot_site").id)
                else:
                    mj.mj_jacSite(model, data, Jct, None, data.site("right_foot_site").id)

            J2 = Jcm - Jct
            delx2 = 1 * self.ZMPctrl * (self.ocp_des - self.rcop) / model.opt.timestep
            dq_err = 1 * np.matmul(np.linalg.pinv(J2), delx2)
            self.q_err = self.q_err + dq_err * model.opt.timestep

            self.tau_ZMP = 0 * data.ctrl
            for i in np.arange(0, model.nu):
                self.tau_ZMP[i] = self.Kp[i] * (dq_err[i + 6] * model.opt.timestep) + self.Kv[i] * (dq_err[i + 6])  

            self.dqdes = self.dqdes + dq_err
            self.qdes = self.qdes + dq_err * model.opt.timestep

        return self.tau_ZMP

    def sim(self, model, data, trn, simfreq, simend, saveVid=False):
        self.mj2humn(model, data)
        mocap_id_COP = model.body("COP").mocapid[0]
        mocap_id_COM_des = model.body("COM_des").mocapid[0]
        ActData = []
        DesData = []
        
        if saveVid:
            renderer = mj.Renderer(model, width=1280, height=720)
            frames = []
            framerate = int(simfreq/4) 

        with mj.viewer.launch_passive(model=model, data=data, show_left_ui=True, show_right_ui=False) as viewer:
            mj.mjv_defaultFreeCamera(model, viewer.cam)
            viewer.opt.flags[16] = 1 
            viewer.opt.flags[18] = 1 

            if saveVid:
                camera = mj.MjvCamera()
                mj.mjv_defaultFreeCamera(model, camera)
                camera.distance = self.cam_dist
                scene_option = mj.MjvOption()
                scene_option.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = True

            print("Simulation starting.....")
            time.sleep(2)

            while viewer.is_running() and data.time < simend:
                time_prev = data.time
                clock_start = time.time()
                
                while (data.time - time_prev < 1.0 / simfreq) and data.time < simend:
                    data.ctrl = self.controller(model, data).copy()
                    mj.mj_step(model, data)  
                    for i in range(model.nu):
                        self.WD += abs(data.actuator_force[i] * data.qvel[i+6] * model.opt.timestep)

                if data.time >= simend:
                    break

                self.mj2humn(model, data)
                self.mjfc(model, data)
                
                if abs(self.fcn) > 0:
                    rcop = self.rf / abs(self.fcn)
                    data.mocap_pos[mocap_id_COP, 0:3] = rcop
                else:
                    data.mocap_pos[mocap_id_COP, 0:3] = np.array([1000, 1000, 1000])

                sipCnt = np.array([self.oCPx(data.time), self.oCPy(data.time), self.oCPz(data.time)])
                sipCOM = np.array([self.oCMx(data.time), self.oCMy(data.time), self.oCMz(data.time)])
                data.mocap_pos[mocap_id_COM_des, 0:3] = sipCOM
                
                idx_geom = 0
                for i in range(100):
                    sipPt = sipCOM + i / 100 * (sipCnt - sipCOM)
                    mujoco.mjv_initGeom(viewer.user_scn.geoms[idx_geom], type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.005, 0, 0], pos=sipPt, mat=np.eye(3).flatten(), rgba=np.array([1, 0, 0, 0.3]))
                    idx_geom += 1
                    viewer.user_scn.ngeom = idx_geom
                    if idx_geom > (viewer.user_scn.maxgeom - 50):
                        idx_geom = 1

                self.updateTrajData(model, data)
                viewer.sync()
                
                if saveVid:
                    camera.lookat = self.r_com
                    renderer.update_scene(data, camera, scene_option)
                    geom = renderer.scene.geoms[renderer.scene.ngeom]
                    mujoco.mjv_initGeom(geom, type=mujoco.mjtGeom.mjGEOM_SPHERE, size=np.array([0.0001, 0.0001, 0.0001]), pos=self.r_com + 0.5*self.r_com[2]*np.array([2.0, 0.0, 1.0]), mat=np.eye(3).flatten(), rgba=np.array([1, 0, 0, 1]))
                    geom.label = "0.25 x real-time"
                    renderer.scene.ngeom += 1
                    pixels = renderer.render()
                    frames.append(pixels)

                time_until_next_step = 1 / simfreq - (time.time() - clock_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

        if saveVid:
            clip = ImageSequenceClip(frames, fps=framerate)
            clip.write_gif("simulation_video.gif")

        return DesData, ActData

# inverse kinematics
def numik(model, data, q0, delt, ocm, oleft, oright, ubjnts, gradH0, WLN, k_ub, zeroAM):
    data = q2data(data, q0)
    mj.mj_fwdPosition(model, data)
    ocmi = data.subtree_com[0].copy() 

    olefti = data.site('left_foot_site').xpos.copy() 
    orighti = data.site('right_foot_site').xpos.copy() 

    quat_hip = data.qpos[3:7].copy()
    quat_conj = np.zeros([4])
    err_quat = np.zeros([4])
    err_ori_hip = np.zeros([3])
    err_ori_left = np.zeros([3])
    err_ori_right = np.zeros([3])
    quat_left = np.zeros([4])
    quat_right = np.zeros([4])

    mujoco.mju_negQuat(quat_conj, quat_hip)
    mujoco.mju_mulQuat(err_quat, np.array([1, 0, 0, 0]), quat_conj)
    mujoco.mju_quat2Vel(err_ori_hip, err_quat, 1.0)

    if model.nu - len(ubjnts) >= 6:
        mujoco.mju_mat2Quat(quat_left, data.site('left_foot_site').xmat)
        mujoco.mju_negQuat(quat_conj, quat_left)
        mujoco.mju_mulQuat(err_quat, np.array([1, 0, 0, 0]), quat_conj)
        mujoco.mju_quat2Vel(err_ori_left, err_quat, 1.0)

        mujoco.mju_mat2Quat(quat_right, data.site('right_foot_site').xmat)
        mujoco.mju_negQuat(quat_conj, quat_right)
        mujoco.mju_mulQuat(err_quat, np.array([1, 0, 0, 0]), quat_conj)
        mujoco.mju_quat2Vel(err_ori_right, err_quat, 1.0)

    delE = np.linalg.norm(oleft - olefti) + np.linalg.norm(err_ori_left) + np.linalg.norm(oright - orighti) + np.linalg.norm(err_ori_right) + np.linalg.norm(ocm - ocmi)
    k = 1 
    
    while delE > 1e-8:
        Jcm = np.zeros((3, model.nv)) 
        mj.mj_jacSubtreeCom(model, data, Jcm, 0)
        Jwb = np.zeros((3, model.nv))  
        Jwb[0:3, 3:6] = np.eye(3)
        Jvleft = np.zeros((3, model.nv)) 
        Jwleft = np.zeros((3, model.nv))
        mj.mj_jacSite(model, data, Jvleft, Jwleft, model.site('left_foot_site').id)
        Jvright = np.zeros((3, model.nv)) 
        Jwright = np.zeros((3, model.nv))
        mj.mj_jacSite(model, data, Jvright, Jwright, model.site('right_foot_site').id)
        
        if ubjnts.size:
            Jub = np.zeros((len(ubjnts), model.nv))  
            Jub[:, ubjnts] = np.eye(len(ubjnts))
        
        Iwb = np.zeros([3, model.nv])
        mj.mj_angmomMat(model, data, Iwb, 0)

        Avec = np.zeros([18 + len(ubjnts) + 3 + 3, model.nv])
        bvec = np.zeros([18 + len(ubjnts) + 3 + 3])
        
        Avec[0:3, 0:model.nv] = Jcm
        bvec[0:3] = ocm - ocmi
        Avec[3:6, 0:model.nv] = Jwb
        bvec[3:6] = err_ori_hip 
        Avec[6:9, 0:model.nv] = Jvleft
        bvec[6:9] = oleft - olefti
        Avec[9:12, 0:model.nv] = Jwleft
        bvec[9:12] = err_ori_left 
        Avec[12:15, 0:model.nv] = Jvright
        bvec[12:15] = oright - orighti
        Avec[15:18, 0:model.nv] = Jwright
        bvec[15:18] = err_ori_right 

        invWroot = np.eye(model.nv) 
        if ubjnts.size:
            qref = 0 * np.ones([model.nv])            
            Avec[18:18 + len(ubjnts), 0:model.nv] = Jub
            bvec[18:18 + len(ubjnts)] = (qref[ubjnts] - q0[ubjnts]) / 1 
            Avec[18 + len(ubjnts):18 + len(ubjnts) + 3, 0:model.nv] = Iwb
            bvec[18 + len(ubjnts):18 + len(ubjnts) + 3] = np.zeros([3])
        else:
            Avec[18:18 + 3, 0:model.nv] = Iwb
            bvec[18:18 + 3] = np.zeros([3])

        if (model.nv - len(ubjnts)) < 18:
            eqnJ1 = np.append(np.array([0, 2]), np.array([6, 8, 10, 12, 14, 16])) 
        else:
            eqnJ1 = np.append(np.array([0, 1, 2]), np.arange(6, 18))

        J1 = Avec[eqnJ1, :].copy()
        delx1 = bvec[eqnJ1].copy() / delt
        dqN1 = np.matmul(np.linalg.pinv(J1), delx1)
        InJ1 = np.eye(model.nv) - np.matmul(np.linalg.pinv(J1), J1)

        eqnJ2 = np.append(np.array([3, 4, 5]), np.arange(18, 18 + len(ubjnts)))
        if not WLN:
            invWroot = np.eye(model.nv)
            
        J2 = Avec[eqnJ2, :] @ invWroot 
        delx2 = bvec[eqnJ2].copy() / delt
        Jt2 = np.matmul(J2, InJ1)
        dqN2 = dqN1 + invWroot @ np.matmul(np.linalg.pinv(Jt2), delx2 - np.matmul(J2, dqN1))
        dq = dqN2.copy() 

        if delt < 1:
            data = q2data(data, q0)
            q = data.qpos.copy()  
            mujoco.mj_integratePos(model, q, dq, delt)

            q0 = 0 * data.qvel.copy()
            qqt = q[3:7].copy()
            qeulr = quat2euler(qqt)
            for i in np.arange(0, 3): q0[i] = q[i].copy()
            for i in np.arange(3, 6): q0[i] = qeulr[i - 3].copy()
            for i in np.arange(6, len(data.qvel)): q0[i] = q[i + 1].copy()
            return q0, gradH0

        while delE <= (np.linalg.norm(oleft - olefti) + np.linalg.norm(err_ori_left) + np.linalg.norm(oright - orighti) + np.linalg.norm(err_ori_right) + np.linalg.norm(ocm - ocmi)):
            data = q2data(data, q0)
            q = data.qpos.copy()  
            mujoco.mj_integratePos(model, q, dq, delt / k)

            qi = 0 * data.qvel.copy()
            qqt = q[3:7].copy()
            qeulr = quat2euler(qqt)
            for i in np.arange(0, 3): qi[i] = q[i].copy()
            for i in np.arange(3, 6): qi[i] = qeulr[i - 3].copy()
            for i in np.arange(6, len(data.qvel)): qi[i] = q[i + 1].copy()

            data = q2data(data, qi)
            mj.mj_fwdPosition(model, data)
            ocmi = data.subtree_com[0]
            olefti = data.site('left_foot_site').xpos.copy()  
            orighti = data.site('right_foot_site').xpos.copy()  
            
            if model.nu - len(ubjnts) >= 60:
                mujoco.mju_mat2Quat(quat_left, data.site('left_foot_site').xmat)
                mujoco.mju_negQuat(quat_conj, quat_left)
                mujoco.mju_mulQuat(err_quat, np.array([1, 0, 0, 0]), quat_conj)
                mujoco.mju_quat2Vel(err_ori_left, err_quat, 1.0)
                mujoco.mju_mat2Quat(quat_right, data.site('right_foot_site').xmat)
                mujoco.mju_negQuat(quat_conj, quat_right)
                mujoco.mju_mulQuat(err_quat, np.array([1, 0, 0, 0]), quat_conj)
                mujoco.mju_quat2Vel(err_ori_right, err_quat, 1.0)

            k = 2 * k
            if k > 2:
                break

        if delE > (np.linalg.norm(oleft - olefti) + np.linalg.norm(err_ori_left) + np.linalg.norm(oright - orighti) + np.linalg.norm(err_ori_right) + np.linalg.norm(ocm - ocmi)):
            delE = np.linalg.norm(oleft - olefti) + np.linalg.norm(err_ori_left) + np.linalg.norm(oright - orighti) + np.linalg.norm(err_ori_right) + np.linalg.norm(ocm - ocmi)
            q0 = qi.copy()
            k = 1

    return q0, gradH0

def tauinvd(model, data, ddqdes):
    data.qacc = ddqdes.copy()
    mj.mj_inverse(model, data)
    tau = data.qfrc_inverse[6:].copy()
    return tau

#tính toán lực
class trnparam:
    def __init__(self, nocp, zeta, zpln):
        self.zeta = zeta
        self.zpln = zpln
        self.nocp = nocp
        self.r, self.rdot, self.aref, self.f, self.efc_f, self.q, self.dq, self.ddq = [], [], [], [], [], [], [], []

    def mjparam(self, model):
        self.solimp, self.solref, self.pos, self.size, self.xmean = [], [], [], [], []
        i = 0
        while model.geom_bodyid[i] == 0:
            self.pos.append(model.geom_pos[i].copy())
            self.size.append(model.geom_size[i].copy())
            solimp = model.geom_solimp[i].copy()
            solref = model.geom_solref[i].copy()
            self.solimp.append(solimp)
            d0, dwidth, width, midpt, power = solimp[0], solimp[1], solimp[2], solimp[3], solimp[4]
            dmean = (d0 + dwidth) / 2
            deln = width * self.nocp
            xmean = deln / 2
            
            if solref[0] < 0:
                dampratio = self.zeta
                stiffness = (9.81 * (1 - dmean) * dwidth * dwidth) / (xmean * dmean * dmean) 
                timeconst = 1 / (dampratio * np.sqrt(stiffness))
            else:
                timeconst = solref[0] 
                dampratio = solref[1]
                stiffness = 1 / ((timeconst**2) * (dampratio**2))
                
            damping = 2 / timeconst 
            i += 1
            self.solref.append([-stiffness, -damping])
            self.xmean.append(xmean)

    def cntplane(self, cntpt, spno):
        i = 0
        for pos in self.pos:
            size = self.size[i].copy()
            if cntpt[0] > (pos[0] - size[0]) and cntpt[0] < (pos[0] + size[0]):
                if cntpt[1] > (pos[1] - size[1]) and cntpt[1] < (pos[1] + size[1]):
                    self.cntgeomid = i
                    self.cntpos = self.pos[i].copy()
                    self.cntsize = self.size[i].copy()
                    if spno == 1:
                        self.cntsolref = self.solref[i].copy()
                        self.cntsolimp = self.solimp[i].copy()
                        self.cntnocp = self.nocp
                        self.cntpos[2] += self.cntsize[2]
                    else:
                        self.cntsolref = [0.02, 1] 
                        self.cntsolimp = [0.9, 0.95, 0.001, 0.5, 2] 
                        self.cntnocp = 2 * self.nocp
                        self.cntpos[2] = cntpt[2] + 0.00036 
                    break
            i += 1

def DepthvsForce(model, data, plotdata):
    weight = model.body_subtreemass[1] * np.linalg.norm(model.opt.gravity)
    mujoco.mj_inverse(model, data)
    
    if data.ncon: 
        dz = 0.000001
    else: 
        dz = -0.000001
        
    while True:
        data.qpos[2] += dz
        mujoco.mj_inverse(model, data)
        if (dz > 0) * (data.ncon == 0) or (dz < 0) * (data.ncon > 0):
            z_0 = data.qpos[2]
            break
            
    height_arr = np.linspace(z_0 - 0.025, z_0, 101)
    vertical_forces = []
    
    for z in height_arr:
        data.qpos[2] = z
        mujoco.mj_inverse(model, data)
        vertical_forces.append(data.qfrc_inverse[2])

    height_offsets = height_arr - z_0
    vertical_forces = np.array(vertical_forces)
    
    idx = np.argmin(np.abs(vertical_forces))
    best_offset = height_offsets[idx]

    return best_offset