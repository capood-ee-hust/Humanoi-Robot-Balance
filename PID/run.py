import mujoco as mj
import numpy as np
import matplotlib.pyplot as plt
import os, time
from datactrlpd import selectRobot, q2data, trnparam, DepthvsForce
from scipy.interpolate import CubicSpline


simend = 20.0 # Thời gian mô phỏng
simfreq = 100 # Tần số hiển thị
spno = 2 # Chế độ đứng 2 chân (DSP)

humn, model, data = selectRobot(vel=0.15, step_len=0.25, spno=spno)

# Thông số môi trường
humn.plno = 0 
humn.zpln = 0 
zeta1 = 1.5 
nocp = 4 
trn = trnparam(nocp, zeta1, humn.zpln)
trn.mjparam(model)


for nocp1 in np.arange(nocp, 0, -1/10):
    trn1 = trnparam(nocp1, zeta1, humn.zpln)
    trn1.mjparam(model)

    i = 0
    while model.geom_bodyid[i] == 0:
        model.geom_solimp[i] = trn1.solimp[i]
        model.geom_solref[i] = trn1.solref[i]
        i += 1
    
    
    if abs(DepthvsForce(model, data, 0)) < (model.geom_solimp[0][2]/2):
        print(f'nocp={nocp1:.2f}, stiffness={model.geom_solref[0][0]:.1f}')
        # Sửa số 1 thành số 0 ở đây nữa
        print('Des vs Act Deformation is:', (model.geom_solimp[0][2]/2), DepthvsForce(model, data, 0))
        break


ttraj = np.arange(0, simend, 1/1000)
qtraj = np.tile(humn.q0, (len(ttraj), 1))

humn.qspl = []
for i in np.arange(0, model.nv):
    humn.qspl.append(CubicSpline(ttraj, qtraj[:, i]))

# Thiết lập mục tiêu tĩnh cho thăng bằng
humn.oCMx = lambda t: humn.r_com[0]
humn.oCMy = lambda t: humn.r_com[1]
humn.oCMz = lambda t: humn.r_com[2]
humn.oLx = lambda t: humn.o_left[0]
humn.oLy = lambda t: humn.o_left[1]
humn.oLz = lambda t: humn.o_left[2]
humn.oRx = lambda t: humn.o_right[0]
humn.oRy = lambda t: humn.o_right[1]
humn.oRz = lambda t: humn.o_right[2]
humn.oCPx = lambda t: humn.r_com[0]
humn.oCPy = lambda t: humn.r_com[1]
humn.oCPz = lambda t: 0.0

# Run
data = q2data(data, humn.q0)
print(f"Bắt đầu mô phỏng đứng im trong {simend} giây...")

humn.sim(model, data, trn, simfreq, simend, saveVid=False)


print("Mô phỏng kết thúc.")