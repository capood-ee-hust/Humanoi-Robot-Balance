import mujoco as mj
import numpy as np
import os, time
from mpc import selectRobot, q2data, trnparam, DepthvsForce
from scipy.interpolate import CubicSpline

simend = 20.0 # Thời gian mô phỏng (20 giây)

simfreq = 100 # Tần số hiển thị (100 khung hình/giây)
spno = 2 # Chế độ đứng 2 chân (Double Support Phase)

humn, model, data = selectRobot(vel=0.15, step_len=0.25, spno=spno)

humn.plno = 0 
humn.zpln = 0 
zeta1 = 1.5 
nocp = 4 
trn = trnparam(nocp, zeta1, humn.zpln)
trn.mjparam(model)

for nocp1 in np.arange(nocp, 0, -1/10):
    
    if abs(DepthvsForce(model, data, 0)) < (model.geom_solimp[0][2]/2):
        print(f'nocp={nocp1:.2f}, stiffness={model.geom_solref[0][0]:.1f}')
        print('Des vs Act Deformation is:', (model.geom_solimp[0][2]/2), DepthvsForce(model, data, 0))
        break
ttraj = np.arange(0, simend, 1/1000)
qtraj = np.tile(humn.q0, (len(ttraj), 1))

humn.qspl = []
for i in np.arange(0, model.nv):
    humn.qspl.append(CubicSpline(ttraj, qtraj[:, i]))
    
humn.oCMx = lambda t: humn.r_com[0]
humn.oCMy = lambda t: humn.r_com[1]
humn.oCMz = lambda t: humn.r_com[2]
# ... (tương tự cho oLx, oLy, oLz, oRx, oRy, oRz)
humn.oCPx = lambda t: humn.r_com[0]
humn.oCPy = lambda t: humn.r_com[1]
humn.oCPz = lambda t: 0.0        

data = q2data(data, humn.q0)

humn.sim(model, data, trn, simfreq, simend, saveVid=False)

print("Mô phỏng kết thúc.")
