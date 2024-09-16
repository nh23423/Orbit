
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
from mpl_toolkits.mplot3d import Axes3D

e_radius = 6378.0
e_mu = 398600.0


def plot(r):

    f = plt.figure(figsize=(10,10))
    ax = f.add_subplot(111, projection = '3d')

    ax.plot(r[:,0],r[:,1],r[:,2],'k')
    ax.plot([r[0,0]],[r[0,1]],[r[0,2]],'ko')

    r_plot = e_radius

    _u,_v = np.mgrid[0:2*np.pi:20j,0:np.pi:10j]
    _x = r_plot*np.cos(_u)*np.sin(_v)
    _y = r_plot*np.sin(_u)*np.sin(_v)
    _z = r_plot*np.cos(_v)
    ax.plot_surface(_x,_y,_z,cmap = 'Blues')

    l = r_plot*2.0
    x,y,z = [[0,0,0],[0,0,0],[0,0,0]]
    u,v,w = [[1,0,0],[0,1,0],[0,0,1]]
    ax.quiver(x,y,z,u,v,w,color = 'k')

    max_value = np.max(np.abs(r))

    ax.set_xlim([-max_value,max_value])
    ax.set_ylim([-max_value, max_value])
    ax.set_zlim([-max_value, max_value])
    ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)'); ax.set_zlabel('Z (km)');

    ax.set_aspect('auto')

    plt.legend(['Trajectory','Starting Position'])

    plt.show()

def diff(t,y,mu):

    rx,ry,rz,vx,vy,vz = y
    r = np.array([rx, ry, rz])
    v = np.array([vx, vy, vz])

    mag_r = np.linalg.norm(r)

    ax,ay,az = -r*mu/mag_r**3

    return [vx, vy, vz, ax, ay, az]


def propagate(r,v,r0,v0,tspan):

    r_mag = r
    v_mag = v

    r0 = r0
    v0 = v0

    tspan = tspan

    dt = 100.0

    steps = int(np.ceil(tspan/dt))

    #preallocates memory for effeciency
    ys = np.zeros((steps,6))
    ts = np.zeros((steps,1))

    y0 = r0+v0
    ys[0] = np.array(y0)
    step = 1

    #initialise ODE solver
    solver = ode(diff)
    solver.set_integrator('lsoda') #specific differential equation solver
    solver.set_initial_value(y0,0)
    solver.set_f_params(earth_mu)


    while solver.successful() and step<steps:
        solver.integrate(solver.t+dt)
        ts[step] = solver.t
        ys[step] = solver.y
        step+=1

    rs = ys[:,:3] #position rate extracted

    plot(rs)

earth_radius = 6378.0
earth_mu = 398600.0

if __name__ == "__main__":

    # Geostationary orbit

    # Geostationary transfer orbit

    r = earth_radius + 400.0
    v = np.sqrt(earth_mu / r)  # v = GM/r

    r0 = [r, r * 0.01, r * -0.5]
    v0 = [0, v, v * 0.8]

    tspan = 10000 * 60.0

    propagate(r, v, r0, v0, tspan)

    # polar orbit









