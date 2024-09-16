import tools as t
import planetary_data as pd
from scipy.integrate import ode
import spiceypy as spice
import numpy as np

def null_parts():
    return {
        'J2':False,
        'aero':False,
        'moon_grav':False,
        'solar_gravity':False
    }

class OrbitPropagator:

    def __init__(self,state0,tspan,date0,frame = 'J2000',deg=True,cb = pd.earth,coes = True):
        if coes:
            self.r0, self.v0 = t.coes2rv(state0,deg = deg,mu = cb['mu'])
        else:
            self.r0 = state0[:3]
            self.v0 = state0[3:]

        self.config = {
            'date0':'2020-08-29',
            'tspan': 2.5,
             'dt': 100.0
        }

        self.y0 = self.r0.tolist()+self.v0.tolist()
        self.dt = self.config['dt']
        self.date0 = date0
        self.frame = frame
        self.cb = cb

        self.et0 = spice.utc2et(self.config['date0'])

        self.tspan = np.arange(self.et0,
                               self.et0 + self.config['tspan'] + self.config['dt'],
                               self.config['dt'])

        if type(tspan) == str:
            self.tspan = float(tspan) * t.rv2period(self.r0, self.v0, self.cb['mu'])
        else:
            self.tspan = tspan

        self.n_steps = int(np.ceil(self.tspan/self.dt))

        self.start_time = spice.utc2et(self.date0)

        self.spice_tspan = np.linspace(self.start_time, self.start_time+self.tspan,self.n_steps)

        self.ts = np.zeros((self.n_steps+1,1))
        self.ys = np.zeros((self.n_steps+1,6))
        self.alts = np.zeros((self.n_steps+1))
        self.ts[0] = 0
        self.ys[0,:] = self.y0
        self.step = 0
        self.solver = ode(self.diffy_q)
        self.solver.set_integrator('|soda')
        self.solver.set_initial_value(self.y0,0)

        self.propagate_orbit()

    def propagate_orbit(self):
        while self.solver.successful() and self.step<self.n_steps:
            self.solver.integrate(self.solver.t+100.0)
            self.ts[self.step] = self.solver.t
            self.ys[self.step] = self.solver.y
            self.step += 1

        self.rs = self.ys[:,:3]
        self.vs = self.ys[:,3:]

    def diffy_q(self,t,y):

        rx,ry,rz,vx,vy ,vz = y
        r = np.array([rx,ry,rz])
        v = np.array([vx,vy,vz])

        norm_r = np.linalg.norm(r)

        ax,ay,az = -r*self.cb['mu']/norm_r**3

        return [vx,vy,vz,ax,ay,az]

    def calc_lat_longs(self):
        print(self.spice_tspan)
        self.lat_longs, self.rs_ecef, _ = t.inert2latlong(self.rs, self.spice_tspan, self.frame)



