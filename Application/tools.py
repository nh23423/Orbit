
import numpy as np
import spiceypy as spice
import math
import planetary_data as pd
import math as m
d2r = np.pi/180
r2d = 180/np.pi


def ecc_anomaly(arr,method,tol=1e-8):
    if method == 'newton':
       Me,e = arr
       if Me<np.pi/2.0:
          E0=Me+e/2.0
       else:
          E0=Me-e/2.0

       for n in range(200):
        ratio = (E0 -e*np.sin(E0)-Me)/(1-e*np.cos(E0));
        if abs(ratio)<tol:
            if n == 0: return E0
            else: return E0-ratio
        else:
           E1 = E0-ratio
           E0 = E1
        return False

    elif method == 'tae':

          ta,e = arr
          return 2*m.atan(m.sqrt((1-e)/(1+e))*m.tan(ta/2.0))
    else:
        print("invalid method for e anomaly")


def eci2perif(raan,aop,i):

    row0 = [-m.sin(raan)*m.cos(i)*m.sin(aop)+m.cos(raan)*m.cos(aop),m.cos(raan)*m.cos(i)*m.sin(aop)+m.sin(raan)*m.cos(aop),m.sin(i)*m.sin(aop)]
    row1 = [-m.sin(raan)*m.cos(i)*m.cos(aop)+m.cos(raan)*m.sin(aop),m.cos(raan)*m.cos(i)*m.cos(aop)+m.sin(raan)*m.sin(aop),m.sin(i)*m.cos(aop)]
    row2 = [m.sin(raan)*m.sin(i),-m.cos(raan)*m.sin(i),m.cos(i)]

    return np.array([row0,row1,row2])


def coes2rv(coes,deg = False, mu=pd.earth['mu'] ):
    if deg:
        a,e,i,ta,aop,raan = coes
        i *= d2r
        ta *= d2r
        aop *= d2r
        raan *= d2r
    else:
        a,e,i,ta,aop,raan = coes

    E = ecc_anomaly([ta,e],'tae')

    r_norm = a*(1-e**2)/(1+e*np.cos(ta))

    r_perif = r_norm*np.array([m.cos(ta),m.sin(ta),0])
    v_perif = m.sqrt(mu*a)/r_norm*np.array([-m.sin(E),m.cos(E)*m.sqrt(1-e**2),0])

    perif2eci = np.transpose(eci2perif(raan,aop,i))

    r = np.dot(perif2eci,r_perif)
    v = np.dot(perif2eci, v_perif)

    return r,v


def inert2ecef(rs, tspan, frame = 'J2000', filenames = None):
    steps = rs.shape[0]
    print(tspan)
    Cs = np.zeros( (steps, 3 ,3))
    rs_ecef = np.zeros( rs.shape )

    for step in range( steps-1 ):
        Cs[ step, :, :] = spice.pxform('J2000','ITRF93', tspan[step])

        rs_ecef[step, : ] = np.dot( Cs[step, :, :], rs[step, :])

    return rs_ecef, Cs


def inert2latlong(rs, tspan, frame = 'J2000', filenames = None):
    steps = rs.shape[0]

    lat_longs = np.zeros( (steps,3))

    rs_ecef, Cs = inert2ecef( rs, tspan, frame, filenames)

    for step in range( rs.shape[0]):
        r_norm, lon, lat = spice.reclat( rs_ecef[step, :])
        lat_longs[step, :] = [lat * r2d, lon * r2d, r_norm]

    return lat_longs, rs_ecef, Cs

def rv2period( r, v, mu = pd.earth[ 'mu' ] ):

    epsilon = np.linalg.norm(v) ** 2 / 2.0 - mu / np.linalg.norm(r)

    a = -mu / (2.0 * epsilon)

    return 2 * np.pi * math.sqrt(a ** 3 / mu)