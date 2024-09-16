
from sys import path
path.append('/Users/zou/PycharmProjects/tools.py')
import numpy as np
import matplotlib.pyplot as plt
import spiceypy as spice

import cities_lat_long

from cities_lat_long import city_list0
from OrbitPropagator import OrbitPropagator as OP

tspan = 3600 *24 *3.0
dt = 100.0

earth_radius = 6378.0
date0 = '2020-08-29'
FRAME = 'ECLIPJ2000'

import os
cwd = os.getcwd()
print(cwd)

def groundtracks( coords, labels, city_names = [] ,cs = ['C3'],
                  show_plot = False, save_plot = False, filename = 'groundtracks.png', dpi = None ):

    plt.figure( figsize = (16 ,8))

    coast_coords = np.genfromtxt('coastlines.csv', delimiter=',')

    plt.plot(coast_coords[: ,0], coast_coords[: ,1], 'mo', markersize = 0.3)

    cities = cities_lat_long.city_dict()
    n = 0

    for x in range(len(coords)):

        if labels is None:
            label = str(x)
        else:
            label = labels[x]

        plt.plot(coords[x][0 ,1], coords[x][0 ,0] ,cs[x] +'o', label=label)
        plt.plot(coords[x][0:, 1], coords[x][:, 0], cs[x] + 'o', markersize=3)

    for city in city_names:
        coords = cities[city]
        plt.plot([coords[1]], [coords[0]], cs[n % 1] + 'o', markersize=4)

        if n % 2 == 0:
            xytext = (0, 2)
        else:
            xytext = (0, -8)

        plt.annotate(city, [coords[1], coords[0]],
                     textcoords='offset points', xytext=xytext,
                     ha='center', color=cs[n % 1], fontsize='small'
                     )
        n += 1

    plt.grid(linestyle='dotted')
    plt.xlim([-180, 180])
    plt.ylim([-90, 90])
    plt.xlabel('Longitude (degrees $^\circ$)')
    plt.ylabel('Latitude (degrees $^\circ$)')
    plt.legend()

    if show_plot:
        plt.show()

    if save_plot:
        plt.savefig(filename, dpi)


if __name__ == '__main__':
    date0 = '2020-08-29'
    spice.furnsh('solar_system.mk')

    coes0 = [earth_radius + 400.0, 1e-6, 40.0, 10.0, 0.0, 0.0]
    coes1 = [earth_radius + 400.0, 1e-6, 90.0, 0.0, 0.0, 0.0]
    coes2 = [earth_radius + 400.0, 1e-6, 0.0, -10.0, 0.0, 0.0]

    op0 = OP(coes0, '1', date0, 50.0, coes=True)
    op1 = OP(coes1, '1', date0, 50.0, coes=True)
    op2 = OP(coes2, '1', date0, 50.0, coes=True)

    op0.calc_lat_longs()
    op1.calc_lat_longs()
    op2.calc_lat_longs()

    op0.lat_longs[:, 0].max()
    op1.lat_longs[:, 0].min()
    op2.lat_longs[:, 0].max()

    for i in range(2):
        groundtracks([op1.lat_longs], labels=['Polar Orbit'],
                 city_names=city_list0, show_plot=True, filename='v31_coastlines/groundtracks.png', dpi=500)





