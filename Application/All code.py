
#Dictionary of all cities plotted on the map
city_list0 = [

    'vienna',
    'tokyo',
    'london',
    'birmingham',
    'los angeles',
    'mapai',
    'seoul',
    'moscow',
    'buenos aires',
    'shanghai',
    'sydney'

]

#extracts coordinates from file
def city_dict():
    with open('world_cities.csv', 'r') as f:
         lines = f.readlines()

    header = lines[0]

    cities = {}


    for line in lines[1:]:
        line = line.split(',')

        try:
            cities[line[1]] = [float(line[2]), float(line[3])]
        except:
            pass

    return cities

from tkinter import *
import tkinter as tk
from tkinter import ttk

from pil import image as pim, imagetk
import customtkinter
import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.integrate import solve_ivp
import sympy as smp


class Page(tk.frame):

    def __init__(self,*args, **kwargs):
        tk.frame.__init__(self, *args, **kwargs)

#showing a page
    def show(self):
        self.lift()



class Page1(page):


    def __init__(self, *args, **kwargs):

        page.__init__(self, *args, **kwargs)

        img = (pim.open("satellite1.png"))
        self.resized_image = img.resize((853, 480), pim.antialias)
        self.new_image = imagetk.photoimage(self.resized_image)
        self.newi = label(self, image=self.new_image,
                          highlightthickness=0, bd=0,
                          relief='ridge')
        self.newi.place(x=0, y=0)
        self.command = none
        self.labe1()

        self.button1 = customtkinter.ctkbutton(self, text='start', width=200,
                                               height=20, border_width=0, bg_color='#00468c',
                                               command= lambda: self.command)
        self.button1.place(x=1000, y=1000)
        self.button1.configure(bg_color = 'white')

        self.title = label(self, text = 'programs', width = 7, height = 2, font = ('modern',40))
        self.title.place(x=100, y=40)
        self.option = none
        self.values = []


#creates window to show groundtrack plotting and 3D plot
    def opennewwindow3(self):

        options = [
            "geostationary orbit",
            "polar orbit",
            "40 degree inclination"
        ]

        options2 = [
            "geostationary transfer orbit",
            "geostationary orbit",
            "polar orbit "
        ]
#new window initialised
        newwindow = toplevel(root)
        newwindow.title = ("new window 1")
        newwindow.geometry("665x335")
        newwindow.configure(background='#008b8b')


        l1 = label(newwindow, bg='light grey', width=35, height=20).place(x=5, y=5)
        l1 = label(newwindow, bg='light grey', width=35, height=20).place(x=340, y=5)
        header = label(newwindow, bg ='light grey', text='groundtrack', width=10, height=2, font=('helvetica', 25)).place(x=100,y=40)
        header1 = label(newwindow, bg ='light grey', text='3d graph plot', width=15, height=2, font=('helvetica', 25)).place(x=400,y=40)

#runs sepearate python scripts based on user selection
        def run():

            if clicked.get() == options[0]:
                os.system('python nea3.py')

            elif clicked.get() == options[1]:
                os.system('python nea4.py')

            elif clicked.get() == options[2]:
                os.system('python nea5.py')

        def run1():
            os.system('python nea6.py')


        def run2():
            os.system('python nea7.py')

        def run3():
            os.system('python nea8.py')


#Initialising radio buttons
        clicked = stringvar()
        clicked.set(none)


        clicked.trace('w',self) #Can used to trace user selectiom


        drop = optionmenu(newwindow, clicked, *options) #dropmenu button object created

        drop.config(width = 15, bg = 'light grey')

#separate buttons created
        my_buttont = customtkinter.ctkbutton(newwindow, text='run', width=50,
                                height=20, border_width=0, bg_color='light grey',
                                command=run)


        my_buttont.place(x=150, y=210)


        my_button0 = customtkinter.ctkbutton(newwindow, text='geo transfer orbit', width=150,
                                            height=20, border_width=0, bg_color='light grey',
                                            command=run1)
        my_button0.pack()
        my_button0.place(x=430, y=140)

        my_button1 = customtkinter.ctkbutton(newwindow, text='geostationary orbit', width=150,
                                            height=20, border_width=0, bg_color='light grey',
                                            command=run2)
        my_button1.pack()
        my_button1.place(x=430, y=170)

        my_button2 = customtkinter.ctkbutton(newwindow, text='polar orbit', width=150,
                                            height=20, border_width=0, bg_color='light grey',
                                            command=run3)
        my_button2.pack()
        my_button2.place(x=430, y=200)


        drop.pack()
        drop.place(x=90, y=150)

        root.mainloop()

#creates window for the rocket equation
    def opennewwindow2(self):

        newwindow = toplevel(root)
        newwindow.title = ("new window 2")
        newwindow.geometry("300x300")

        x1 = label(newwindow, text="enter launch time:")
        x1.pack()
        x1.place(x=20,y=23)

        x2 = label(newwindow, text="enter fuel emission:")
        x2.pack()
        x2.place(x=20, y=83)

        x3 = label(newwindow, text="enter rocket mass:")
        x3.pack()
        x3.place(x=20, y=143)

        x4 = label(newwindow, text="enter ar coefficient:")
        x4.pack()
        x4.place(x=20, y=203)
#textboxes for user to enter values
        lt = entry(newwindow, width = 10, borderwidth = 5)
        lt.pack()
        lt.place(x=170,y=20)

        rfe = entry(newwindow, width = 10, borderwidth = 5)
        rfe.pack()
        rfe.place(x=170,y=80)

        rm = entry(newwindow, width=10, borderwidth=5)
        rm.pack()
        rm.place(x=170, y=140)

        ca = entry(newwindow, width=10, borderwidth=5)
        ca.pack()
        ca.place(x=170, y=200)

#Start button to run program
        self.button4 = customtkinter.ctkbutton(newwindow, text='confirm', width=200,
                                               height=20, border_width=0, bg_color='#00468c',
                                               command=lambda:
                                               self.rocketequation(float(lt.get()),float(rfe.get()), float(rm.get()),float(ca.get())))
        self.button4.place(x=50, y=250)
        self.button4.configure(bg_color='white')


#runs 3D 2-body simulation
    def opennewwindow1(self):

        os.system('python nea.py')

#creates image
    def labe2(self,r):

        img = (pim.open("file/ground-track.png"))
        resized_image = img.resize((400, 400), pim.antialias)
        new_image = imagetk.photoimage(resized_image)
        newi = label(r, image=new_image,
                     highlightthickness=0, bd=0,
                     relief='ridge')
        newi.place(x=0, y=0)



    def viewselected(self,var):
#Conditions for selected values of the radiobutton
        if var.get() == 1:

            self.command = self.opennewwindow1 #Command the start button runs
            self.button1.command = self.command

        elif var.get() == 2:

            self.command = self.opennewwindow2
            self.button1.command = self.command

        elif var.get() == 3:

            self.command = self.opennewwindow3
            self.button1.command = self.command

    def labe1(self):

        self.block = label(self, bg = 'white', width = 30, height = 20)
        self.block.place(x = 45, y = 40)


    def rocketequation(self,t,v,m,b):

        self.t0 = t  # launch time
        g = 9.81
        self.vg = v  # rate of fuel emission
        self.m0 = m  # rocket mass
        self.b = b  # coefficient of air resistance

        m0 = self.m0 / (self.b * self.vg * self.t0)
        m0 = np.inf

        g * self.t0 / self.vg #parameters
        self.m0 / (self.b * self.vg * self.t0)

        t, n = smp.symbols('t n')
        z = 1 - (9 / 10) * t ** n

        t, n = smp.symbols('t n')
        z = 1 - (9 / 10) * t ** n
        #differentiates parameters P2
        dzdt = smp.diff(z, t).simplify()
        z = smp.lambdify([t, n], z)
        dzdt = smp.lambdify([t, n], dzdt)

        t = np.linspace(1e-4, 1, 1000)
        z1 = z(t, 1)
        z2 = z(t, 0.7)
        z3 = z(t, 5)
#sets up plot for different fuel usages
        plt.figure(figsize=(6, 3))
        plt.plot(t, z1, label='m1')
        plt.plot(t, z2, label='m2')
        plt.plot(t, z3, label='m3')
        plt.ylabel('$m(t)/m_0$')
        plt.xlabel('$t/t_0$')
        plt.legend()
#Differential equation for rocket displacement
        def dsdt(t, s, m0, vg, n, t0=40):
            x, v = s[0], s[1]
            if t < 1:
                dxdt = v
                dvdt = -g * t0 / vg - 1 / (m0 * z(t, n)) * v ** 2 * np.sign(v) - 1 / z(t, n) * dzdt(t, n)
            else:
                dxdt = v
                dvdt = -g * t0 / vg - 1 / (m0 * z(1, n)) * v ** 2 * np.sign(v)
            if (dvdt < 0) * (dxdt < 0) * (x <= 0):
                dxdt = 0
                dvdt = 0
            return [dxdt, dvdt]

        n1, n2, n3 = 1, 0.7, 5.05 #different fuel emission rate
        sol1 = solve_ivp(dsdt, [1e-4, 3], y0=[0, 0], t_eval=np.linspace(1e-4, 3, 1000), args=(m0, self.vg, n1, self.t0))
        sol2 = solve_ivp(dsdt, [1e-4, 3], y0=[0, 0], t_eval=np.linspace(1e-4, 3, 1000), args=(m0, self.vg, n2, self.t0))
        sol3 = solve_ivp(dsdt, [1e-4, 3], y0=[0, 0], t_eval=np.linspace(1e-4, 3, 1000), args=(m0, self.vg, n3, self.t0))
#plot configuration
        fig, axes = plt.subplots(1, 2, figsize=(10, 3))
        ax = axes[0]
        ax.plot(sol1.t, sol1.y[0], label='n={}'.format(n1))
        ax.plot(sol2.t, sol2.y[0], label='n={}'.format(n2))
        ax.plot(sol3.t, sol3.y[0], label='n={}'.format(n3))
        ax.axvline(1, ls='--', color='k')
        ax.set_ylabel('$x(t)/(v_g t_0)$')
        ax.set_title('position')
        ax.legend()
        ax = axes[1]
        ax.plot(sol1.t, sol1.y[1], label='n=1')
        ax.plot(sol2.t, sol2.y[1], label='n=0.7')
        ax.plot(sol3.t, sol3.y[1], label='n=1.3')
        ax.axvline(1, ls='--', color='k')
        ax.set_ylabel('$v(t)/v_g$')
        ax.set_title('velocity')
        fig.text(0.5, -0.04, '$t/t_0$', ha='center', fontsize=20)
        fig.tight_layout()
        plt.show()

#Main program
class Main(tk.frame):

    def __init__(self, *args, **kwargs):
        tk.frame.__init__(self, *args, **kwargs)
#Defines page for transition
        self.page1 = page1(self)


        buttonframe = tk.frame(self)
        container = tk.frame(self)
        buttonframe.pack(side="top", fill="x", expand=false)
        container.pack(side="top", fill="both", expand=true)

        self.radiobuttons()
        self.page1.place(in_=container, x=0, y=0, relwidth=1, relheight=1)
        self.labe1()

        customtkinter.set_appearance_mode("system")

        customtkinter.set_default_color_theme("blue")

        self.button1 = customtkinter.ctkbutton(self, text='start', width=200,
                                              height=20, border_width = 0, bg_color= '#00468c',
                                              command=lambda: self.page1.show()
                                              )
        self.button1.place(x=330, y=190)

        customtkinter.set_default_color_theme("blue")

        self.button2  = customtkinter.ctkbutton(self, text='description',width=200,
                                              height=20, border_width = 0, bg_color= '#00468c',
                                              command= self.newwindow1
                                              )
        self.button2.place(x=330, y=220)

        customtkinter.set_default_color_theme("blue")

        self.button4 = customtkinter.ctkbutton(self, text='exit', width=200,
                                               height=20, border_width=0, bg_color='#00468c',
                                                command= self.quit
                                                )
        self.button4.place(x=330, y=250)

        self.im = pim.open("file/orbit.png")
        self.ph = imagetk.photoimage(self.im)
#Sets up GIF
        self.framecnt = 7
        self.frames = [photoimage(file='file/orbit.png', format='gif -index %i' % (i)) for i in range(self.framecnt)]


    def newwindow1(self):

        new = toplevel(root)
        framecnt = 79
        frames = [photoimage(file='satellite-circling-the-earth-animated-clipart.gif', format='gif -index %i' % (i)) for i in range(framecnt)]
        #Updates frame for GIF
        def update(ind):
            frame = frames[ind]
            ind += 1
            if ind == framecnt:
                ind = 0
            label.configure(image=frame)
            new.after(100, update, ind)

        label = label(new)
        label.pack()
        new.after(0, update, 0)
#Description label
        label2 = label(new, text = 'this is a educational program which consists of 3 main sections. the first section \n'
                                   'is a simulation of 2 bodys for a understanding on a how an object with large mass\n'
                                   ' affects another object of miniscule mass. the second is a plot program that presents \n '
                                   ' optimum fuel consumption when launching a rocket. the final is a plot of a satellites \n'
                                   'groundtrack for different orbits',
                       height = 5)
        label2.pack()
        new.mainloop()




    def updategif(self, f):
        frame = self.frames[f]
        f += 1
        if f == self.framecnt:
            f = 0
        self.label.configure(image=frame)
        self.label.place(x=0, y=0)
        self.page1.after(100, self.updategif, f)

    def labe1(self):

        self.backgroundimage = imagetk.photoimage(file ="milky-way.jpeg")
        self.backgroundimagelabe1 = label(self, image = self.backgroundimage,
                                          highlightthickness=0, bd = 0,
                                          relief = 'ridge')
        self.backgroundimagelabe1.place(x=0,y=0)

        title = (pim.open("title.png"))


    def radiobuttons(self):
            var = intvar()

            values = {
                '3d model': 1,
                'rocket equation graphing': 2,
                'groundtrack/orbit plotting': 3,
            }
            h = 160
            for i in range(3):
                h += 25
                self.r = tk.radiobutton(self.page1, text=list(values)[i], value=values[list(values)[i]], variable = var,
                                     command=lambda: [self.page1.button1.place(x=82,y=310), self.page1.viewselected(var)]).place(x=90, y=h)


if __name__ == "__main__":
    root = tk.tk()
    main = main(root)
    p1 = page1(root)
    option1 = p1.option
    print(option1)
    main.pack(side="top", fill="both", expand=true)
    root.geometry("853x480")
    root.resizable(true, true)
    root.mainloop()

from ursina import*
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import time
from ursina.prefabs import trail_renderer
import numpy as np
import math
import sympy as smp
from scipy.spatial.transform import rotation as r
vec = [1,1,1]

rotation_degrees = 90
rotation_radians = np.radians(rotation_degrees)
rotation_axis = np.array([0, 0, 1])

rotation_vector = rotation_radians * rotation_axis
rotation = r.from_rotvec(rotation_vector)
rotated_vec = rotation.apply(vec)


app = ursina()
window.title = 'satellite prototype'
window.borderless = false
window.exit_button.visible = false
window.fps_counter.enabled = false

t = 0
momentum1 = [0,0,0]
momentum2 = [0,30,0]

au = (149.6e6 * 1000)
scale = 250 / au
dt = 3600 * 24


class Object(entity):

    def __init__(self,mass,radius,model,texture,_position,velocity,scale):
        super().__init__(model=model, texture=texture)
        self.scale = scale
        self.mass = mass
        self.radius = radius
        self._position = _position
        self._origin = [0,0,0]
        self.velocity = velocity
        self._acceleration = [[0.0, 0.0, 0.0]]
        self._vnew = [0.0, 0.0, 0.0]
        self._partialpos = [0.0, 0.0, 0.0]
        self.g = 1
        self.static = False

    def update_partial_pos(self, index, dt):
        self._partialpos[0] = self._position[index][0] + self._velocity[index][0] * dt / 2.0
        self._partialpos[1] = self._position[index][1] + self._velocity[index][1] * dt / 2.0
        self._partialpos[2] = self._position[index][2] + self._velocity[index][2] * dt / 2.0

        self._vnew = self._velocity[index].copy()


    def gravitational_force(self):
        g = 1
        r_vec = np.subtract(self._position,self._origin)
        self.r_mag = np.linalg.norm(r_vec)

        f = (g *self.mass*10000) / np.power(self.r_mag, 2)


        return f

    def v_vec(self):

        theta = 0
        theta_vec = vec3(-math.sin(theta),0,math.cos(theta))
        print(theta_vec)
        d0dt = 0.021
        v_vec = self.r_mag * d0dt * theta_vec
        theta += d0dt
        print(v_vec)
        return v_vec

e_scale = 5.972e24/6500
s_scale = 6500/5.972e24
e = object(mass = 5.972e24, radius = 695508e3, model='sphere',texture="earth-map", _position = [0,0,0],velocity= [0,0,0],scale = e_scale)
e.position = e._position
e.scale = 10
e.static = false

g = 6.67e-11

v2 = g*e.mass/(e.radius+35786e3)
v = math.sqrt(v2)/3000
print(v)
s = object(mass = 6500, radius = 2439.4e3, model='sphere', texture ="texty", _position = [100,0,0],velocity= [0,0,v],scale = s_scale)
s.position = s.position
s.staic = false
s.scale = s.radius/e.radius*100
s._maxpoint = 366


class Trailrenderer(entity):
    def __init__(self, thickness=10, color=color.white, end_color=color.clear, length=500, **kwargs):
        super().__init__(**kwargs)
        self.renderer = entity(
            model = mesh(
            vertices=[self.world_position for i in range(length)],
            colors=[lerp(end_color, color, i/length*2) for i in range(length)],
            mode='line',
            thickness=thickness,
            static=false
            )
        )
        self._t = 0
        self.update_step = .025msi


    def update(self):
        self._t += time.dt
        if self._t >= self.update_step:
            self._t = 0
            self.renderer.model.vertices.pop(0)
            self.renderer.model.vertices.append(self.world_position)
            self.renderer.model.generate()

    def on_destroy(self):
        destroy(self.renderer)




if __name__ == '__main__':
    player = s
    trail_renderer = trailrenderer(parent=player, thickness=100, color=color.yellow, length=500)

    pivot = entity(parent=s)
    trail_renderer = trailrenderer(parent=pivot, x=.1, thickness=20, color=color.orange)
    trail_renderer = trailrenderer(parent=pivot, y=1, thickness=20, color=color.orange)
    trail_renderer = trailrenderer(parent=pivot, thickness=2, color=color.orange, alpha=.5, position=(.4,.8))
    trail_renderer = trailrenderer(parent=pivot, thickness=2, color=color.orange, alpha=.5, position=(-.5,.7))

    def update():
        player.position = lerp(player.position, mouse.position*10, time.dt*4)

        if pivot:
            pivot.rotation_z -= 3
            pivot.rotation_x -= 2

    def input(key):
        if key == 'space':
            destroy(pivot)




def update():

    global t

    r_vec = np.subtract(s._position, e._position)
    distance = np.linalg.norm(r_vec)

    direction_vector = [s._position[0] - e._position[0], s._position[1] - e._position[1],
                            s._position[2] - e._position[2]]

    a = s.gravitationalforce()

    xv = (a / s.mass * time.dt / 20) / distance * -direction_vector[0]
    yv = (a / s.mass * time.dt / 20) / distance * -direction_vector[1]
    zv = (a / s.mass * time.dt / 20) / distance * -direction_vector[2]

    s.velocity[0] += xv
    s.velocity[1] += yv
    s.velocity[2] += zv

    new_position = [s._position[0]+s.velocity[0], s._position[1]+s.velocity[1], s._position[2]+s.velocity[2]]
    s._position = new_position

    s.position = new_position
    print(s._position)



speed = 5.1
sky(texture = 'cry.png')
index = 0
pause = false
editorcamera(x=0,y=0,z=-50)
app.run()


from sys import path
path.append('/users/zou/pycharmprojects/tools.py')
import numpy as np
import matplotlib.pyplot as plt
import spiceypy as spice

import cities_lat_long

from cities_lat_long import city_list0
from orbitpropagator import orbitpropagator as op

tspan = 3600*24*3.0
dt = 100.0

earth_radius = 6378.0
date0 = '2020-08-29'
frame = 'eclipj2000'

import os
cwd = os.getcwd()
print(cwd)

def groundtracks( coords, labels, city_names = [],cs = ['c2', 'c3', 'b', 'g', 'c1'],
                  show_plot = false, save_plot = false, filename = 'groundtracks.png', dpi = none ):

    plt.figure( figsize = (16,8))

    coast_coords = np.genfromtxt('coastlines.csv', delimiter=',')

    plt.plot(coast_coords[:,0], coast_coords[:,1], 'mo', markersize = 0.3)

    cities = cities_lat_long.city_dict()
    n = 0

    for x in range(len(coords)):

         if labels is none:
             label = str(x)
         else:
             label = labels[x]

         plt.plot(coords[x][0,1], coords[x][0,0],cs[x] +'o',label = label)
         plt.plot(coords[x][0:,1],coords[x][:,0], cs[x] + 'o',markersize = 3)


    for city in city_names:
        coords = cities[city]
        plt.plot( [coords[1]], [coords[0]], cs[n%5] + 'o', markersize = 4)

        if n%2 == 0:
            xytext = (0,2)
        else:
            xytext = (0,-8)

        plt.annotate( city, [coords[1], coords[0]],
                      textcoords = 'offset points', xytext = xytext,
                      ha = 'center', color = cs[n%5], fontsize = 'small'
                      )
        n += 1

    plt.grid(linestyle = 'dotted')
    plt.xlim([-180,180])
    plt.ylim([-90,90])
    plt.xlabel( 'longitude (degrees $^\circ$)')
    plt.ylabel( 'latitude (degrees $^\circ$)')
    plt.legend()

    if show_plot:
        plt.show()

    if save_plot:
        plt.savefig(filename, dpi)

if __name__ == '__main__':
    date0 = '2020-08-29'
    spice.furnsh('solar_system.mk')

    coes0 = [earth_radius + 400.0, 1e-6, 40.0, 10.0, 0.0, 0.0]
    coes1 = [earth_radius + 400.0, 1e-6, 90.0, 0.0, 20, 0.0]
    coes2 = [earth_radius + 400.0, 1e-6, 0.0, -10.0, 0.0, 0.0]

    op0 = op(coes0, '1', date0, 90.0, coes = true)
    op1 = op(coes1, '1', date0, 50.0, coes = true)
    op2 = op(coes2,'1', date0, 50.0, coes = true )

    op0.calc_lat_longs()
    op1.calc_lat_longs()
    op2.calc_lat_longs()

    op0.lat_longs[ :, 0].max()
    op1.lat_longs[ :, 0].min()
    op2.lat_longs[ :, 0].max()

    groundtracks([op2.lat_longs], labels=['geostationary orbit'],
                 city_names=city_list0, show_plot=true, filename='v31_coastlines/groundtracks.png', dpi=400)


from sys import path
path.append('/users/zou/pycharmprojects/tools.py')
import numpy as np
import matplotlib.pyplot as plt
import spiceypy as spice

import cities_lat_long

from cities_lat_long import city_list0
from orbitpropagator import orbitpropagator as op

tspan = 3600 *24 *3.0
dt = 100.0

earth_radius = 6378.0
date0 = '2020-08-29'
frame = 'eclipj2000'

import os
cwd = os.getcwd()
print(cwd)

def groundtracks( coords, labels, city_names = [] ,cs = ['c3'],
                  show_plot = false, save_plot = false, filename = 'groundtracks.png', dpi = none ):

    plt.figure( figsize = (16 ,8))

    coast_coords = np.genfromtxt('coastlines.csv', delimiter=',')

    plt.plot(coast_coords[: ,0], coast_coords[: ,1], 'mo', markersize = 0.3)

    cities = cities_lat_long.city_dict()
    n = 0

    for x in range(len(coords)):

        if labels is none:
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
    plt.xlabel('longitude (degrees $^\circ$)')
    plt.ylabel('latitude (degrees $^\circ$)')
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

    op0 = op(coes0, '1', date0, 50.0, coes=true)
    op1 = op(coes1, '1', date0, 50.0, coes=true)
    op2 = op(coes2, '1', date0, 50.0, coes=true)

    op0.calc_lat_longs()
    op1.calc_lat_longs()
    op2.calc_lat_longs()

    op0.lat_longs[:, 0].max()
    op1.lat_longs[:, 0].min()
    op2.lat_longs[:, 0].max()

    for i in range(2):
        groundtracks([op1.lat_longs], labels=['polar orbit'],
                 city_names=city_list0, show_plot=true, filename='v31_coastlines/groundtracks.png', dpi=500)


from sys import path

path.append('/users/zou/pycharmprojects/tools.py')
import numpy as np
import matplotlib.pyplot as plt
import spiceypy as spice

import cities_lat_long

from cities_lat_long import city_list0
from orbitpropagator import orbitpropagator as op

tspan = 3600 * 24 * 3.0
dt = 100.0

earth_radius = 6378.0
date0 = '2020-08-29'
frame = 'eclipj2000'

import os

cwd = os.getcwd()
print(cwd)


def groundtracks(coords, labels, city_names=[], cs=['b'],
                 show_plot=false, save_plot=false, filename='groundtracks.png', dpi=none):
    plt.figure(figsize=(16, 8))

    coast_coords = np.genfromtxt('coastlines.csv', delimiter=',')

    plt.plot(coast_coords[:, 0], coast_coords[:, 1], 'mo', markersize=0.3)

    cities = cities_lat_long.city_dict()
    n = 0

    for x in range(len(coords)):

        if labels is none:
            label = str(x)
        else:
            label = labels[x]

        plt.plot(coords[x][0, 1], coords[x][0, 0], cs[x] + 'o', label=label)
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
    plt.xlabel('longitude (degrees $^\circ$)')
    plt.ylabel('latitude (degrees $^\circ$)')
    plt.legend()

    if show_plot:
        plt.show()

    if save_plot:
        plt.savefig(filename, dpi)


if __name__ == '__main__':
    date0 = '2020-08-29'
    spice.furnsh('solar_system.mk')

    coes0 = [earth_radius + 400.0, 1e-6, 40.0, 10.0, 0.0, 0.0]
    coes1 = [earth_radius + 400.0, 1e-6, 90.0, 0.0, 20, 0.0]
    coes2 = [earth_radius + 400.0, 1e-6, 0.0, -10.0, 0.0, 0.0]

    op0 = op(coes0, '1', date0, 50.0, coes=true)
    op1 = op(coes1, '1', date0, 50.0, coes=true)
    op2 = op(coes2, '1', date0, 50.0, coes=true)

    op0.calc_lat_longs()
    op1.calc_lat_longs()
    op2.calc_lat_longs()

    op0.lat_longs[:, 0].max()
    op1.lat_longs[:, 0].min()
    op2.lat_longs[:, 0].max()

    groundtracks([op0.lat_longs], labels=['40 degrees inclination'],
                 city_names=city_list0, show_plot=true, filename='v31_coastlines/groundtracks.png', dpi=400)


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
from mpl_toolkits.mplot3d import axes3d

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
    ax.plot_surface(_x,_y,_z,cmap = 'blues')

    l = r_plot*2.0
    x,y,z = [[0,0,0],[0,0,0],[0,0,0]]
    u,v,w = [[1,0,0],[0,1,0],[0,0,1]]
    ax.quiver(x,y,z,u,v,w,color = 'k')

    max_value = np.max(np.abs(r))

    ax.set_xlim([-max_value,max_value])
    ax.set_ylim([-max_value, max_value])
    ax.set_zlim([-max_value, max_value])
    ax.set_xlabel('x (km)'); ax.set_ylabel('y (km)'); ax.set_zlabel('z (km)');

    ax.set_aspect('auto')

    plt.legend(['trajectory','starting position'])

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

    #initialise ode solver
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

    r1 = earth_radius + 35785.0
    v2 = np.sqrt(earth_mu/r1) # v = gm/r

    r00 = [r1,0,0]
    v00 = [0,v2,0]

    tspan1 = 10000*60.0

    propagate(r1,v2,r00,v00,tspan1)


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
from mpl_toolkits.mplot3d import axes3d

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
    ax.plot_surface(_x,_y,_z,cmap = 'blues')

    l = r_plot*2.0
    x,y,z = [[0,0,0],[0,0,0],[0,0,0]]
    u,v,w = [[1,0,0],[0,1,0],[0,0,1]]
    ax.quiver(x,y,z,u,v,w,color = 'k')

    max_value = np.max(np.abs(r))

    ax.set_xlim([-max_value,max_value])
    ax.set_ylim([-max_value, max_value])
    ax.set_zlim([-max_value, max_value])
    ax.set_xlabel('x (km)'); ax.set_ylabel('y (km)'); ax.set_zlabel('z (km)');

    ax.set_aspect('auto')

    plt.legend(['trajectory','starting position'])

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

    #initialise ode solver
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

    r1 = earth_radius + 35785.0
    v2 = np.sqrt(earth_mu/r1) # v = gm/r

    r00 = [r1,0,0]
    v00 = [0,v2,0]

    tspan1 = 10000*60.0

    propagate(r1,v2,r00,v00,tspan1)


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
from mpl_toolkits.mplot3d import axes3d

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
    ax.plot_surface(_x,_y,_z,cmap = 'blues')

    l = r_plot*2.0
    x,y,z = [[0,0,0],[0,0,0],[0,0,0]]
    u,v,w = [[1,0,0],[0,1,0],[0,0,1]]
    ax.quiver(x,y,z,u,v,w,color = 'k')

    max_value = np.max(np.abs(r))

    ax.set_xlim([-max_value,max_value])
    ax.set_ylim([-max_value, max_value])
    ax.set_zlim([-max_value, max_value])
    ax.set_xlabel('x (km)'); ax.set_ylabel('y (km)'); ax.set_zlabel('z (km)');

    ax.set_aspect('auto')

    plt.legend(['trajectory','starting position'])

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

    #initialise ode solver
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

    r1 = earth_radius + 35785.0
    v2 = np.sqrt(earth_mu*(2/r1-1/500)) # v = gm/r

    r00 = [r1,0,0]
    v00 = [0,v2,0]

    tspan1 = 10000*60.0

    propagate(r1,v2,r00,v00,tspan1)

import tools as t
import planetary_data as pd
from scipy.integrate import ode
import spiceypy as spice
import numpy as np

def null_parts():
    return {
        'j2':false,
        'aero':false,
        'moon_grav':false,
        'solar_gravity':false
    }

class Orbitpropagator:

    def __init__(self,state0,tspan,date0,frame = 'j2000',deg=true,cb = pd.earth,coes = true):
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


g_meters = 6.67408e-11
g = g_meters*10**-9

sun = {
    'name':'sun',
    'mass':1.989e30,
    'mu':1.32712e11,
    'radius':695700
}

earth = {
    'name':'earth',
    'mass':5.972e24,
    'mu':5.972e24*g,
    'radius':6378.0,
    'j2':-1.082635854e-3
}

import numpy as np
import spiceypy as spice
import math
import planetary_data as pd
import math as m
d2r = np.pi/180
r2d = 180/np.pi


def ecc_anomaly(arr,method,tol=1e-8):
    if method == 'newton':
       me,e = arr
       if me<np.pi/2.0:
          e0=me+e/2.0
       else:
          e0=me-e/2.0

       for n in range(200):
        ratio = (e0 -e*np.sin(e0)-me)/(1-e*np.cos(e0));
        if abs(ratio)<tol:
            if n == 0: return e0
            else: return e0-ratio
        else:
           e1 = e0-ratio
           e0 = e1
        return false

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


def coes2rv(coes,deg = false, mu=pd.earth['mu'] ):
    if deg:
        a,e,i,ta,aop,raan = coes
        i *= d2r
        ta *= d2r
        aop *= d2r
        raan *= d2r
    else:
        a,e,i,ta,aop,raan = coes

    e = ecc_anomaly([ta,e],'tae')

    r_norm = a*(1-e**2)/(1+e*np.cos(ta))

    r_perif = r_norm*np.array([m.cos(ta),m.sin(ta),0])
    v_perif = m.sqrt(mu*a)/r_norm*np.array([-m.sin(e),m.cos(e)*m.sqrt(1-e**2),0])

    perif2eci = np.transpose(eci2perif(raan,aop,i))

    r = np.dot(perif2eci,r_perif)
    v = np.dot(perif2eci, v_perif)

    return r,v


def inert2ecef(rs, tspan, frame = 'j2000', filenames = none):
    steps = rs.shape[0]
    print(tspan)
    cs = np.zeros( (steps, 3 ,3))
    rs_ecef = np.zeros( rs.shape )

    for step in range( steps-1 ):
        cs[ step, :, :] = spice.pxform('j2000','itrf93', tspan[step])

        rs_ecef[step, : ] = np.dot( cs[step, :, :], rs[step, :])

    return rs_ecef, cs


def inert2latlong(rs, tspan, frame = 'j2000', filenames = none):
    steps = rs.shape[0]

    lat_longs = np.zeros( (steps,3))

    rs_ecef, cs = inert2ecef( rs, tspan, frame, filenames)

    for step in range( rs.shape[0]):
        r_norm, lon, lat = spice.reclat( rs_ecef[step, :])
        lat_longs[step, :] = [lat * r2d, lon * r2d, r_norm]

    return lat_longs, rs_ecef, cs

def rv2period( r, v, mu = pd.earth[ 'mu' ] ):

    epsilon = np.linalg.norm(v) ** 2 / 2.0 - mu / np.linalg.norm(r)

    a = -mu / (2.0 * epsilon)

    return 2 * np.pi * math.sqrt(a ** 3 / mu)





























