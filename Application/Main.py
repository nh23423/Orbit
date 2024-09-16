
from tkinter import *
import tkinter as tk
from tkinter import ttk

from PIL import Image as PIM, ImageTk
import customtkinter
import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.integrate import solve_ivp
import sympy as smp


class Page(tk.Frame):

    def __init__(self,*args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)


    def show(self):
        self.lift()



class Page1(Page):


    def __init__(self, *args, **kwargs):

        Page.__init__(self, *args, **kwargs)

        img = (PIM.open("satellite1.png"))
        self.resized_image = img.resize((853, 480), PIM.ANTIALIAS)
        self.new_image = ImageTk.PhotoImage(self.resized_image)
        self.newi = Label(self, image=self.new_image,
                          highlightthickness=0, bd=0,
                          relief='ridge')
        self.newi.place(x=0, y=0)
        self.command = None
        self.Labe1()

        self.button1 = customtkinter.CTkButton(self, text='Start', width=200,
                                               height=20, border_width=0, bg_color='#00468c',
                                               command= lambda: self.command)
        self.button1.place(x=1000, y=1000)
        self.button1.configure(bg_color = 'white')

        self.title = Label(self, text = 'Programs', width = 7, height = 2, font = ('Modern',40))
        self.title.place(x=100, y=40)
        self.Option = None
        self.values = []



    def OpenNewwindow3(self):

        options = [
            "Geostationary Orbit",
            "Polar Orbit",
            "40 degree Inclination"
        ]

        options2 = [
            "Geostationary Transfer Orbit",
            "Geostationary Orbit",
            "Polar Orbit "
        ]

        newwindow = Toplevel(root)
        newwindow.title = ("New Window 1")
        newwindow.geometry("665x335")
        newwindow.configure(background='#008B8B')


        l1 = Label(newwindow, bg='light grey', width=35, height=20).place(x=5, y=5)
        l1 = Label(newwindow, bg='light grey', width=35, height=20).place(x=340, y=5)
        header = Label(newwindow, bg ='light grey', text='Groundtrack', width=10, height=2, font=('Helvetica', 25)).place(x=100,y=40)
        header1 = Label(newwindow, bg ='light grey', text='3D graph plot', width=15, height=2, font=('Helvetica', 25)).place(x=400,y=40)

        def run():

            if clicked.get() == options[0]:
                os.system('python NEA3.py')

            elif clicked.get() == options[1]:
                os.system('python NEA4.py')

            elif clicked.get() == options[2]:
                os.system('python NEA5.py')

        def run1():
            os.system('python NEA6.py')


        def run2():
            os.system('python NEA7.py')

        def run3():
            os.system('python NEA8.py')



        clicked = StringVar()
        clicked.set(None)


        clicked.trace('w',self)


        drop = OptionMenu(newwindow, clicked, *options)

        drop.config(width = 15, bg = 'light grey')

        my_buttont = customtkinter.CTkButton(newwindow, text='Run', width=50,
                                height=20, border_width=0, bg_color='light grey',
                                command=run)


        my_buttont.place(x=150, y=210)


        my_button0 = customtkinter.CTkButton(newwindow, text='Geo Transfer Orbit', width=150,
                                            height=20, border_width=0, bg_color='light grey',
                                            command=run1)
        my_button0.pack()
        my_button0.place(x=430, y=140)

        my_button1 = customtkinter.CTkButton(newwindow, text='Geostationary Orbit', width=150,
                                            height=20, border_width=0, bg_color='light grey',
                                            command=run2)
        my_button1.pack()
        my_button1.place(x=430, y=170)

        my_button2 = customtkinter.CTkButton(newwindow, text='Polar Orbit', width=150,
                                            height=20, border_width=0, bg_color='light grey',
                                            command=run3)
        my_button2.pack()
        my_button2.place(x=430, y=200)


        drop.pack()
        drop.place(x=90, y=150)

        root.mainloop()


    def OpenNewwindow2(self):

        newwindow = Toplevel(root)
        newwindow.title = ("New Window 2")
        newwindow.geometry("300x300")

        x1 = Label(newwindow, text="Enter Launch Time:")
        x1.pack()
        x1.place(x=20,y=23)

        x2 = Label(newwindow, text="Enter Fuel Emission:")
        x2.pack()
        x2.place(x=20, y=83)

        x3 = Label(newwindow, text="Enter Rocket Mass:")
        x3.pack()
        x3.place(x=20, y=143)

        x4 = Label(newwindow, text="Enter AR Coefficient:")
        x4.pack()
        x4.place(x=20, y=203)

        LT = Entry(newwindow, width = 10, borderwidth = 5)
        LT.pack()
        LT.place(x=170,y=20)

        RFE = Entry(newwindow, width = 10, borderwidth = 5)
        RFE.pack()
        RFE.place(x=170,y=80)

        RM = Entry(newwindow, width=10, borderwidth=5)
        RM.pack()
        RM.place(x=170, y=140)

        CA = Entry(newwindow, width=10, borderwidth=5)
        CA.pack()
        CA.place(x=170, y=200)


        self.button3 = customtkinter.CTkButton(newwindow, text='Confirm', width=200,
                                               height=20, border_width=0, bg_color='#00468c',
                                               command=lambda:
                                               self.Rocketequation(float(LT.get()),float(RFE.get()), float(RM.get()),float(CA.get())))
        self.button3.place(x=50, y=250)
        self.button3.configure(bg_color='white')


    def OpenNewwindow1(self):

        os.system('python NEA.py')

    def Labe2(self,r):

        img = (PIM.open("File/Ground-track.png"))
        resized_image = img.resize((400, 400), PIM.ANTIALIAS)
        new_image = ImageTk.PhotoImage(resized_image)
        newi = Label(r, image=new_image,
                     highlightthickness=0, bd=0,
                     relief='ridge')
        newi.place(x=0, y=0)



    def ViewSelected(self,var):

        if var.get() == 1:

            self.command = self.OpenNewwindow1
            self.button1.command = self.command

        elif var.get() == 2:

            self.command = self.OpenNewwindow2
            self.button1.command = self.command

        elif var.get() == 3:

            self.command = self.OpenNewwindow3
            self.button1.command = self.command

    def Labe1(self):

        self.block = Label(self, bg = 'white', width = 30, height = 20)
        self.block.place(x = 45, y = 40)

    def Rocketequation(self,T,V,M,B):

        self.T0 = T  # launch time
        g = 9.81
        self.vg = V  # rate of fuel emission
        self.M0 = M  # rocket mass
        self.b = B  # coefficient of air resistance

        m0 = self.M0 / (self.b * self.vg * self.T0)
        m0 = np.inf

        g * self.T0 / self.vg
        self.M0 / (self.b * self.vg * self.T0)

        t, n = smp.symbols('t n')
        z = 1 - (9 / 10) * t ** n

        t, n = smp.symbols('t n')
        z = 1 - (9 / 10) * t ** n
        dzdt = smp.diff(z, t).simplify()
        z = smp.lambdify([t, n], z)
        dzdt = smp.lambdify([t, n], dzdt)

        t = np.linspace(1e-4, 1, 1000)
        z1 = z(t, 1)
        z2 = z(t, 0.7)
        z3 = z(t, 5)

        plt.figure(figsize=(6, 3))
        plt.plot(t, z1, label='M1')
        plt.plot(t, z2, label='M2')
        plt.plot(t, z3, label='M3')
        plt.ylabel('$m(t)/m_0$')
        plt.xlabel('$t/T_0$')
        plt.legend()

        def dSdt(t, S, m0, vg, n, T0=40):
            x, v = S[0], S[1]
            if t < 1:
                dxdt = v
                dvdt = -g * T0 / vg - 1 / (m0 * z(t, n)) * v ** 2 * np.sign(v) - 1 / z(t, n) * dzdt(t, n)
            else:
                dxdt = v
                dvdt = -g * T0 / vg - 1 / (m0 * z(1, n)) * v ** 2 * np.sign(v)
            if (dvdt < 0) * (dxdt < 0) * (x <= 0):
                dxdt = 0
                dvdt = 0
            return [dxdt, dvdt]

        n1, n2, n3 = 1, 0.7, 5.05
        sol1 = solve_ivp(dSdt, [1e-4, 3], y0=[0, 0], t_eval=np.linspace(1e-4, 3, 1000), args=(m0, self.vg, n1, self.T0))
        sol2 = solve_ivp(dSdt, [1e-4, 3], y0=[0, 0], t_eval=np.linspace(1e-4, 3, 1000), args=(m0, self.vg, n2, self.T0))
        sol3 = solve_ivp(dSdt, [1e-4, 3], y0=[0, 0], t_eval=np.linspace(1e-4, 3, 1000), args=(m0, self.vg, n3, self.T0))

        fig, axes = plt.subplots(1, 2, figsize=(10, 3))
        ax = axes[0]
        ax.plot(sol1.t, sol1.y[0], label='n={}'.format(n1))
        ax.plot(sol2.t, sol2.y[0], label='n={}'.format(n2))
        ax.plot(sol3.t, sol3.y[0], label='n={}'.format(n3))
        ax.axvline(1, ls='--', color='k')
        ax.set_ylabel('$x(t)/(v_g T_0)$')
        ax.set_title('Position')
        ax.legend()
        ax = axes[1]
        ax.plot(sol1.t, sol1.y[1], label='n=1')
        ax.plot(sol2.t, sol2.y[1], label='n=0.7')
        ax.plot(sol3.t, sol3.y[1], label='n=1.3')
        ax.axvline(1, ls='--', color='k')
        ax.set_ylabel('$v(t)/v_g$')
        ax.set_title('Velocity')
        fig.text(0.5, -0.04, '$t/T_0$', ha='center', fontsize=20)
        fig.tight_layout()
        plt.show()


class Main(tk.Frame):

    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)

        self.page1 = Page1(self)


        buttonframe = tk.Frame(self)
        container = tk.Frame(self)
        buttonframe.pack(side="top", fill="x", expand=False)
        container.pack(side="top", fill="both", expand=True)

        self.RadioButtons()
        self.page1.place(in_=container, x=0, y=0, relwidth=1, relheight=1)
        self.Labe1()

        customtkinter.set_appearance_mode("System")

        customtkinter.set_default_color_theme("blue")

        self.button1 = customtkinter.CTkButton(self, text='Start', width=200,
                                              height=20, border_width = 0, bg_color= '#00468c',
                                              command=lambda: self.page1.show()
                                              )
        self.button1.place(x=330, y=190)

        customtkinter.set_default_color_theme("blue")

        self.button2  = customtkinter.CTkButton(self, text='Description',width=200,
                                              height=20, border_width = 0, bg_color= '#00468c',
                                              command= self.newwindow1
                                              )
        self.button2.place(x=330, y=220)

        customtkinter.set_default_color_theme("blue")

        self.button4 = customtkinter.CTkButton(self, text='Exit', width=200,
                                               height=20, border_width=0, bg_color='#00468c',
                                                command= self.quit
                                                )
        self.button4.place(x=330, y=250)

        self.im = PIM.open("File/orbit.png")
        self.ph = ImageTk.PhotoImage(self.im)

        self.frameCnt = 7
        self.frames = [PhotoImage(file='File/orbit.png', format='gif -index %i' % (i)) for i in range(self.frameCnt)]



        #self.page1.after(0, self.updategif, 0)


    def newwindow1(self):

        new = Toplevel(root)
        frameCnt = 79
        frames = [PhotoImage(file='satellite-circling-the-earth-animated-clipart.gif', format='gif -index %i' % (i)) for i in range(frameCnt)]

        def update(ind):
            frame = frames[ind]
            ind += 1
            if ind == frameCnt:
                ind = 0
            label.configure(image=frame)
            new.after(100, update, ind)

        label = Label(new)
        label.pack()
        new.after(0, update, 0)

        label2 = Label(new, text = 'This is a educational program which consists of 3 main sections. The first section \n'
                                   'is a simulation of 2 bodys for a understanding on a how an object with large mass\n'
                                   ' affects another object of miniscule mass. The second is a plot program that presents \n '
                                   ' optimum fuel consumption when launching a rocket. The final is a plot of a satellites \n'
                                   'groundtrack for different orbits',
                       height = 5)
        label2.pack()
        new.mainloop()




    def updategif(self, f):
        frame = self.frames[f]
        f += 1
        if f == self.frameCnt:
            f = 0
        self.label.configure(image=frame)
        self.label.place(x=0, y=0)
        self.page1.after(100, self.updategif, f)

    def Labe1(self):

        self.backgroundimage = ImageTk.PhotoImage(file ="milky-way.jpeg")
        self.backgroundimageLabe1 = Label(self, image = self.backgroundimage,
                                          highlightthickness=0, bd = 0,
                                          relief = 'ridge')
        self.backgroundimageLabe1.place(x=0,y=0)

        title = (PIM.open("title.png"))


    def RadioButtons(self):
            var = IntVar()

            values = {
                '3D model': 1,
                'Rocket equation graphing': 2,
                'Groundtrack/orbit plotting': 3,
            }
            h = 160
            for i in range(3):
                h += 25
                self.r = tk.Radiobutton(self.page1, text=list(values)[i], value=values[list(values)[i]], variable = var,
                                     command=lambda: [self.page1.button1.place(x=82,y=310), self.page1.ViewSelected(var)]).place(x=90, y=h)


if __name__ == "__main__":
    root = tk.Tk()
    main = Main(root)
    p1 = Page1(root)
    Option1 = p1.Option
    print(Option1)
    main.pack(side="top", fill="both", expand=True)
    root.geometry("853x480")
    root.resizable(True, True)
    root.mainloop()




