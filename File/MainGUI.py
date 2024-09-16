
from tkinter import *
import time
import os
root = Tk()

frameCnt = 12
frames = [PhotoImage(file='sa.gif', format ='gif -index %i' % (i)) for i in range(frameCnt)]

def update(ind):

    frame = frames[ind]
    ind += 1
    if ind == frameCnt:
        ind = 0
    label.configure(image=frame)
    root.after(100, update, ind)
label = Label(root)
label.pack()
root.after(0, update, 0)
root.mainloop()

'''
from tkinter import *
from PIL import ImageTk
from tkinter import ttk
import customtkinter
import tkinter as tk
import time
import os

main_window = Tk()
root = tk.Tk()

main_window.title('Satellite Simulation Program')
main_window.geometry('590x332')
main_window.resizable(True,True)

frameCnt = 79
frames = [PhotoImage(file='satellite-circling-the-earth-animated-clipart.gif',format = 'gif -index %i' %(i)) for i in range(frameCnt)]
print(frames)
customtkinter.set_appearance_mode("System")
customtkinter.set_default_color_theme("blue")
button = customtkinter.CTkButton(main_window, text = 'Start', width = 10, command = lambda:main_window.quit())

def Label1():
    backgroundImage = ImageTk.PhotoImage(file ="backgroundimage.png")
    backgroundImageLabel1 = Label(main_window,image = backgroundImage)
    backgroundImageLabel1.place(x=0,y=0)

    canvas1 = Canvas(main_window, width=400,height=330)
    canvas1.place(x = 300, y = 60)

    title = Label(text = "Satellite Simulation", font = "Bold 30")
    title.place(x=300, y=80)




def update(f):

    frame = frames[f]
    f += 1
    if f == frameCnt:
        f = 0
    label.configure(image=frame)
    main_window.after(100, update, f)




label = Label(main_window)
main_window.after(0, update, 0)
button.pack(ipadx = 1, ipady = 1, expand = False)
main_window.mainloop()
'''