'''
Created on Feb 12, 2018

@author: animesh
'''

import Tkinter as tk
from main import *

root = tk.Tk()
root.resizable(width=False, height=False)

frame1 = tk.Frame(width=300, height=576, bg="", colormap="new")
frame1.pack()
FSlabel = tk.Label(frame1, text="Feature Selection: ").pack(side = 'left')
#Drop down menu for feature selection
variable = tk.StringVar(root)
variable.set("Information Gain") # default value
w = tk.OptionMenu(frame1, variable, "Information Gain", "Chi Squared", "ReliefF")
w.pack()
FSRlabel = tk.Label(frame1, text="Recommended: ReliefF").pack(side = 'left')

clxlabel = tk.Label(frame1, text="Classification: ").pack(side = 'left')
#Drop down menu for feature selection
variable = tk.StringVar(root)
variable.set("Information Gain") # default value
w = tk.OptionMenu(frame1, variable, "Naive Bayes", "SVM", "Decision Trees", "Random Forest")
w.pack()
clxRlabel = tk.Label(frame1, text="Recommended: Decision Tree").pack(side = 'left')

def callback():
    print "click!"
    main()

b = tk.Button(frame1, text="Detect", command=callback)
b.pack()








frame2 = tk.Frame(width=300, height=576, bg="", colormap="new")
frame2.pack()

root.mainloop()