'''
Created on Feb 12, 2018

@author: animesh
'''
from Tkinter import *

root = Tk()
firstFrame = Frame(root)
firstFrame.pack(side = TOP)
secondFrame = Frame(root)
secondFrame.pack()
thirdFrame = Frame(root)
thirdFrame.pack()
fourthFrame = Frame(root)
fourthFrame.pack(side = BOTTOM)


greenbutton = Button(firstFrame, text="green", fg="green")
greenbutton.pack( side = LEFT )

separator = Frame(height=2, bd=1, relief=SUNKEN)
separator.pack(fill=X, padx=5, pady=5)

bluebutton = Button(secondFrame, text="blue", fg="blue")
bluebutton.pack( side = LEFT )

root.mainloop()