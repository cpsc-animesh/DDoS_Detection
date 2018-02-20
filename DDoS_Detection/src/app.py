'''
Created on Feb 12, 2018

@author: animesh
'''

import Tkinter as tk
import pygubu

FS_selected = 0
clx_selected = 0
class Application:
    def __init__(self, master):
        self.master = master
        self.builder = builder = pygubu.Builder()
        builder.add_from_file('/home/animesh/Desktop/mainapp.ui')
        self.mainwindow = builder.get_object('mainWindow', master)
        
        #Set main menu for Feature Selection
        self.mainmenu = menu = builder.get_object('mainmenu', self.master)
        #self.set_menu(menu)
        
        #Set main menu for Classification
        self.mainmenuclx = menu = builder.get_object('mainmenuclx', self.master)
        
        builder.connect_callbacks(self)
    

    
    def IG_submenu_clicked(self, itemid):
        if itemid == 'IG_submenu':
            FS_selected = itemid
            return FS_selected
            print("Hello from IG")

    def chi2_submenu_clicked(self, itemid):
        if itemid == 'chi2_submenu':
            FS_selected = itemid
            print("Hello from chi2")
            
    def reliefF_submenu_clicked(self, itemid):
        if itemid == 'reliefF_submenu':
            FS_selected = itemid
            print("Hello from reliefF")

    def naivebayes_submenu_clicked(self, itemid):
        if itemid == 'NaiveBayes_submenu':
            clx_selected = itemid
            print("Hello from Naive Bayes")
            
    def SVM_submenu_clicked(self, itemid):
        if itemid == 'SVM_submenu':
            clx_selected = itemid
            print("Hello from SVM")
            
    def decisionTree_submenu_clicked(self, itemid):
        if itemid == 'DecisionTree_submenu':
            clx_selected = itemid
            print("Hello from Decision Tree")
            
    def RandomForest_submenu_clicked(self, itemid):
        if itemid == 'RandomForest_submenu':
            clx_selected = itemid
            print("Hello from Random Forest")       
    
    def detectButton_clicked(self):
        self.master.quit()
    
    

if __name__== '__main__':
    root = tk.Tk()
    root.title('DDoS Detect')
    app = Application(root)
    
    root.mainloop()

