# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 09:48:56 2016

@author: eendebakpt
"""

#%% Make update widget

import qtpy.QtCore as QtCore
import qtpy.QtGui as QtGui
import qtpy.QtWidgets as QtWidgets

class paramUpdateWidget(QtWidgets.QWidget):


    '''
    
    TODO: improve layout
    TODO: convert value label to value textbox 
    TODO: 
    
    '''
    def __init__(self, name, gates):
        super(paramUpdateWidget, self).__init__()

        self.gates=gates
        self.name = name
        self.delta = 10
        self.label=QtWidgets.QLabel(name)
        val=gates.get(name)
        self.value=QtWidgets.QLabel(str(val))
        
        self.plus=QtWidgets.QPushButton('+')
        self.minus=QtWidgets.QPushButton('-')
        vLayout = QtWidgets.QVBoxLayout()
        vLayout.addWidget(self.plus)
        vLayout.addWidget(self.minus)

        hLayout = QtWidgets.QHBoxLayout()
        hLayout.addWidget(self.label)
        hLayout.addWidget(self.value)
        #hLayout.setStyleSheet(" { margin: 0px; padding: 0px; }");
        #hLayout.addWidget(vLayout)
        hLayout.addLayout(vLayout)
        hLayout.setSpacing(1)
        hLayout.setMargin(1)
        #hLayout.setPadding(1)
        
        self.setLayout(hLayout)

        # colors
        self.label.setStyleSheet("QLabel { background-color : #baccba; margin: 2px; padding: 2px; }");
        self.value.setStyleSheet("QLabel { background-color : #cadaca; margin: 2px; padding: 2px; }");
        self.plus.setStyleSheet("QPushButton { margin: 0px; padding: 0px; }");
        self.minus.setStyleSheet("QPushButton { margin: 0px; padding: 0px; }");

        # connect
        self.plus.clicked.connect(self.addValue)
        self.minus.clicked.connect(self.subtractValue)

    def updateLabel(self, value=None):
                if value is None:
                    value=self.gates.get(self.name)
                self.value.setText( str(value) )
                            
    def addValue(self ):
                v=self.gates.get(self.name)+self.delta
                self.gates.set(self.name, v)
                self.updateLabel()
    def subtractValue(self ):
                v=self.gates.get(self.name)-self.delta
                self.gates.set(self.name, v)
                self.updateLabel()



name='L'
if 0:
    p = paramUpdateWidget(name, gates)    
    
    p.setGeometry(1700,110,200,40); p.show()
    self=p


#%% Local test

if __name__=='__main__':
    w=QtWidgets.QWidget()
    w.show()
    w.setGeometry(1640,160,200,800)
    
    layout = QtWidgets.QGridLayout()
    w.setLayout(layout)
    
    for ii, name in enumerate(['L', 'P1','P2','P3','P4','R','SD1a','SD1b','SD1c']):
        pp = paramUpdateWidget(name, gates)    
        layout.addWidget(pp)
    

#%% Remote option


def createUpdateWidget():
    w=QtWidgets.QWidget()
    w.show()
    w.setGeometry(1640,160,200,800)
    
    layout = QtWidgets.QGridLayout()
    w.setLayout(layout)
    
    for ii, name in enumerate(['L', 'P1','P2','P3','P4','R','SD1a','SD1b','SD1c']):
        pp = paramUpdateWidget(name, gates)    
        layout.addWidget(pp)

    if gates.get('_server_name', None) is None:
        print('createUpdateWidget: cannot be used for remote widget since gates not on server..')
    print('created update widget...')
    return w
    
#w = createUpdateWidget()

import multiprocessing as mp

if __name__=='__main__':
    p=mp.Process(target=createUpdateWidget)
    p.start()

#%% 
if 0: 
  FIXME: all instruments on good server
  FIXME: auto check on server
  FIXME: show2D
  FIXME: 1dot_script
  
  