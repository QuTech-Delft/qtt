#%% Load packages
from imp import reload
import numpy as np
from numpy import linalg as la
import itertools
import matplotlib.pyplot as plt
import time
import pdb

try:
    import graphviz
except:
    pass
try:
    import pmatlab
except:
    pass

#%%


import qtt; reload(qtt)
from qtt.simulation.dotsystem import DotSystem, TripleDot


#%%
class DoubleDot(DotSystem):
    
    def __init__(self, name='doubledot'):
        super().__init__(name=name)
        
        self.ndots = 2
        self.makebasis(ndots=self.ndots, maxelectrons=3)
        self.varnames = ['det1','det2',
           'osC1','osC2', 'isC1','isC2', 'tun1','tun2']
        self.varnames += itertools.chain( * [ ['eps%d%d' % (d+1,orb+1) for d in range(self.ndots)] for orb in range(0, self.maxelectrons) ])
        self.makevars()        
        self.makevarMs()

ds=DoubleDot()
self=ds

ds.osC1 = 37
ds.osC2 = 36
ds.osC3 = 38
ds.isC1 = 3
ds.isC2 = 2.4

ds.eps11 = .1
ds.eps12 = .15
ds.eps21 = .1
ds.eps22 = .15


plt.figure(20); plt.clf()
ds.visualize(fig=20)
plt.title('Double dot')
pmatlab.tilefigs(20, [2,2])

#%%

class FourDot(DotSystem):
    
    def __init__(self, name='doubledot'):
        super().__init__(name=name, ndots=3)
        
        self.makebasis(ndots=self.ndots, maxelectrons=2)
        self.varnames = ['det%d' % (i+1) for i in range(self.ndots)]
        self.varnames += ['osC%d' % (i+1) for i in range(self.ndots)]
        self.varnames += ['isC%d' % (i+1) for i in range(self.ndots)]
        self.varnames += ['tun%d' % (i+1) for i in range(self.ndots)]
        self.varnames += itertools.chain( * [ ['eps%d%d' % (d+1,orb+1) for d in range(self.ndots)] for orb in range(0, self.maxelectrons) ])
        self.makevars()        
        self.makevarMs()

ds=FourDot()
self=ds

for ii in range(ds.ndots):
    setattr(ds, 'osC%d' % ( ii+1), 35)
for ii in range(ds.ndots-1):
    setattr(ds, 'isC%d' % (ii+1), 3)

if 0:
    ds.eps11 = .1
    ds.eps12 = .15
    ds.eps21 = .1
    ds.eps22 = .15


plt.figure(20); plt.clf()
ds.visualize(fig=20)
plt.title('Double dot')
pmatlab.tilefigs(20, [2,2])


#%%
ds.resetMu(0)
ds.det2=4
#ds.isC3=3

paramnames = ['det1','det3']
minmax = [[-40,80,-40,80],[-40,-40,80,80]]
#minmax = 1.5*np.array([[-20,80,-20,80],[-20,-20,80,80]])
npointsx = 60
npointsy = 60
ds.makeparamvalues2D(paramnames,minmax,npointsx,npointsy)

ds.simulatehoneycomb(verbose=1, usediag=True)

plt.figure(10); plt.clf()
plt.pcolor(ds.vals2D[ paramnames[0]],ds.vals2D[ paramnames[1]],ds.honeycomb,cmap='Blues')
plt.xlabel('Abcissa gate (mV)')
plt.ylabel('Ordinate gate (mV)')
plt.colorbar()
plt.show()

self=ds

i = 30
j = 30
print(ds.hcgs[j,i])
print(ds.hcgs[0,0])

pmatlab.tilefigs(10, [2,2])

#%%
ds.makeH()
ds.solveH()
print(ds.OCC)

#%%


val=5*ds.hcgs[:,:,0] + 2*ds.hcgs[:,:,1] + ds.hcgs[:,:,2]

plt.figure(11); plt.clf()
for ii in range(3):
    plt.subplot(2,2,ii+1)
    val=ds.hcgs[:,:,ii]
    plt.pcolor(ds.vals2D[ paramnames[0]],ds.vals2D[ paramnames[1]],val,cmap='Blues')
    plt.xlabel('Abcissa gate (mV)')
    plt.ylabel('Ordinate gate (mV)')
    plt.colorbar()
plt.show()


#%%

def showMmatrix(self, name='det1', fig=10):
    plt.figure(fig);
    plt.clf();  
    plt.imshow(getattr(self, 'M' + name), interpolation='nearest'); plt.title('M'+name)
    plt.grid('on')


#%%

def showM(self, fig=10):
    plt.figure(fig);
    plt.clf();  
    plt.subplot(2,2,1)
    plt.imshow(self.Mdet1, interpolation='nearest'); plt.title('Mdet1')
    plt.grid('on')

    plt.subplot(2,2,2)
    plt.imshow(self.MosC1, interpolation='nearest'); plt.title('MosC1')
    plt.grid('on')
    plt.subplot(2,2,3)
    plt.imshow(self.MisC1, interpolation='nearest'); plt.title('MisC1')
    plt.grid('on')
    plt.subplot(2,2,4)
    plt.imshow(self.Meps11, interpolation='nearest'); plt.title('Meps11')
    plt.grid('on')

#MosC
#MisC
#Meps

showM(self, fig=10)
        
#%%
if 0:
    import pydotplus as pd
    graph = pd.Dot(graph_type='graph')
    
    import networkx as nx
    import matplotlib.pyplot as plt
    G = nx.Graph()
    G=nx.DiGraph()
    
    G = nx.DiGraph()
    G.graph['rankdir'] = 'LR'
    G.graph['dpi'] = 120
    G.add_cycle(range(4))
    G.add_node(0, color='red', style='filled', fillcolor='pink')
    G.add_node(1, shape='square')
    G.add_node(3, style='filled', fillcolor='#00ffff')
    G.add_edge(0, 1, color='red', style='dashed')
    G.add_edge(3, 3, label='a')
    nx.draw(G)
    from nxpd import draw
    draw(G)
    
    for ii in range(self.ndots):
        G.add_node(ii)
    
    for ii in range(self.ndots):
        G.add_edge(ii, ii, label='det%d' % ii)
    
        #G.add_edge('1','2')
    
    #G.add_node(10, {'name': 'sdfs', 'label': 'hi'})
    
    plt.figure(10)
    plt.clf()
    plt.axis('off')
    pos=nx.layout.spring_layout(G)
    #nx.draw(G, pos=pos)
    nx.draw_networkx_nodes(G, pos=pos, node_size=4000, node_color="g")
    nx.draw_networkx_edges(G, pos=pos)
    
    labels=dict( [ (i, 'node%d' % i) for i in range(3) ] ) 
    nx.draw_networkx_labels(G,pos=pos, labels=labels,font_size=16)
    plt.show()
    
    


#%% Testing

def testTriple():
    tripledot = TripleDot(name='tripledot')
    
    tripledot.det1 = 0
    tripledot.det2 = 0
    tripledot.det3 = 0
    tripledot.eps11 = .1
    tripledot.eps12 = .15
    tripledot.eps13 = .2
    tripledot.eps21 = .1
    tripledot.eps22 = .15
    tripledot.eps23 = .2
    tripledot.eps31 = .1
    tripledot.eps32 = .15
    tripledot.eps33 = .2
    tripledot.osC1 = 27
    tripledot.osC2 = 26
    tripledot.osC3 = 28
    tripledot.isC1 = 6
    tripledot.isC2 = 6
    tripledot.isC3 = 2
    tripledot.tun1 = 0.1
    tripledot.tun2 = 0.1
    
    return tripledot
    

#%%

if __name__ == "__main__":
    import doctest
    #doctest.testmod()
    
    
#%%
tripledot = testTriple()    
self=tripledot


paramnames = ['det1','det3']
minmax = [[-20,80,-20,80],[-20,-20,80,80]]
npointsx = 100
npointsy = 100
tripledot.makeparamvalues2D(paramnames,minmax,npointsx,npointsy)

tripledot.simulatehoneycomb(verbose=1)

plt.figure(10); plt.clf();
plt.pcolor(tripledot.vals2D['det1'],tripledot.vals2D['det3'],tripledot.honeycomb,cmap='Blues')
plt.xlabel('Abcissa gate (mV)')
plt.ylabel('Ordinate gate (mV)')
plt.colorbar()
plt.show()

self=tripledot

#%%
if 0:
    paramnames = ['det1','det2','det3']
    bottomtop = [[-20,100],[-20,100],[-20,100]]
    rangex = 60
    npointsx = 200
    npointsy = 200
    tripledot.makeparamvalues2Dx(paramnames,bottomtop,rangex,npointsx,npointsy)
    tripledot.simulatehoneycomb()
    
    fig=plt.figure(10); plt.clf()
    fig.set_size_inches(10.5, 10.5)
    plt.pcolor(tripledot.vals2D['det3']-tripledot.vals2D['det1'],tripledot.vals2D['det2'],tripledot.honeycomb,cmap='Blues')
    plt.xlabel('Detuning L-R (mV)')
    plt.ylabel('Filling L+M+R (mV)')
    plt.colorbar()
    plt.axis('image')
    plt.show()

#%%


def makeH(self):
    #self.H = np.full((self.Nt, self.Nt), 0, dtype=float)
    self.H = np.zeros( (self.Nt, self.Nt), dtype=float)
    for name in self.varnames:
        self.H = self.H + getattr(self, 'M' + name) * getattr(self,  name)
    self.solved=False
    return self.H

#%timeit makeH(self)

#%%

def isdiagonal(HH):
    return not(np.any(HH-np.diag(np.diagonal(HH))))


self.tun1=.0
self.makeH()

def solveH2(self, usediag=False):
        if usediag:
            self.energies=self.H.diagonal()
            idx=np.argsort(self.energies)
            self.energies=self.energies[idx]
            self.eigenstates[:]=0 # =np.zeros( (self.Nt, self.Nt), dtype=float)
            for i,j in enumerate(idx):
                self.eigenstates[j, i]=1
        else:
            self.energies,self.eigenstates = la.eigh(self.H)
        self.states = self.eigenstates
        self.stateprobs = np.square(np.absolute(self.states))
        self.stateoccs = np.dot(self.stateprobs.T, self.basis)
        self.nstates = np.sum(self.stateoccs,axis=1,dtype=float)
        self.orderstatesbyE()
        self.solved=True
        self.findcurrentoccupancy()
        return self.energies,self.eigenstates

#%timeit a,b=solveH(self)
%timeit a2,b2=solveH2(self, usediag=True)

#%%
self.makeH()

def showH(self, fig=10):
    plt.figure(fig);
    plt.clf();  
    #plt.subplot(2,2,1)
    plt.imshow(self.H, interpolation='nearest'); plt.title('Hamiltonian')
    plt.grid('on')

showH(self)