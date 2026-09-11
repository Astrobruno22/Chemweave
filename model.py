#Loader the solutions from the model provided - model.tgz

import os
import numpy as np

class DiskModel:
    def __init__(self,model,net,xion_path=None,dustdensity=None):
        self.dir = model
        self.net = net
        self.density = np.loadtxt(os.path.join(model,"density.txt"))
        self.temp = np.loadtxt(os.path.join(model,"gastemp.txt"))
        self.av = np.loadtxt(os.path.join(model,"av.txt"))
        self.g0 = np.loadtxt(os.path.join(model,"G0.txt"))
        self.ncells = len(self.density)self.gr = np.loadtxt(dustdensity)[:, 0]

        #abundance matrix
        self.x = np.zeros((net.n_species,self.ncells))
        missing = []
        for s in net.species:
            p = os.path.join(model,s.name + ".txt")
            if os.path.exists(p):
                self.x[s.i0] = np.loadtxt(p)
            else:
                missing.append(s.name)
        self.missing = missing
        self.xion = np.loadtxt(xion_path)
        self.gr = np.loadtxt(dustdensity)[:, 0]

    def cell(self,c):
        # Physical state and number densities for one cell
        return dict(
            density = self.density[c], T=self.temp[c], Av = self.av[c], G0 = self.g0[c],
            xion = self.xion[c], n=self.x[:,c]*self.density[c],x=self.x[:,c],
            gr = self.gr[c],
        )
