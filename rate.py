#Rate evaluator

import numpy as np
xion = ["H", "He", "C", "O", "Mg", "Si", "S", "Fe"]
ALPHA_LABELS = ("AR", "RR", "ER", "CI", "3B", "CR", "UV", "GR")

def dead_reactions(net, include_xr=False, pa_rate=0.0):
    """1-based indices of reactions that can never produce a non-zero rate:
 
      * alpha is blank or zero, for a label whose formula uses alpha
      * PA reactions, while pa_rate is 0
    """
    dead = []
    for rx in net.reactions:
        if rx.label == "PA":
            if pa_rate == 0.0:
                dead.append(rx.index)
        elif rx.label in ALPHA_LABELS:
            if rx.alpha is None or rx.alpha == 0.0:
                dead.append(rx.index)
        elif rx.label == "XR" and include_xr:
            if rx.alpha is None or rx.alpha == 0.0:
                dead.append(rx.index)
    return dead

class RateEvaluator: #check if it multiples by xion value
    def __init__(self, net, include_xr=False, pa_rate=0.0):
        self.net = net
        dead = set(dead_reactions(net, include_xr, pa_rate))
        self.dead_j = [j for j, rx in enumerate(net.reactions) if rx.index in dead]
        self.xr_weights = {}
        for j, rx in enumerate(net.reactions):
            if rx.label != 'XR':
                continue
            sp = net.species[rx.reactants[0]-1]
            w = np.zeros(len(xion))
            for e, c in sp.comp.items():
                if e in xion:
                    w[xion.index(e)] += c
                elif e in ("Ne", "Ar"):
                    w[xion.index('H')] += c
            self.xr_weights[j]=w
    def k(self, T, nH, G0, Av, xion, cr, sigma_nd):
        net = self.net
        cr = net.cr
        k = np.zeros(net.n_reactions)
        for j, rx in enumerate(net.reactions):
            lab = rx.label
            a,b,g = rx.alpha,rx.beta,rx.gamma

            if lab in ("AR", "RR", "ER", "CI", "3B"):
                if a is None:
                    k[j] = 0.0
                else:
                    k[j] = a * (T/300.0) ** b * np.exp(-g/T)
            elif lab == "CR":
                k[j] = (a or 0.0) * cr
            elif lab == "UV":
                k[j] = (a or 0.0) * G0 * np.exp(-g * Av)
            elif lab == "XR":
                if xion is None:
                    k[j] = 0.0
                else:
                    w = self.xr_weights[j]
                    k[j] = 0.0 if np.any(np.isnan(w)) else float(np.dot(w, xion[:8]))
            elif lab == "GR":
                k[j] = (a or 0.0) * sigma_nd
            elif lab == "PA":
                k[j] = 0.0
            for j in self.dead_j:
                k[j] = 0.0
        return k
        return k
    def fluxes(self, k, n):
        net = self.net
        R = np.empty(net.n_reactions)
        for j, rx in enumerate(net.reactions):
            v = k[j]
            for i in rx.reactants:
                v*= n[i-1]
            R[j] = v
        return R
                    
