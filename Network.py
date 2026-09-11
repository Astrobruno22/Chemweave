#Parser for fixed-width ratefile

from dataclasses import dataclass, field
import numpy as np

Elements = ["H", "He", "C", "O", "Mg", "Si", "Fe", "S", "PAH", "Ne", "Ar", "D"]


@dataclass
class Species:
    index: int #1-based index as used in ratefile
    name: str
    charge: int
    comp: dict #{"H":2, "O":1, ..} only non-zero entries

    @property
    def i0(self):
        return self.index - 1

@dataclass
class Reaction:
    index: int
    reactants: list
    products: list
    alpha: float | None
    beta: float
    gamma: float
    label: str
    r_counts: dict = field(default_factory=dict) #{species_index: multiplicity}
    p_counts: dict = field(default_factory=dict)

class Network:
    def __init__(self, path):
        raw = open(path).read().split("\n")
        self.n_elements = int(raw[0].split()[0])
        self.n_species = int(raw[1].split()[0])
        self.n_reactions = int(raw[2].split()[0])
        self.cr = float(raw[3].split()[0])

        self.e_abundance = {}
        for i,line in enumerate(raw[4:4 + self.n_elements]):
            self.e_abundance[Elements[i]] = float(line.split()[0])

        #__species__
        self.species: list[Species] = []
        off = 4 + self.n_elements + 1
        for line in raw[off:off + self.n_species]:
            Fw = line.ljust(60)
            index = int(Fw[0:3])
            name = Fw[3:8].strip()
            charge = int(Fw[8:10]) if Fw[8:10].strip() else 0
            comp = {}
            for i, e in enumerate(Elements):
                f = Fw[10+3*i: 13+3*i]
                if f.strip():
                    comp[e] = int(f)
            self.species.append(Species(index,name,charge,comp))
        self.name_of = [s.name for s in self.species]
        self.index_of = {s.name: s.index for s in self.species}
        #__reactions__
        self.reactions: list[Reaction] = []
        off += self.n_species
        for line in raw[off:off+self.n_reactions]:
            Fw = line.ljust(60)
            index = int(Fw[0:3])
            slots = [Fw[3+3*k:6+3*k] for k in range(8)]
            ids = [int(s) if s.strip() else 0 for s in slots]
            reactants = [i for i in ids[0:3] if i > 0]
            products = [i for i in ids[3:8] if i > 0]
            tok = Fw[27:].split()
            if len(tok) == 4:
                alpha, beta, gamma, label = float(tok[0]), float(tok[1]), float(tok[2]), tok[3]
            elif len(tok) == 3:
                alpha = None
                beta, gamma, label = float(tok[0]), float(tok[1]), tok[2]
            #check file; sigman = 
            rx = Reaction(index, reactants, products, alpha, beta, gamma, label)
            for i in reactants:
                rx.r_counts[i] = rx.r_counts.get(i,0)+1
            for i in products:
                rx.p_counts[i] = rx.p_counts.get(i, 0)+1
            self.reactions.append(rx)

        assert len(self.species) == self.n_species
        assert len(self.reactions) == self.n_reactions

        #__composition matrix__
        self.comp_matrix = np.zeros((self.n_species, len(Elements)))
        self.charge = np.zeros(self.n_species)
        for s in self.species:
            self.charge[s.i0] = s.charge
            for k,e in enumerate(Elements):
                self.comp_matrix[s.i0,k] = s.comp.get(e,0)

    def pretty(self, rx: Reaction) -> str:
            lhs = " + ".join(self.name_of[i - 1] for i in rx.reactants)
            rhs = " + ".join(self.name_of[i - 1] for i in rx.products)
            return f"{rx.index:3d} [{rx.label}] {lhs}  ->  {rhs}"

    def check_conservation(self, verbose=True):
            """Every reaction must conserve each element and total charge."""
            bad = []
            for rx in self.reactions:
                dl = self.comp_matrix[[i - 1 for i in rx.reactants]].sum(0)
                dr = self.comp_matrix[[i - 1 for i in rx.products]].sum(0)
                cl = self.charge[[i - 1 for i in rx.reactants]].sum()
                cr = self.charge[[i - 1 for i in rx.products]].sum()
                if not np.allclose(dl, dr) or cl != cr:
                    bad.append((rx, dl - dr, cl - cr))
            if verbose:
                print(f"conservation check: {len(bad)} / {len(self.reactions)} reactions unbalanced")
                for rx, d, c in bad[:20]:
                    els = {Elements[k]: d[k] for k in range(len(Elements)) if d[k] != 0}
                    print(f"   {self.pretty(rx)}   dElem={els} dCharge={c}")
            return bad


if __name__ == "__main__":
    net = Network("ratefile")
    print(f"{net.n_species} species, {net.n_reactions} reactions, cosmic ray rate={net.cr}")
    print("elemental abundances:", net.e_abundance)
    print(net.pretty(net.reactions[1]))
    net.check_conservation()
