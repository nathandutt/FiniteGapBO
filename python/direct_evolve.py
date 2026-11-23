import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass, field

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "clib"))

import sim

DATA_FILE = ROOT / "data" / "initial_qs.res"

@dataclass
class DESimulation:
    r_vals     : np.ndarray
    phase_vals : np.ndarray

    xmin : float = -10.
    xmax : float = 10.
    xpts : int   = 2000

    tmin : float = 0.
    tmax : float = 0.1
    tpts : int   = 1000

    xs    : np.ndarray = field(init=False)
    times : np.ndarray = field(init=False)
    field : np.ndarray = field(init=False)


    def __post_init__(self):
        self.xs = np.linspace(self.xmin, self.xmax, self.xpts)
        self.times = np.linspace(self.tmin, self.tmax, self.tpts)
        self.field = np.empty(0)

    def evolve(self):
        self.field = np.asarray(sim.directEvolve(self.r_vals, self.phase_vals, self.times, self.xs))
    
    def plot(self):

        #Plotting
        fig, ax = plt.subplots()
        line, = ax.plot(self.xs, self.field[0, :])
        ax.set_xlim(self.xs[0], self.xs[-1])
        ax.set_ylim(0, np.max(self.field)) 
        ax.set_xlabel("x")
        title = ax.set_title(f"t = {self.times[0]:.4f}")
        
        def update(i):
            line.set_ydata(self.field[i, :])
            title.set_text(f"t = {self.times[i]:.4f}")
            return line, title
        
        anim = FuncAnimation(fig, update, frames=len(self.times), interval=50)
        plt.show()



def load_qs(path):
    raw = np.loadtxt(path)
    init_qs = raw[0::2] + 1j*raw[1::2]
    rs = np.abs(init_qs)
    phases = np.angle(init_qs)
    return [rs, phases]

def main():
    rs, phases = load_qs(DATA_FILE)
    simu = DESimulation(r_vals = rs, phase_vals = phases)
    simu.evolve()
    simu.plot()




if __name__ == "__main__":
    main()
