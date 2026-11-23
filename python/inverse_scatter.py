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
class ISTSimulation:
    lambda_vals           : np.ndarray
    norming_constant_vals : np.ndarray

    xmin : float = -10.
    xmax : float = 10.
    xpts : int   = 2000

    tmin : float = 0.
    tmax : float = 0.001
    tpts : int   = 10

    xs    : np.ndarray = field(init=False)
    times : np.ndarray = field(init=False)
    field : np.ndarray = field(init=False)


    def __post_init__(self):
        self.xs = np.linspace(self.xmin, self.xmax, self.xpts)
        self.times = np.linspace(self.tmin, self.tmax, self.tpts)
        self.field = np.empty(0)

    def evolve(self):
        print(f'\n Evolving from t={self.tmin}, to t={self.tmax} \n')
        self.field = np.asarray(sim.inverseScatter(self.lambda_vals, self.norming_constant_vals, self.times, self.xs))
    
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




def main():
    N = 100
    ls = np.linspace(0.01, 2.*N, N)
    norms = np.zeros(N)
    simu = ISTSimulation(lambda_vals = ls, norming_constant_vals = norms)
    simu.evolve()
    simu.plot()

if __name__ == "__main__":
    main()
