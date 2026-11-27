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
SAVE_FILE = ROOT / "data" / "animation.mp4"

@dataclass
class Params:

    period: float = 2*np.pi

    xmin : float = -10
    xmax : float = 10
    xpts : int   = 1000

    tmin : float = 0.
    tmax : float = 0.1
    tpts : int   = 10
    
@dataclass
class InverseScattering:
    p     : Params
    spectrum          : np.ndarray
    norming_constants : np.ndarray

    x_grid: np.ndarray = field(init=False)
    times : np.ndarray = field(init=False)
    kernels: np.ndarray = field(init=False)


    def __post_init__(self):
        p = self.p
        self.x_grid = np.linspace(p.xmin, p.xmax, p.xpts)
        self.times = np.linspace(p.tmin, p.tmax, p.tpts)
        self.kernels = np.empty(0, dtype=np.complex128)

    def run(self):
        print(f'\nEvolving from t={self.p.tmin}, to t={self.p.tmax} \n')
        k = 2*np.pi/self.p.period
        res = np.asarray(sim.inverseScatter(self.spectrum, self.norming_constants, k*k*self.times))
        self.kernels = res

@dataclass
class DirectEvolution:
    p     : Params
    initial_qs : np.ndarray

    xs    : np.ndarray = field(init=False)
    times : np.ndarray = field(init=False)
    kernels: np.ndarray = field(init=False)


    def __post_init__(self):
        p = self.p
        self.x_grid = np.linspace(p.xmin, p.xmax, p.xpts)
        self.times = np.linspace(p.tmin, p.tmax, p.tpts)
        self.kernels = np.empty(0, dtype=np.complex128)

    def run(self):
        print(f'\nEvolving from t={self.p.tmin}, to t={self.p.tmax} \n')
        k = np.pi*2/self.p.period
        self.kernels = np.asarray(sim.directEvolve(self.initial_qs, k*k*self.times))


def plot(xs, times, field):
    fig, ax = plt.subplots()
    line, = ax.plot(xs, field[0, :], color="black")
    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(0, np.max(field)) 
    ax.set_xlabel("x")
    title = ax.set_title(f"t = {times[0]:.0f}")
    
    def update(i):
        line.set_ydata(field[i, :])
        title.set_text(f"t = {times[i]:.0f}")
        return line, title
    
    anim = FuncAnimation(fig, update, frames=len(times), interval=50, repeat=False)
    #anim.save(filename=SAVE_FILE, writer="ffmpeg")
    plt.show()

def calculate_field(period, xs, kernels):
    k = 2*np.pi/period
    kernels = kernels[:, :, None]
    r, theta = np.abs(kernels), np.angle(kernels)
    poisson = k*(1 - r*r) / (1 + r*r - 2*r*np.cos(k*(xs + theta/k)))
    return poisson.sum(axis=1)

def load_kernels(path):
    raw = np.loadtxt(path)
    return raw[0::2] + 1j*raw[1::2]

def ist_example():
    N = 100
    L = 100

    params = Params(
            period = L,
            xmin = -2*L,
            xmax =  2*L,
            xpts = 10000,
            tmin = 0.,
            tmax = 10,
            tpts = 10,
            )

    spectrum = 2*np.arange(0, N)
    norming_constants = np.zeros(N)

    simu = InverseScattering(p = params,
                             spectrum = spectrum,
                             norming_constants = norming_constants)
    simu.run()

    times, xs, qs, param = simu.times, simu.x_grid, simu.kernels, simu.p
    field = calculate_field(param.period, xs, qs)
    plot(xs, times, field)

def direct_example():
    N = 100
    L = 100

    params = Params(
            period = L,
            xmin = -2*L,
            xmax =  2*L,
            xpts = 1000,
            tmin = 0.,
            tmax = 10,
            tpts = 10,
            )


    init_qs = load_kernels(DATA_FILE)
    simu = DirectEvolution(p = params,
                           initial_qs = init_qs)
    simu.run()

    times, xs, qs, param = simu.times, simu.x_grid, simu.kernels, simu.p
    field = calculate_field(param.period, xs, qs)
    plot(xs, times, field)

if __name__ == "__main__":
    direct_example()
    ist_example()
