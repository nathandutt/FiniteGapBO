import numpy as np
import matplotlib.pyplot as plt
import inverse_scatter as ist
from bisect import bisect
from dataclasses import dataclass, field

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "clib"))
import sim


def calculate_field(xs, all_qs):
    qs = all_qs[:, None]                 
    r, theta = np.abs(qs), np.angle(qs)          
    kernel = (1 - r*r) / (1 + r*r - 2*r*np.cos(xs + theta))   
    field = kernel.sum(axis=0)              
    return field

def window_field(a, b, all_qs, pts_per_width=100):
    n = int(pts_per_width * (b - a) / (1 - np.abs(all_qs)).min())
    xs = a + (np.arange(n) + 0.5) * (b - a) / n
    return calculate_field(xs, all_qs)

@dataclass
class AllBins:

    all_qs : np.ndarray

    xmin: float = -np.pi
    xmax: float = np.pi
    num_bins: int = 20
    pts_per_bin : float = 2000


    bin_size: float = (xmax - xmin)/num_bins
    bins: list[list[complex]] = field(init=False)
    bin_edges : np.ndarray = field(init=False)

    def __post_init__(self):
        self.bin_edges = np.linspace(self.xmin, self.xmax, self.num_bins+1, endpoint=True)
        self.bins = [[] for _ in range(self.num_bins)]
        for q in self.all_qs:
            r, x0 = np.abs(q), -np.angle(q)
            bin_idx = bisect(self.bin_edges, x0) - 1
            bin_size = self.bin_edges[bin_idx+1] - self.bin_edges[bin_idx]
            self.bins[bin_idx].append(q**self.num_bins) #rescale the qs

    def diagonalize(self):
        eigs = sim.localIST(self.bins)
        scaling = 2*np.pi/self.bin_size
        rescaled_eigs = [[-scaling*eig for eig in bin_eigs] for bin_eigs in eigs]
        return rescaled_eigs

    def mean_var(self):
        means = np.zeros(self.num_bins)
        variances = np.zeros(self.num_bins)
        for i in range(self.num_bins):
            u = window_field(self.bin_edges[i], self.bin_edges[i+1], self.all_qs)
            means[i], variances[i] = u.mean(), u.var()
        return means, variances


def main():
    L = 2*np.pi
    N = 300
    ls = 0.001 + 2*np.arange(0, N)
    norms = np.zeros(N)
    simu = ist.ISTSimulation(lambda_vals = ls, norming_constant_vals = norms)
    simu.evolve()
    binning = AllBins(simu.all_qs[9])

    eigs = binning.diagonalize()

    ns = np.array([len(bin_eigs) for bin_eigs in eigs])
    avgs = 2*np.pi*np.array([len(bin_eigs) for bin_eigs in eigs])/binning.bin_size
    real_avgs = binning.mean_var(1)

    real_sq_avgs = np.sqrt(binning.mean_var(2))
    sq_avgs = 2.*2*np.pi/binning.bin_size*np.array([sum(bin_eigs) for bin_eigs in eigs]) - ns * (2.*np.pi/binning.bin_size)**2
    normalized_sq = np.sqrt(sq_avgs - avgs**2)
    
    
    plt.plot(np.arange(0, binning.num_bins), normalized_sq, marker='o', color="black")
    plt.plot(np.arange(0, binning.num_bins), real_sq_avgs, marker = 'o', color="blue")
#    plt.plot(np.arange(0, binning.num_bins), avgs, color = "red")
#    plt.plot(np.arange(0, binning.num_bins), real_avgs, color = "black")
    plt.show()


if __name__ == "__main__":
    main()
