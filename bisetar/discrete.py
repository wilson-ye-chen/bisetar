import numpy as np
from scipy.special import logsumexp

class GridDist():
    def __init__(self, r1, r2, logp):
        t1, t2 = np.meshgrid(r1, r2, indexing='ij')
        a1 = t1[:-1, 1:]
        b1 = t1[1:, 1:]
        a2 = t2[1:, :-1]
        b2 = t2[1:, 1:]
        c1 = (a1 + b1) / 2
        c2 = (a2 + b2) / 2
        self.u1 = np.hstack((a1.reshape(-1, 1), b1.reshape(-1, 1)))
        self.u2 = np.hstack((a2.reshape(-1, 1), b2.reshape(-1, 1)))
        self.r = np.hstack((c1.reshape(-1, 1), c2.reshape(-1, 1)))
        self.n = self.r.shape[0]

        lp = np.empty(self.n)
        for i in range(self.n):
            lp[i] = logp(self.r[i])
        self.logp = lp - logsumexp(lp)

    def sample(self, n):
        j = np.empty(n, dtype=int)
        for i in range(n):
            g = np.random.gumbel(size=self.n)
            j[i] = np.argmax(g + self.logp)
        r1 = np.random.uniform(self.u1[j, 0], self.u1[j, 1])
        r2 = np.random.uniform(self.u2[j, 0], self.u2[j, 1])
        return np.vstack((r1, r2)).T
