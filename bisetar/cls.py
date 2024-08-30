import numpy as np
from bisetar.mcmc import BiSetar

class BiSetarCls(BiSetar):
    def __init__(self, x):
        super().__init__(x)
        p = np.arange(0.1, 0.9, 0.01)
        q = np.nanquantile(x, p)
        r1, r2 = np.meshgrid(q, q)
        r1 = r1.reshape(-1, 1)
        r2 = r2.reshape(-1, 1)
        self.r_grid = np.hstack((r1, r2))

    def learn_r(self, r_grid=None):
        if r_grid == None:
            r_grid = self.r_grid

        n_grid = r_grid.shape[0]
        v_ttl = np.zeros(n_grid)
        for i in range(n_grid):
            xr = self.splitx(r_grid[i])
            for j in range(4):
                if len(xr[j][0]) < 5:
                    v_ttl[i] = np.nan
                else:
                    v = self.lsfit(xr[j][0], xr[j][1], xr[j][2])[1]
                    v_ttl[i] += v
        i_min = np.nanargmin(v_ttl)
        return(r_grid[i_min], v_ttl[i_min], v_ttl)

    def learn_phi(self, r):
        xr = self.splitx(r)
        phi = np.empty((4, 4))
        for i in range(4):
            b, v = self.lsfit(xr[i][0], xr[i][1], xr[i][2])[:2]
            phi[i, :3] = b
            phi[i, 3] = v
        return phi.flatten()

    def find_feasible(self, n_min):
        n_grid = self.r_grid.shape[0]
        nr = np.zeros((n_grid, 4))
        for i in range(n_grid):
            xr = self.splitx(self.r_grid[i])
            for j in range(4):
                nr[i, j] = len(xr[j][0])
        fsb = np.sum(nr >= n_min, axis=1) == 4
        return (self.r_grid[fsb], nr)

class BiSetarClsUpper(BiSetarCls):
    def __init__(self, x):
        # Set lower observations to NaNs
        self.x = x.copy()
        n = x.shape[0]
        np.fliplr(self.x)[np.tril_indices(n, k=-1)] = np.nan
        super().__init__(self.x)
