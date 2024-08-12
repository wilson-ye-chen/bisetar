import numpy as np
from bisetar.mcmc import BiSetar

class BiSetarCls(BiSetar):
    def __init__(self, x):
        super().__init__(x)
        p = np.arange(0.1, 0.9, 0.01)
        q = np.quantile(x, p)
        r1, r2 = np.meshgrid(q, q)
        r1 = r1.reshape(-1, 1)
        r2 = r2.reshape(-1, 1)
        self.r_grid = np.hstack((r1, r2))

    def cls(self, r_grid=None):
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
