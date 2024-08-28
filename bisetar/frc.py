import numpy as np

class BiSetarForecast:
    def __init__(self, x, theta, nmc):
        # Set lower observations to NaNs
        self.x = x.copy()
        n = x.shape[0]
        np.fliplr(self.x)[np.tril_indices(n, k=-1)] = np.nan

        # Indices of NaNs
        self.ifrc = np.argwhere(np.isnan(self.x))
        self.nfrc = self.ifrc.shape[0]

        # Reverse indices
        self.irev = np.full((n, n), -1)

        # Sampled forecasts
        self.xhat = np.empty((self.nfrc, nmc))

        # If there is only one parameter vector
        if np.ndim(theta) == 1:
            theta = np.tile(theta, (nmc, 1))

        # Organise parameters
        self.r = theta[:, :2]
        self.b = [
            theta[:, 2:5],
            theta[:, 6:9],
            theta[:, 10:13],
            theta[:, 14:17]]
        self.v = theta[:, [5, 9, 13, 17]]

        # Monte Carlo sample size
        self.nmc = nmc

    def forecast_one(self, x1, x2):
        n = len(x1)
        y = np.empty(n)
        f = np.vstack((np.ones(n), x1, x2)).T
        for i in range(n):
            gtr1 = x1[i] > self.r[i, 0]
            gtr2 = x2[i] > self.r[i, 1]
            j = 2 * gtr1 + gtr2
            mu = np.vdot(f[i], self.b[j][i])
            sd = np.sqrt(self.v[i, j])
            y[i] = np.random.normal(mu, sd)
        return y

    def forecast_all(self):
        for i in range(self.nfrc):
            s = self.ifrc[i, 0]
            t = self.ifrc[i, 1]
            top = self.x[s - 1, t]
            lft = self.x[s, t - 1]
            if np.isnan(top):
                x1 = self.xhat[self.irev[s - 1, t]]
                x2 = self.xhat[self.irev[s, t - 1]]
            else:
                x1 = np.full(self.nmc, top)
                x2 = np.full(self.nmc, lft)
            self.xhat[i] = self.forecast_one(x1, x2)
            self.irev[s, t] = i
