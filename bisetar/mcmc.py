import numpy as np
from scipy.stats import multivariate_normal as mvn
from scipy.stats import multivariate_t as mvt
from scipy.stats import invgamma
from scipy.stats import norm
from scipy.special import logsumexp
from scipy.special import loggamma
from statsmodels.stats.correlation_tools import cov_nearest
from mcmclib.metropolis import rwm_adapt
from bisetar.discrete import GridDist

class BiSetar:
    def __init__(self, x):
        self.x = x

        # Initial r
        r01 = np.nanmedian(x[:-1, 1:])
        r02 = np.nanmedian(x[1:, :-1])
        r0 = [r01, r02]

        # Initial phi
        xr = self.splitx(r0)
        phi0 = np.empty((4, 4))
        for i in range(4):
            b, sse, c, cinv, n = self.lsfit(xr[i][0], xr[i][1], xr[i][2])
            phi0[i, :3] = b
            phi0[i, 3] = sse / (n - 3)
        phi0 = phi0.flatten()

        # Initial theta
        self.theta0 = np.concatenate((r0, phi0))

    def splitx(self, r):
        # Label each observation
        ler1 = self.x[:-1, 1:] <= r[0]
        ler2 = self.x[1:, :-1] <= r[1]
        i1 = ler1 & ler2
        i2 = ler1 & (~ler2)
        i3 = (~ler1) & ler2
        i4 = (~ler1) & (~ler2)
        # R1
        y1 = self.x[1:, 1:][i1]
        x11 = self.x[:-1, 1:][i1]
        x21 = self.x[1:, :-1][i1]
        # R2
        y2 = self.x[1:, 1:][i2]
        x12 = self.x[:-1, 1:][i2]
        x22 = self.x[1:, :-1][i2]
        # R3
        y3 = self.x[1:, 1:][i3]
        x13 = self.x[:-1, 1:][i3]
        x23 = self.x[1:, :-1][i3]
        # R4
        y4 = self.x[1:, 1:][i4]
        x14 = self.x[:-1, 1:][i4]
        x24 = self.x[1:, :-1][i4]
        # Find upper observations
        u1 = ~np.isnan(y1)
        u2 = ~np.isnan(y2)
        u3 = ~np.isnan(y3)
        u4 = ~np.isnan(y4)

        return ((y1[u1], x11[u1], x21[u1]),
                (y2[u2], x12[u2], x22[u2]),
                (y3[u3], x13[u3], x23[u3]),
                (y4[u4], x14[u4], x24[u4]))

    def lsfit(self, y, x1, x2):
        n = len(y)
        x = np.vstack((np.ones(n), x1, x2)).T
        c = x.T @ x
        cinv = np.linalg.pinv(c)
        b = cinv @ x.T @ y
        e = y - x @ b
        sse = np.dot(e, e)
        return (b, sse, c, cinv, n)


class BiSetarBayes(BiSetar):
    def __init__(self, x):
        super().__init__(x)

    def logp(self, theta):
        # Split data based on r1 and r2
        ((y1, x11, x21),
         (y2, x12, x22),
         (y3, x13, x23),
         (y4, x14, x24)) = self.splitx(theta[:2])

        # Check if each regime contains sufficient observations
        if len(y1) < 8 or len(y2) < 8 or len(y3) < 8 or len(y4) < 8:
            return -np.inf

        # Conditional means
        yhat1 = theta[2] + theta[3] * x11 + theta[4] * x21
        yhat2 = theta[6] + theta[7] * x12 + theta[8] * x22
        yhat3 = theta[10] + theta[11] * x13 + theta[12] * x23
        yhat4 = theta[14] + theta[15] * x14 + theta[16] * x24

        # Log-posterior
        ll1 = norm.logpdf(y1, yhat1, np.sqrt(theta[5]))
        ll2 = norm.logpdf(y2, yhat2, np.sqrt(theta[9]))
        ll3 = norm.logpdf(y3, yhat3, np.sqrt(theta[13]))
        ll4 = norm.logpdf(y4, yhat4, np.sqrt(theta[17]))
        lp = np.sum(ll1) + np.sum(ll2) + \
             np.sum(ll3) + np.sum(ll4) - \
             np.log(theta[5]) - np.log(theta[9]) - \
             np.log(theta[13]) - np.log(theta[17])
        return lp

    def sample_phi(self, r):
        xr = self.splitx(r)
        phi = np.empty((4, 4))
        for i in range(4):
            b, sse, c, cinv, n = self.lsfit(xr[i][0], xr[i][1], xr[i][2])
            phi[i, 3] = invgamma.rvs(a=((n - 3) / 2), scale=(sse / 2))
            phi[i, :3] = mvn.rvs(b, phi[i, 3] * cinv)
        return phi.flatten()

    def apply_cons(self, theta):
        a = theta[:, [2, 6, 10, 14]]
        b = [
            theta[:, [3, 4]],
            theta[:, [7, 8]],
            theta[:, [11, 12]],
            theta[:, [15, 16]]]
        v = theta[:, [5, 9, 13, 17]]
        n = theta.shape[0]
        out = np.empty((n, 4), dtype=bool)
        for i in range(4):
            out[:, i] = np.sum(np.abs(b[i]), axis=1) >= 1
        keep = np.sum(out, axis=1) == 0
        return theta[keep]


class BiSetarMarginal(BiSetarBayes):
    def __init__(self, x):
        self.x = x

        # Default minimum regime size
        n = np.count_nonzero(np.isfinite(x))
        self.m = int(np.clip(0.05 * n, a_min=10, a_max=None))

        # Default sampler configuration
        self.u = np.arange(0.01, 1.0, 0.01)
        self.alpha = [0.3] * 10
        self.epoch = [200] * 10

    def logp_i(self, y, x1, x2):
        b, sse, c, cinv, n = self.lsfit(y, x1, x2)
        t1 = (3 - n) / 2 * np.log(np.pi * sse)
        t2 = loggamma((n - 3) / 2)
        t3 = -0.5 * np.linalg.slogdet(c)[1]
        return t1 + t2 + t3

    def logp(self, r, m=None):
        # Split data based on r1 and r2
        ((y1, x11, x21),
         (y2, x12, x22),
         (y3, x13, x23),
         (y4, x14, x24)) = self.splitx(r)

        # Check if each regime contains sufficient observations
        # This can be considered a prior on r1 and r2
        if m is None:
            m = self.m
        if len(y1) < m or len(y2) < m or len(y3) < m or len(y4) < m:
            return -np.inf

        # Log marginal posterior
        lp1 = self.logp_i(y1, x11, x21)
        lp2 = self.logp_i(y2, x12, x22)
        lp3 = self.logp_i(y3, x13, x23)
        lp4 = self.logp_i(y4, x14, x24)
        return lp1 + lp2 + lp3 + lp4

    def sample_r(self, n, method='grid', m=None):
        q = np.nanquantile(self.x, self.u)
        if m is None:
            logp = self.logp
        else:
            logp = lambda r: self.logp(r, m)
        if method == 'grid':
            gd = GridDist(q, q, logp)
            return gd.sample(n)
        elif method == 'mcmc':
            med = q[int(len(q) / 2)]
            r0 = np.array([med, med])
            return rwm_adapt(
                logp, r0,
                1.0, np.eye(2),
                self.alpha, self.epoch + [n],
                pb=False)[0][-1]
        else:
            raise ValueError('Invalid method')


class BiSetarRwm(BiSetarBayes):
    def __init__(self, x):
        super().__init__(x)

    def sample_r(self, r_old, phi_old, cq):
        r_new = mvn.rvs(r_old, cq)
        theta_new = np.concatenate((r_new, phi_old))
        theta_old = np.concatenate((r_old, phi_old))
        lp_new = self.logp(theta_new)
        lp_old = self.logp(theta_old)
        lu = np.log(np.random.uniform(0, 1))
        if lu <= (lp_new - lp_old):
            r = r_new
            ia = True
        else:
            r = r_old
            ia = False
        return (r, ia)

    def sample_theta(self, n_obs, theta0, cq):
        theta = np.empty((n_obs, len(theta0)))
        n_acc = 0
        theta[0] = theta0
        for i in range(1, n_obs):
            theta[i, :2], ia = self.sample_r(
                theta[i - 1, :2],
                theta[i - 1, 2:],
                cq)
            theta[i, 2:] = self.sample_phi(theta[i, :2])
            n_acc += ia
        return (theta, n_acc / (n_obs - 1))

    def sample_theta_adapt(self, h0, cq0, w, ep):
        n_ep = len(ep)
        theta = n_ep * [None]
        ar = np.empty(n_ep)

        # First epoch
        h = h0
        cq = cq0
        theta[0], ar[0] = self.sample_theta(
            ep[0], self.theta0, h ** 2 * cq)

        for i in range(1, n_ep):
            rs = theta[i - 1][:, :2]
            cq = (1 - w[i - 1]) * cq + w[i - 1] * np.cov(rs.T)
            cq = cov_nearest(cq)
            h = h * np.exp(ar[i - 1] - 0.23)
            theta0_new = theta[i - 1][-1]
            theta[i], ar[i] = self.sample_theta(
                ep[i], theta0_new, h ** 2 * cq)
            print(f'Epoch: {i + 1}/{n_ep}', end='\r')

        return (theta, ar)


class BiSetarIs(BiSetarBayes):
    def __init__(self, x):
        super().__init__(x)

    def tm_logpdf(self, x, mu, sd, df):
        n = mu.shape[0]
        d = mu.shape[1]
        s = sd ** 2 * np.eye(d)
        lpi = mvt.logpdf(mu, loc=x, shape=s, df=df)
        return logsumexp(-np.log(n) + lpi)

    def tm_random(self, mu, sd, df):
        n = mu.shape[0]
        d = mu.shape[1]
        i = np.random.choice(n)
        s = sd ** 2 * np.eye(d)
        return mvt.rvs(mu[i], s, df)

    def jmpdst(self, x):
        dt = np.diff(x, axis=0)
        n2 = np.sum(dt ** 2, axis=1)
        return np.mean(n2)

    def sample_r(self, r_old, phi_old, mu, sd, df):
        r_new = self.tm_random(mu, sd, df)
        theta_new = np.concatenate((r_new, phi_old))
        theta_old = np.concatenate((r_old, phi_old))
        lp_new = self.logp(theta_new)
        lp_old = self.logp(theta_old)
        lq_new = self.tm_logpdf(r_new, mu, sd, df)
        lq_old = self.tm_logpdf(r_old, mu, sd, df)

        lu = np.log(np.random.uniform(0, 1))
        la = (lp_new - lp_old) + (lq_old - lq_new)
        if lu <= la:
            r = r_new
            ia = True
        else:
            r = r_old
            ia = False
        return (r, ia)

    def sample_theta(self, n_obs, theta0, mu, sd, df=4):
        theta = np.empty((n_obs, len(theta0)))
        theta[0] = theta0
        ia = np.empty(n_obs - 1)
        for i in range(1, n_obs):
            theta[i, :2], ia[i - 1] = self.sample_r(
                theta[i - 1, :2],
                theta[i - 1, 2:],
                mu, sd, df)
            theta[i, 2:] = self.sample_phi(theta[i, :2])
        return (theta, np.mean(ia))

    def learn_scale(self, theta0, mu, sd_grid, df=4, n_mc=200):
        n_grid = len(sd_grid)
        ar = np.empty(n_grid)
        for i in range(n_grid):
            theta, ar[i] = self.sample_theta(
                n_mc, theta0, mu, sd_grid[i], df)
            print(f'Epoch: {i + 1}/{n_grid}', end='\r')
        sd = sd_grid[np.argmax(ar)]
        return (sd, ar)


class BiSetarMarginalUpper(BiSetarMarginal):
    def __init__(self, x, offset=-1):
        # Set lower observations to NaNs
        self.x = x.copy()
        n = x.shape[0]
        np.fliplr(self.x)[np.tril_indices(n, k=offset)] = np.nan
        super().__init__(self.x)


class BiSetarRwmUpper(BiSetarRwm):
    def __init__(self, x, offset=-1):
        # Set lower observations to NaNs
        self.x = x.copy()
        n = x.shape[0]
        np.fliplr(self.x)[np.tril_indices(n, k=offset)] = np.nan
        super().__init__(self.x)


class BiSetarIsUpper(BiSetarIs):
    def __init__(self, x, offset=-1):
        # Set lower observations to NaNs
        self.x = x.copy()
        n = x.shape[0]
        np.fliplr(self.x)[np.tril_indices(n, k=offset)] = np.nan
        super().__init__(self.x)
