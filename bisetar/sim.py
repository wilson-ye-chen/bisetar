import numpy as np

def sim_bdsetar(omg, a1, b1, ss, r1, r2, n):
    ''' Usage example:
    omg = [0, 0, 0, 0]
    a1 = [0.6, 0.3, 0.2, 0.1]
    b1 = [0.1, 0.3, 0.5, 0.4]
    ss = [400, 400, 400, 400]
    r1 = 10
    r2 = 10
    n = 50
    x = sim_bdsetar(omg, a1, b1, s, r1, r2, n)
    '''

    x = np.empty((n, n))
    x[0, 0] = np.random.normal(omg[0], np.sqrt(ss[0]))

    # Set the initial values for x(s,0)
    for i in range(1, n):
        if x[i - 1, 0] <= r1:
            reg = 0
        elif x[i - 1, 0] > r1:
            reg = 2

        mn = omg[reg] + a1[reg] * x[i - 1, 0]
        sd = np.sqrt(ss[reg])
        x[i, 0] = np.random.normal(mn, sd)

    # Set the initial values for x(0,t)
    for i in range(1, n):
        if x[0, i - 1] <= r2:
            reg = 0
        elif x[0, i - 1] > r2:
            reg = 1

        mn = omg[reg] + b1[reg] * x[0, i - 1]
        sd = np.sqrt(ss[reg])
        x[0, i] = np.random.normal(mn, sd)

    # For each x(s,t)
    for i in range(1, n):
        for j in range(1, n):
            if x[i - 1, j] <= r1 and x[i, j - 1] <= r2:
                reg = 0
            elif x[i - 1, j] <= r1 and x[i, j - 1] > r2:
                reg = 1
            elif x[i - 1, j] > r1 and x[i, j - 1] <= r2:
                reg = 2
            elif x[i - 1, j] > r1 and x[i, j - 1] > r2:
                reg = 3

            # Simulate based on the parameters
            mn = omg[reg] + a1[reg] * x[i - 1, j] + b1[reg] * x[i, j - 1]
            sd = np.sqrt(ss[reg])
            x[i, j] = np.random.normal(mn, sd)

    return x
