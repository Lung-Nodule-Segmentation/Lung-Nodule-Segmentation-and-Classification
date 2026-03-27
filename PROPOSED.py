import numpy as np
import time


def polr(A, R0, N, t, MaxIt, r):
    th = (1 + t / MaxIt) * A * np.pi * np.random.rand(N)
    R = (r - t / MaxIt) * R0 * np.random.rand(N)
    xR = R * np.sin(th)
    yR = R * np.cos(th)
    xR = xR / np.max(np.abs(xR))
    yR = yR / np.max(np.abs(yR))
    return xR, yR


def Levy(d):
    beta = 1.5
    sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (
                np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.randn(d) * sigma
    v = np.random.randn(d)
    step = u / np.abs(v) ** (1 / beta)
    return step


def PROPOSED(Xpos, fobj, low, high, Tmax):
    # Position update is done at line 43
    [N, dim] = Xpos.shape
    Xbestcost = np.inf
    Xbestpos = np.random.rand(dim)  # Initialize Xbestpos with random values
    Xcost = np.zeros(N)

    for i in range(N):
        Xpos[i, :] = low[i] + (high[i] - low[i]) * np.random.rand(dim)
        Xcost[i] = fobj(Xpos[i, :])
        if Xcost[i] < Xbestcost:
            Xbestpos = Xpos[i, :]
            Xbestcost = Xcost[i]

    A = 15
    # R0 = 0.5
    # Update the position using the below R0
    R0 = np.min(Xcost) / np.sqrt(np.min(Xcost) ** 1.5 + np.max(Xcost) ** 1.5)
    r = 1.5

    Convergence_curve = np.zeros(Tmax)
    ct = time.time()

    for t in range(Tmax):
        Xmean = np.mean(Xpos, axis=0)
        TF = 1 + np.sin(2.5 - t / Tmax)

        for i in range(N):
            levy = Levy(dim)
            Xnewpos = Xbestpos + (Xmean - Xpos[i, :]) * levy * TF
            Xnewpos = np.maximum(Xnewpos, low[i])
            Xnewpos = np.minimum(Xnewpos, high[i])
            Xnewcost = fobj(Xnewpos)

            if Xnewcost < Xcost[i]:
                Xpos[i, :] = Xnewpos
                Xcost[i] = Xnewcost
                if Xcost[i] < Xbestcost:
                    Xbestpos = Xpos[i, :]
                    Xbestcost = Xcost[i]

        for i in range(N - 1):
            aa = np.random.permutation(N)
            Xpos = Xpos[aa, :]
            Xcost = Xcost[aa]

            x, y = polr(A, R0, N, t, Tmax, r)
            StepSize = Xpos[i, :] - Xmean
            Xnewpos = Xbestpos + (y[i] + x[i]) * StepSize
            Xnewpos = np.maximum(Xnewpos, low[i])
            Xnewpos = np.minimum(Xnewpos, high[i])
            Xnewcost = fobj(Xnewpos)

            if Xnewcost < Xcost[i]:
                Xpos[i, :] = Xnewpos
                Xcost[i] = Xnewcost
                if Xcost[i] < Xbestcost:
                    Xbestpos = Xpos[i, :]
                    Xbestcost = Xcost[i]

        Xmean = np.mean(Xpos, axis=0)
        TF = 1 + 0.5 * np.sin(2.5 - t / Tmax)

        for i in range(N):
            b = np.random.permutation(N)
            Xpos = Xpos[b, :]
            Xcost = Xcost[b]
            x, y = polr(A, R0, N, t, Tmax, r)
            alpha = (np.sin(2.5 - t / Tmax) ** 2)
            G = 2 * (1 - (t / Tmax))
            StepSize1 = 1 * Xpos[i, :] - TF * Xmean
            StepSize2 = G * Xpos[i, :] - TF * Xbestpos
            Xnewpos = alpha * Xbestpos + x[i] * StepSize1 + y[i] * StepSize2
            Xnewpos = np.maximum(Xnewpos, low[i])
            Xnewpos = np.minimum(Xnewpos, high[i])
            Xnewcost = fobj(Xnewpos)

            if Xnewcost < Xcost[i]:
                Xpos[i, :] = Xnewpos
                Xcost[i] = Xnewcost
                if Xcost[i] < Xbestcost:
                    Xbestpos = Xpos[i, :]
                    Xbestcost = Xcost[i]

        Convergence_curve[t] = Xbestcost

    Cost = Xbestcost
    Pos = Xbestpos
    ct = time.time() - ct
    return Cost, Convergence_curve, Pos, ct
