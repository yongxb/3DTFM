import numpy as np
from scipy import optimize

from numba import prange


def l_curve(u, sm, b):
    npoints = 200
    smin_ratio = 16 * np.finfo(float).eps

    m, n = u.shape
    sm = sm[:, np.newaxis]
    p, ps = sm.shape

    beta = np.matmul(u.T, b)

    if ps == 1:
        s = sm
        beta = beta[:p]
    else:
        s = sm[p - 1::-1, 0] / sm[p - 1::-1, 1]
        beta = beta[p - 1::-1]

    xi = beta[:p] / s
    xi[np.isinf(xi)] = 0

    eta = np.zeros((npoints, 1), dtype=np.float64)
    rho = np.zeros((npoints, 1), dtype=np.float64)
    reg_param = np.zeros((npoints, 1), dtype=np.float64)
    reg_param[-1] = max(s[p - 1], s[0] * smin_ratio)
    # reg_param[-1] = s[0]*smin_ratio
    ratio = (s[0] / reg_param[-1]) ** (1 / (npoints - 1))

    for i in range(npoints - 2, -1, -1):
        reg_param[i] = ratio * reg_param[i + 1]

    s2 = s ** 2
    for i in range(npoints):
        f = s2 / (s2 + reg_param[i] ** 2)
        eta[i] = np.linalg.norm(f * xi)
        rho[i] = np.linalg.norm((1 - f) * beta[:p])
    del s2

    beta2 = np.linalg.norm(b) ** 2 - np.linalg.norm(beta) ** 2
    del beta

    if m > n and beta2 > 0:
        rho = np.sqrt(rho ** 2 + beta2)
    del beta2

    reg_corner, rho_c, eta_c = l_corner(rho, eta, reg_param, u, sm, b)

    return reg_corner, rho, eta, reg_param


def l_corner(rho, eta, reg_param, u, s, b):
    m, n = u.shape
    p, ps = b.shape

    beta = np.matmul(u.T, b)
    b0 = b - np.matmul(u, beta)
    if ps == 2:
        s = s[p - 1::-1, 0] / s[p - 1::-1, 1]
        beta = beta[p - 1::-1]

    xi = beta / s
    if m > n:
        beta = np.append(beta, np.linalg.norm(b0))

    g = lcfun(reg_param, s, beta, xi)
    gi = np.argmin(g)

    reg_c = optimize.fminbound(lcfun, reg_param[min(gi, len(g) - 1)], reg_param[max(gi - 1, 0)], args=(s, beta, xi))
    kappa_max = -lcfun(reg_c, s, beta, xi)

    if kappa_max < 0:
        lr = len(rho)
        reg_c = reg_param[lr - 1]

        rho_c = rho[lr - 1]
        eta_c = eta[lr - 1]
    else:
        f = s ** 2 / (s ** 2 + reg_c ** 2)
        eta_c = np.linalg.norm(f * xi)
        rho_c = np.linalg.norm((1 - f) * beta[:len(f)])
        if m > n:
            rho_c = np.sqrt(rho_c ** 2 + np.linalg.norm(b0) ** 2)

    return reg_c, rho_c, eta_c


def lcfun(reg_param, s, beta, xi):
    phi = np.zeros(reg_param.shape, dtype=np.float64)
    dphi = np.zeros(reg_param.shape, dtype=np.float64)
    psi = np.zeros(reg_param.shape, dtype=np.float64)
    dpsi = np.zeros(reg_param.shape, dtype=np.float64)
    eta = np.zeros(reg_param.shape, dtype=np.float64)
    rho = np.zeros(reg_param.shape, dtype=np.float64)
    # print(beta)
    if len(beta) > len(s):
        LS = True
        rhoLS2 = beta[-1] ** 2
        beta = beta[:-1]
        beta = beta[:, np.newaxis]
    else:
        LS = False

    for i in prange(len(reg_param)):
        f = s ** 2 / (s ** 2 + reg_param[i] ** 2)

        cf = 1 - f
        eta[i] = np.linalg.norm(f * xi)
        rho[i] = np.linalg.norm(cf * beta)
        f1 = -2 * f * cf / reg_param[i]
        f2 = -f1 * (3 - 4 * f) / reg_param[i]
        phi[i] = np.sum(f * f1 * np.abs(xi) ** 2)
        psi[i] = np.sum(cf * f1 * np.abs(beta) ** 2)
        dphi[i] = np.sum((f1 ** 2 + f * f2) * np.abs(xi) ** 2)
        dpsi[i] = np.sum((-f1 ** 2 + cf * f2) * np.abs(beta) ** 2)

    if LS:
        rho = np.sqrt(rho ** 2 + rhoLS2)

    deta = phi / eta
    drho = -psi / rho
    ddeta = dphi / eta - deta * (deta / eta)
    ddrho = -dpsi / rho - drho * (drho / rho)

    dlogeta = deta / eta
    dlogrho = drho / rho
    ddlogeta = ddeta / eta - dlogeta ** 2
    ddlogrho = ddrho / rho - dlogrho ** 2

    g = - (dlogrho * ddlogeta - ddlogrho * dlogeta) \
        / (dlogrho ** 2 + dlogeta ** 2) ** 1.5

    return g


def tikhonov(U, s, V, b, reg_param):
    if min(reg_param) < 0:
        print('Illegal regularization parameter lambda')

    # m = U.shape[0]
    n = V.shape[0]

    s = s[:, np.newaxis]
    p, ps = s.shape

    beta = np.matmul(U[:, :p].T, b)
    zeta = s[:, 0, np.newaxis] * beta
    # del beta

    ll = len(reg_param)
    x_lambda = np.zeros((n, ll), dtype=np.float64)
    rho = np.zeros((ll, 1), dtype=np.float64)
    eta = np.zeros((ll, 1), dtype=np.float64)

    if ps == 1:
        for i in range(ll):
            x_lambda[:, i] = np.squeeze(np.matmul(V[:, :p], np.divide(zeta, (s ** 2 + reg_param[i] ** 2))))
            rho[i] = reg_param[i] ** 2 * np.linalg.norm(beta / (s ** 2 + reg_param[i] ** 2))
            eta[i] = np.linalg.norm(x_lambda[:, i])

    #         if m > p:
    #             temp = np.append(beta, np.matmul(U[:,p:n].T, b))
    #             rho = np.sqrt(rho**2 + np.linalg.norm(b - np.matmul(U[:,:n], temp**2)))

    return x_lambda, rho, eta


def picard(u, s, b, d=0):
    n = s.shape[0]
    beta = np.abs(np.matmul(u[:, :n].T, b))
    eta = np.zeros((n, 1))

    d21 = 2 * d + 1
    keta = np.arange(0, n)

    for i in keta:
        eta[i] = (np.prod(beta[i - d:i + d + 1]) ** (1 / d21)) / s[i]

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6), dpi=300)
    plt.semilogy(keta[::10], s[::10], '.-', markersize=0.1)
    plt.semilogy(keta[::10], beta[::10], 'x', markersize=0.1)
    plt.semilogy(keta, eta, '+', markersize=0.1)