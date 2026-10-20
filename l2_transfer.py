#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Перелёт НОО -> орбита в окрестности точки L2 системы Солнце-Земля, два импульса.
Первое приближение: круговая ограниченная задача трёх тел (CR3BP).

Дано (Zadanie_neyronke.docx): масса КА 2500 kg, наклонение НОО 23.5 deg,
высота НОО 200 km, схема из двух импульсов (разгонный + тормозной).
Тип орбиты у L2 выбирается расчётом: семейство пространственных гало-орбит,
критерий выбора - минимум суммарной характеристической скорости.

Геометрия наклонения. Плоскость НОО наклонена к эклиптике на i = 23.5 deg; долгота
восходящего узла считаем выбранной так, чтобы вектор разгона лежал в плоскости НОО
(это всегда можно сделать моментом старта). Тогда первый импульс коллинеярен скорости
НОО - разворота плоскости нет, и i в бюджет dv не входит. Место включения импульса
смещено из эклиптики на R_p*sin(i) = 2620 km (0.2% масштаба перелёта) - этим пренебрегаем.

Запуск:  python3 l2_transfer.py            (numpy + scipy, ~5 мин)
         python3 l2_transfer.py --check    (проверка ядра модели, ~35 с)
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, fsolve, minimize

# ----------------------------------------------------------------- constants
GM_S, GM_E = 1.32712440018e20, 3.986004418e14          # m3/s2
GM_M = 4.9048695e12                                     # m3/s2 (Moon)
AU, RE, KM = 1.495978707e11, 6371e3, 1e3               # m, m, m
MU = GM_E / (GM_S + GM_E)                               # Earth / (Sun + Earth)
MU_MOON = GM_M / (GM_S + GM_E)                          # Moon GM in synodic units
N_ORB = np.sqrt((GM_S + GM_E) / AU**3)                  # Earth mean motion, 1/s
T_U, V_U = 1.0 / N_ORB, N_ORB * AU / KM                 # CR3BP time (s), vel (km/s)
DAY, KMD = T_U / 86400.0, AU / KM                       # TU in days, AU in km

H_PARK, I_PARK, M_SC = 200e3, np.deg2rad(23.5), 2500.0
R_P = (RE + H_PARK) / AU                                # perigee radius, AU
V_LEO = np.sqrt(GM_E / (RE + H_PARK)) / KM / V_U        # circular speed, synodic

# ------------------------------------------------------------------- CR3BP
# --------------------------------------------------- optional perturbations
# All None = pure CR3BP (the path that produced result.txt).  l2_perturb.py sets them.
SRP = None        # dict(CR=1.3, A_m=area/mass in m2/kg)
MOON = None       # dict(a=AU, e, i=rad, n=sidereal rad/TU, ph=rad, Omdot, wdot  [rad/TU])
SUN_E = None      # dict(e=Earth orbital eccentricity, M0=mean anomaly at t=0)


def moon_pos(t):
    """Moon's geocentric position in the rotating frame: inclined (5.145 deg) ellipse
    whose node regresses (18.6 yr) and whose apsidal line advances (8.85 yr).
    shortcut: true anomaly is 1st order in e (e_M = 0.055 -> < 0.4 deg of position)."""
    m = MOON
    Mn = m["n"] * t + m["ph"]
    nu = Mn + 2 * m["e"] * np.sin(Mn)
    r = m["a"] * (1 - m["e"] * np.cos(Mn))
    w = m.get("w", 0.0) + m.get("wdot", 0.0) * t
    Om = m.get("Om", 0.0) + m.get("Omdot", 0.0) * t
    u = nu + w
    lam = Om + np.arctan2(np.cos(m["i"]) * np.sin(u), np.cos(u))
    bet = np.arcsin(np.sin(m["i"]) * np.sin(u))
    ang = lam - t
    return np.array([r * np.cos(bet) * np.cos(ang),
                     r * np.cos(bet) * np.sin(ang),
                     r * np.sin(bet)])


def kep(t, e, M0=0.0):
    """Earth's true anomaly minus mean anomaly, and radius (relative orbit, a = 1)."""
    Mn = (t + M0) % (2 * np.pi)
    E = Mn
    for _ in range(12):
        E -= (E - e * np.sin(E) - Mn) / (1 - e * np.cos(E))
    nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))
    R = (1 - e * np.cos(E))
    return nu - Mn, R, nu


def primaries(t):
    """(Sun, Earth, Earth inertial velocity) in the rotating frame, which turns at the
    constant mean motion omega = 1.  With e = 0 this is exactly the circular CR3BP."""
    if SUN_E is None:
        return (np.array([-MU, 0.0, 0.0]), np.array([1 - MU, 0.0, 0.0]),
                np.array([0.0, 1 - MU, 0.0]))
    psi, R, nu = kep(t, e=SUN_E["e"], M0=SUN_E.get("M0", 0.0))
    h = np.sqrt(1 - SUN_E["e"]**2)
    nud = h / R**2                                     # nu_dot, angular momentum h
    Rd = R * SUN_E["e"] * np.sin(nu) * nud / (1 + SUN_E["e"] * np.cos(nu))
    c, s = np.cos(psi), np.sin(psi)
    re_ = (1 - MU) * np.array([R * c, R * s, 0.0])
    ve = (1 - MU) * (Rd * np.array([c, s, 0.0]) + R * nud * np.array([-s, c, 0.0]))
    return -MU * np.array([R * c, R * s, 0.0]), re_, ve


def third(p, origin, gm):
    """tidal form: direct attraction minus the same pull on the central (geocentric) body."""
    d = p - origin
    return -gm * (d / np.dot(d, d)**1.5 - origin / np.dot(origin, origin)**1.5)


def accel(s, t=0.0):
    """rotating-frame acceleration; Sun at (-mu,0,0), Earth at (1-mu,0,0), omega=1"""
    rs, re_, _ = primaries(t)
    p = np.array(s[0:3])
    d1, d2 = p - rs, p - re_
    n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
    if SUN_E is None:                                  # exact circular CR3BP form
        x, y = s[0], s[1]
        a = np.array([2 * s[4] + x, -2 * s[3] + y, 0.0])
        a += -(1 - MU) * d1 / n1**3 - MU * d2 / n2**3
    else:                 # frame turns at the MEAN motion while the line of centres does
        a = np.array([2 * s[4] + s[0], -2 * s[3] + s[1], 0.0])   # omega x (omega x r): no z
        a += -(1 - MU) * d1 / n1**3 - MU * d2 / n2**3
    if SRP is not None:                 # radial away from the Sun, ~1/r1^2
        k = 4.54e-6 * SRP["CR"] * SRP["A_m"] / (GM_S / AU**2)     # a_SRP / a_gravSun(1 AU)
        a += k * (1 - MU) * d1 / n1**3
    if MOON is not None:
        # geocentric frame: the Earth is pulled by the Moon too, so only the tidal part
        # survives (dropping the indirect term moves dv2 by ~175 m/s on the same geometry)
        a += third(p - re_, moon_pos(t), MU_MOON)
    return a


def rhs(t, s):
    return np.concatenate([s[3:6], accel(s, t)])


def pot(s, t=0.0):
    """pseudo-potential of the *circular* CR3BP; with SUN_E set it uses the true
    instantaneous primary positions (energy-matching only - C is not conserved then)."""
    rs, re_, _ = primaries(t)
    r1 = np.linalg.norm(s[0:3] - rs)
    r2 = np.linalg.norm(s[0:3] - re_)
    return 0.5 * (s[0]**2 + s[1]**2) + (1 - MU) / r1 + MU / r2


def jacobi(s):
    """Jacobi constant C = 2*U - v_rot^2 (Szebehely convention); conserved on legs."""
    return 2 * pot(s) - np.dot(s[3:6], s[3:6])


def fly(s0, tf, rtol=1e-10):
    return solve_ivp(rhs, (0, tf), np.asarray(s0, float), method="DOP853",
                     rtol=rtol, atol=rtol * 1e-2, max_step=0.5).y[:, -1]


def fly_traj(s0, tf, n=1200, rtol=1e-11):
    return solve_ivp(rhs, (0, tf), np.asarray(s0, float), method="DOP853",
                     rtol=rtol, atol=rtol * 1e-2, max_step=0.5,
                     dense_output=True).sol(np.linspace(0, tf, n))


def dist_earth(p):
    """distance from the Earth's centre, p = (3,) or (3,N) in synodic units."""
    p = np.asarray(p, float)
    one = p.ndim == 1
    if one:
        p = p[:, None]
    d = np.hypot(np.hypot(p[0] - (1 - MU), p[1]), p[2])
    return float(d[0]) if one else d

# ------------------------------------------------------------------------ L2
X_L2 = brentq(lambda x: x - (1 - MU) * (x + MU) / abs(x + MU)**3
              - MU * (x - 1 + MU) / abs(x - 1 + MU)**3, 1 - MU + 1e-4, 1 - MU + 0.05)
RHO = X_L2 - (1 - MU)                                   # Earth -> L2, AU
S_L2 = np.array([X_L2, 0.0, 0.0, 0.0, 0.0, 0.0])
C_L2 = jacobi(S_L2)

_h = 1e-6
_J = np.zeros((3, 6))
for _i in range(6):
    _e = np.zeros(6); _e[_i] = _h
    _J[:, _i] = (accel(S_L2 + _e) - accel(S_L2 - _e)) / (2 * _h)
_EIG = np.linalg.eig(np.vstack([np.hstack([np.zeros((3, 3)), np.eye(3)]), _J]))
LAMBDA = np.max(np.real(_EIG[0]))                                   # unstable root
_OSC = np.abs(np.real(_EIG[0])) < np.abs(np.imag(_EIG[0]))          # oscillatory pair
W_VERT = abs(_EIG[0][_OSC][np.argmax(abs(_EIG[1][2, _OSC]))].imag)  # z-dominant mode

# ------------------------------------------------------------ halo family
def planar_lyapunov(dx):
    """in-plane symmetric orbit about L2, dx = x0 - X_L2 -> (state, period)."""
    x0 = X_L2 + dx
    def F(p):
        sf = fly(np.array([x0, 0, 0, 0, p[0], 0]), p[1], 1e-12)
        return [sf[1], sf[3]]
    p = fsolve(F, [-6.5 * dx, np.pi / 2.06])
    return np.array([x0, 0, 0, 0, p[0], 0]), 2 * p[1]


def halo(z_amp, guess):
    """spatial (halo) orbit: two perpendicular xz-crossings -> unknowns x0, vy, T/2."""
    def F(p):
        sf = fly(np.array([p[0], 0, z_amp, 0, p[1], 0]), p[2], 1e-12)
        return [sf[1], sf[3], sf[5]]
    p = fsolve(F, guess)
    return np.array([p[0], 0, z_amp, 0, p[1], 0]), 2 * p[2]


def halo_family(z_list, n=240):
    """continuation from the planar family (seeded at the vertical resonance) to
    spatial halos; returns one dict per orbit with its sampled points."""
    seed = None
    for dx in (-1.2e-3, -1.4e-3, -1.6e-3, -1.8e-3, -2.0e-3, -2.4e-3):
        s, T = planar_lyapunov(dx)
        if seed is None or abs(T - 2 * np.pi / W_VERT) < abs(seed[1] - 2 * np.pi / W_VERT):
            seed = (s, T)
    fam = []
    for z in z_list:
        s, T = halo(z, [seed[0][0], seed[0][4], seed[1] / 2])
        tr = fly_traj(s, T, n, 1e-12)
        fam.append(dict(z=z, s0=s, T=T, C=jacobi(s), P=tr[0:3].T, V=tr[3:6].T,
                        Ax=np.max(np.abs(tr[0] - X_L2)), rmin=np.min(dist_earth(tr[0:3]))))
        seed = (s, T)
    return fam

# ---------------------------------------------------------------- transfer
E_LEO = np.array([0.0, np.cos(I_PARK), np.sin(I_PARK)])   # unit perp. to Sun-Earth line
X_HAT = np.array([1.0, 0.0, 0.0])                         # Earth -> L2 direction


def dep_state(theta, u):
    """state just after dv1: burn at the point of the parking orbit at angle theta
    (r = R_P), velocity along the parking direction => dv1 = u - V_LEO, no plane change"""
    r_hat = np.cos(theta) * X_HAT + np.sin(theta) * E_LEO
    v_hat = -np.sin(theta) * X_HAT + np.cos(theta) * E_LEO
    rs, re_, ve = primaries(0.0)
    r = re_ + R_P * r_hat
    v_rot = ve + u * v_hat - np.array([0.0, r[0], 0.0])       # - omega x r, omega = 1
    return np.concatenate([r, v_rot])


def u_for_jacobi(theta, C):
    """geocentric speed at the burn point giving the Jacobi constant C."""
    s = dep_state(theta, 0.0)
    return np.sqrt(max(2 * pot(s) - C, 0.0))


def closest(interp, ts, Pf, Vf):
    """min |r_SC(t) - r_halo| over the sampled leg -> (t, gap, dv2, point, vel)."""
    Y = interp(ts); P = Y[0:3].T; V = Y[3:6].T
    d = np.sqrt(((P[:, None, :] - Pf[None, :, :])**2).sum(-1))
    i = int(np.argmin(d.min(1))); j = int(np.argmin(d[i]))
    return ts[i], d[i, j], np.linalg.norm(V[i] - Vf[j]), P[i], V[i]


def leg(orb, theta, dC=0.0, tof_max=3.5, n1=200, n2=400, max_step=0.5):
    """one candidate transfer: coast with Jacobi constant C(orb) - dC.
    returns dict with dv1, dv2, gap (km), tof (days) and the arrival state."""
    u = u_for_jacobi(theta, orb["C"] - dC)
    sol = solve_ivp(rhs, (0, tof_max), dep_state(theta, u), method="DOP853",
                    rtol=1e-9, atol=1e-11, max_step=max_step, dense_output=True)
    ts1 = np.linspace(0.15, tof_max, n1)
    t1, _, _, _, _ = closest(sol.sol, ts1, orb["P"], orb["V"])
    h = (ts1[-1] - ts1[0]) / n1
    ts2 = np.linspace(max(0.15, t1 - h), min(tof_max, t1 + h), n2)
    t2, gap, dv2, arr, arrv = closest(sol.sol, ts2, orb["P"], orb["V"])
    return dict(orb=orb, theta=theta, dC=dC, u=u, dv1=u - V_LEO, dv2=dv2, gap=gap * KMD,
                tof=t2 * DAY, arr=arr, arrv=arrv,
                tot=(u - V_LEO + dv2) * V_U * 1000 + gap * KMD / 50.0)


def aim_transfer(fam, n_theta=16, tol=1e-3):
    """coarse scan over the halo family and the launch epoch, then a 2-parameter
    refinement (launch angle + energy offset) that closes the intercept gap."""
    coarse = None
    for orb in fam:
        for ith in range(n_theta):
            r = leg(orb, 2 * np.pi * ith / n_theta)
            if coarse is None or r["tot"] < coarse["tot"]:
                coarse = r
    r0 = minimize(lambda q: leg(coarse["orb"], q[0], q[1])["tot"],
                  [coarse["theta"], coarse["dC"]], method="Nelder-Mead",
                  options=dict(xatol=1e-5, fatol=tol, maxiter=120))
    return leg(coarse["orb"], r0.x[0], r0.x[1]), coarse


# -------------------------------------------------------------------- report
def self_check():
    """Быстрая проверка ядра модели (без перебора перелётов): python3 l2_transfer.py --check"""
    assert abs(V_LEO * V_U - 7.788) < 0.01, V_LEO * V_U
    assert abs(RHO * KMD - (MU / 3)**(1 / 3) * KMD) / RHO / KMD < 0.01
    assert abs(C_L2 - 3.0008867) < 1e-6, C_L2
    q = np.array([1.002, 0.003, -0.0004, 0.05, -0.02, 0.01])   # elliptic branch must
    a0 = accel(q, 0.7)                                         # degenerate to circular
    globals()["SUN_E"] = dict(e=0.0)
    assert np.allclose(a0, accel(q, 0.7), atol=1e-13)
    globals()["SUN_E"] = None
    o = halo_family([2.5e-4], n=200)[0]                    # one halo orbit
    s, T, C0 = o["s0"], o["T"], o["C"]
    assert np.linalg.norm(fly(s, T, 1e-13)[:3] - s[:3]) * KMD < 1.0   # closure < 1 km
    assert 170 < T * DAY < 195, T * DAY
    tr = fly_traj(s, T / 3, 50, 1e-12)
    assert max(abs(jacobi(tr[:, k]) - C0) for k in range(tr.shape[1])) < 1e-9
    u = u_for_jacobi(0.0, C0)
    assert abs((u - V_LEO) * V_U - 3.19) < 0.05, u           # dv1 ~ 3.2 km/s
    print("self_check OK: v_NOO, L2, C(L2), e=0 предельный случай, замыкание гало,")
    print("  интеграл Якоби, dv1")


def main():
    print("=" * 78)
    print("CR3BP Солнце-Земля:  mu = %.6e   1 TU = %.3f сут   1 VU = %.4f km/s"
          % (MU, DAY, V_U))
    print("Точка L2: x = %.8f -> расстояние Земля-L2 = %.0f km   C(L2) = %.7f"
          % (X_L2, RHO * KMD, C_L2))
    print("   контроль: a*(mu/3)^(1/3) = %.0f km; линеаризация: lambda = %.4f,"
          % ((MU / 3)**(1 / 3) * KMD, LAMBDA))
    print("   omega_z = %.4f (вертикальный период %.1f сут)"
          % (W_VERT, 2 * np.pi / W_VERT * DAY))
    print("НОО: r = %.0f km, v_krug = %.4f km/s, v_II косм = %.4f km/s, i = %.1f deg"
          % ((RE + H_PARK) / KM, V_LEO * V_U, V_LEO * np.sqrt(2) * V_U, np.degrees(I_PARK)))
    print("=" * 78)

    fam = halo_family(np.linspace(3e-5, 9e-4, 10), n=480)
    print("Семейство гало-орбит у L2 (тип орбиты выбирается расчётом):")
    print("    Az, km      Ax, km    период, сут        C       min r(Земля), km  v, km/s")
    for o in fam:
        print("  %8.0f %10.0f %11.1f  %.7f %16.0f     %6.3f"
              % (o["z"] * KMD, o["Ax"] * KMD, o["T"] * DAY, o["C"],
                 o["rmin"] * KMD, np.linalg.norm(o["s0"][3:6]) * V_U))

    r, coarse = aim_transfer(fam)
    o = r["orb"]
    print("-" * 78)
    print("Грубый перебор: dv1 = %.1f m/s, dv2 = %.1f m/s, промах %.0f km (theta %.1f deg)"
          % (coarse["dv1"] * V_U * 1000, coarse["dv2"] * V_U * 1000, coarse["gap"],
             np.degrees(coarse["theta"])))
    print("-" * 78)
    print("ПЕРЕЛЁТ НОО -> гало у L2 (два импульса):")
    print("  целевая орбита: Az = %.0f km, Ax = %.0f km, период = %.1f сут, C = %.7f"
          % (o["z"] * KMD, o["Ax"] * KMD, o["T"] * DAY, o["C"]))
    print("  угол старта в плоскости НОО       = %6.1f deg (долгота узла выбрана под"
          " разгон)" % np.degrees(r["theta"]))
    print("  скорость после 1-го импульса      = %.4f km/s (НОО %.4f)"
          % (r["u"] * V_U, V_LEO * V_U))
    print("  dv1 (разгонный)                   = %8.1f m/s" % (r["dv1"] * V_U * 1000))
    print("  время активного перелёта           = %8.1f сут" % r["tof"])
    print("  расстояние в момент включения 2-го импульса от Земли = %.0f km"
          % (dist_earth(r["arr"]) * KMD))
    print("  промах по месту прицеливания        = %6.0f km (после уточнения)" % r["gap"])
    print("  избыток энергии перелёта над C(орбиты): dC = %.2e" % r["dC"])
    print("  dv2 (тормозной)                   = %8.1f m/s" % (r["dv2"] * V_U * 1000))
    print("  СУММАРНАЯ характеристическая скорость dv = %.1f m/s"
          % ((r["dv1"] + r["dv2"]) * V_U * 1000))

    print("-" * 78)
    print("Масса КА = %.0f kg (по заданию);  m_топл = m0*(1 - exp(-dv/(Isp*g0))):" % M_SC)
    for isp in (220.0, 311.0, 330.0):
        dv = (r["dv1"] + r["dv2"]) * V_U * 1000
        mp = M_SC * (1 - np.exp(-dv / (isp * 9.80665)))
        print("  Isp = %5.0f s  ->  топливо %6.0f kg, сухая масса %6.0f kg"
              % (isp, mp, M_SC - mp))

    print("-" * 78)
    s0 = dep_state(r["theta"], r["u"])
    tr = fly_traj(s0, r["tof"] / DAY, 300, 1e-11)
    Cs = np.array([jacobi(tr[:, k]) for k in range(tr.shape[1])])
    v_km = r["u"] * V_U
    mu_e = GM_E / KM**3
    eps = v_km**2 / 2 - mu_e / ((RE + H_PARK) / KM)
    a_km = -mu_e / (2 * eps)
    ecc = np.sqrt(max(1 + 2 * eps * (R_P * KMD * v_km)**2 / mu_e**2, 0.0))
    print("Контроль расчёта:")
    print("  постоянство интеграла Якоби: min %.10f  max %.10f (дрейф %.1e)"
          % (Cs.min(), Cs.max(), Cs.max() - Cs.min()))
    print("  min r(Земля) на перелёте = %.0f km, min r(L2) = %.0f km"
          % (np.min(dist_earth(tr[0:3])) * KMD,
             np.min(np.hypot(np.hypot(tr[0] - X_L2, tr[1]), tr[2]) * KMD)))
    print("  оскулирующая орбита отлёта в геоцентрической СК: a = %.0f km, e = %.3f,"
          " апогей = %.0f km" % (a_km, ecc, a_km * (1 + ecc)))
    print("  замыкание гало-орбиты за период: невязка %.1e AU"
          % np.linalg.norm(fly(o["s0"], o["T"], 1e-12)[:3] - o["s0"][:3]))
    print("-" * 78)
    print("ОТВЕТ: dv1 = %.0f m/s (разгонный с НОО), dv2 = %.0f m/s (тормозной, выход на"
          % (r["dv1"] * V_U * 1000, r["dv2"] * V_U * 1000))
    print("  гало у L2), суммарно dv = %.0f m/s; перелёт %.0f сут; топливо %.0f-%.0f kg"
          % ((r["dv1"] + r["dv2"]) * V_U * 1000, r["tof"],
             M_SC * (1 - np.exp(-(r["dv1"] + r["dv2"]) * V_U * 1000 / (330 * 9.80665))),
             M_SC * (1 - np.exp(-(r["dv1"] + r["dv2"]) * V_U * 1000 / (220 * 9.80665)))))
    print("  Давление Солнца, Луна (наклонение, эксцентриситет, прецессии узла и апсид),")
    print("  эксцентриситет орбиты Земли посчитаны в l2_perturb.py на этой же трассе;")
    print("  там же оценки J2/площадки и удержания на гало (perturb.txt). Влияние на")
    print("  dv1: SRP и Луна - ~0, эксцентриситет Земли +33 м/с; всё остальное в dv2.")


if __name__ == "__main__":
    import sys
    self_check() if "--check" in sys.argv else main()
