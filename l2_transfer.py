#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Перелёт НОО -> орбита в окрестности точки L2 системы Солнце-Земля, два импульса.
Первое приближение: круговая ограниченная задача трёх тел (CR3BP).

Дано (Zadanie_neyronke.docx): сухая масса КА 1800 kg; космодром - Восточный;
разгонный блок КВТК выполняет оба импульса перелёта; высота НОО 200 km;
поддержание орбиты - СПД-100ВМ, не более 100 m/s в год; форма и прецессия лунной
орбиты не учитываются (для Луны оставлены круговая орбита в эклиптике).
Тип орбиты у L2 выбирается расчётом: семейство пространственных гало-орбит,
критерий - минимум суммарной характеристической скорости среди орбит, выходящих
из конуса тени Земли (иначе раз в 180 сут - затмение: сброс давления Солнца и
питания прямо бьют по бюджету удержания, а оно по заданию должно быть минимальным).

Геометрия наклонения. Величина первого импульса задаётся интегралом Якоби, т.е. от
наклонения НЕ зависит; наклонение задаёт ПЛОСКОСТЬ трассы. Плоскость НОО содержит
направление Земля->L2 (этого всегда можно достичь азимутом и моментом старта) и
наклонена к эклиптике на I_ECL = |i_LEO - eps_эклиптики| = 28.2 deg - меньшего угла
при i_LEO = 51.6 deg не даёт ни одна площадка. Трасса интегрируется в этой наклонённой
плоскости (3D), цель - гало эклиптической семьи; цену рассогласования плоскостей
измеряет l2_perturb.py, а снимается она выбором гало с наклонением, равным наклонению
трассы - правило Боровина, Ильина и др.: "из рассчитанного множества выбираются те
траектории перелёта, наклонение которых соответствует наклонению орбиты выведения".

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

# --------------------------------------------------- исходные данные задания
H_PARK, M_DRY = 200e3, 1800.0                    # высота НОО, м; сухая масса КА, кг
SITE, I_LEO = "Восточный", 51.6                # наклонение НОО: принято как у прямых
#   стартов с Восточного к МКС и как у «Спектр-РГ» (Боровин и др.: i = 51,611 deg).
#   Широту площадки в расчёте НЕ используем: по открытым источникам она не
#   подтверждена (см. README, «Чего в расчёте нет»).
EPS_ECL = 23.44                                  # наклон эклиптики к экватору
I_ECL = np.deg2rad(abs(I_LEO - EPS_ECL))         # плоскость НОО <-> эклиптики, min
RB = dict(name="КВТК", ISP=445.0, M_DRY=1300.0)  # РБ делает dv1+dv2; ИСП и "сухая"
#   масса РБ - АССУМPTION: ТТД КВТК в ../Sergio_doc не найдены (grep по всем PDF - 0
#   совпадений), заданы параметрически, чувствительность печатается в блоке МАССА.
SKT = dict(name="СПД-100ВМ", ISP=1600.0, F=0.090, CAP_Y=100.0, LIFE_Y=5.0,
           RES_MNs=4.5e6, RES_H=15000.0, RES_IGN=13000.0)   # 36-55.pdf, табл. 3
G0 = 9.80665
GAP_OK_KM = 500.0       # "трасса попала на орбиту": промах по месту, км (3e-4 масштаба
#   перелёта; меньше этого работа прицела не сходится, а остаток снимается коррекцией
#   середины трассы, которую первое приближение не моделирует)
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


def shadow(P):
    """Fraction of an orbit (points x,y,z in synodic units) inside the Earth's umbra /
    penumbra cone, and the minimum distance to the Sun-Earth axis.  Axis = Sun->Earth = +x;
    xi = km behind the Earth, rho = km off the axis."""
    xi = (P[:, 0] - (1 - MU)) * KMD                       # km behind the Earth
    rho = np.hypot(P[:, 1], P[:, 2]) * KMD                # km off the Sun-Earth axis
    back = xi > 0
    r_umb = RE / KM - (696000.0 - RE / KM) / (AU / KM) * xi      # umbra radius, km
    r_pen = RE / KM + (696000.0 + RE / KM) / (AU / KM) * xi      # penumbra radius, km
    umb = back & (rho < np.maximum(r_umb, 0.0))
    pen = back & (rho < r_pen)
    return umb.mean(), pen.mean(), float(np.min(np.where(back, rho, np.inf)))


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

# Radius of the Earth's penumbral cone at the distance of L2 (the umbra cone ends
# earlier, at ~1.38e6 km, so there is no umbra at L2 at all).
RS_KM, R_PEN_L2 = 696000.0, None
R_PEN_L2 = RE / KM + (RS_KM + RE / KM) / (AU / KM) * RHO * KMD


def halo_zmin(tol=2e-6):
    """Минимальная амплитуда Az, при которой гало вовсе не заходит в полутени Земли.
    Порог считается самой орбитой (rho_min зависит и от того, где орбита проходит
    ближе к Земле), а не геометрической оценкой; бинарный поиск, ~9 решений гало."""
    lo, hi = R_PEN_L2 / KMD / 4, 3e-4
    for _ in range(40):
        if hi - lo < tol:
            break
        mid = 0.5 * (lo + hi)
        free = halo_family([mid], n=600)[0]["sh_pen"] == 0   # плотная выборка: окно тени ~11 deg
        lo, hi = (lo, mid) if free else (mid, hi)
    return hi

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
        sh_umb, sh_pen, rho_min = shadow(tr[0:3].T)
        fam.append(dict(z=z, s0=s, T=T, C=jacobi(s), P=tr[0:3].T, V=tr[3:6].T,
                        Ax=np.max(np.abs(tr[0] - X_L2)), rmin=np.min(dist_earth(tr[0:3])),
                        sh_umb=sh_umb, sh_pen=sh_pen, rho_min=rho_min))
        seed = (s, T)
    return fam

# ---------------------------------------------------------------- transfer
E_LEO = np.array([0.0, np.cos(I_ECL), np.sin(I_ECL)])     # unit perp. to Sun-Earth line
X_HAT = np.array([1.0, 0.0, 0.0])                         # Earth -> L2 direction


def dep_state(theta, u):
    """state just after dv1: burn at the point of the parking orbit at angle theta
    (r = R_P), velocity along the parking direction => dv1 = u - V_LEO, no plane change.
    u is the GEOCENTRIC INERTIAL speed; the rotating-frame velocity is
    v_rot = v_inertial - omega x r, omega = z_hat, omega x r = (-y, x, 0)."""
    r_hat = np.cos(theta) * X_HAT + np.sin(theta) * E_LEO
    v_hat = -np.sin(theta) * X_HAT + np.cos(theta) * E_LEO
    rs, re_, ve = primaries(0.0)
    r = re_ + R_P * r_hat
    v_rot = ve + u * v_hat - np.array([-r[1], r[0], 0.0])      # - omega x r, omega = 1
    return np.concatenate([r, v_rot])


def u_for_jacobi(theta, C):
    """Geocentric speed at the burn point giving the Jacobi constant C - solved exactly
    on jacobi(dep_state), because C uses the ROTATING-frame speed while u is inertial
    (the old closed form u = sqrt(2*Omega - C) mixed the two: ~1 m/s of mis-bookkeeping).
    shortcut: monotone in u for every start angle reachable from a 200 km orbit."""
    return brentq(lambda u: jacobi(dep_state(theta, u)) - C,
                  0.0, 3.0 * V_LEO + 0.5, xtol=1e-13, rtol=1e-14)


def closest(interp, ts, Pf, Vf):
    """min |r_SC(t) - r_halo| over the sampled leg -> (t, gap, dv2, point, vel)."""
    Y = interp(ts); P = Y[0:3].T; V = Y[3:6].T
    d = np.sqrt(((P[:, None, :] - Pf[None, :, :])**2).sum(-1))
    i = int(np.argmin(d.min(1))); j = int(np.argmin(d[i]))
    return ts[i], d[i, j], np.linalg.norm(V[i] - Vf[j]), P[i], V[i]


def leg(orb, theta, dC=0.0, tof_max=3.5, n1=200, n2=400, max_step=0.5, must_brake=True):
    """one candidate transfer: coast with Jacobi constant C(orb) - dC.
    must_brake: по заданию второй импульс ТОРМОЗНОЙ, т.е. трасса обязана прибывать с
    избытком энергии над целевой орбитой. Реализовано ограничением ПОСТРОЕНИЯ (dC >= 0),
    а не штрафом: штрафом оптимизатор упирался в стену и уходил на другую цель.
    returns dict with dv1, dv2, gap (km), tof (days), braking and the arrival state."""
    if must_brake:
        dC = max(dC, 0.0)
    u = u_for_jacobi(theta, orb["C"] - dC)
    sol = solve_ivp(rhs, (0, tof_max), dep_state(theta, u), method="DOP853",
                    rtol=1e-9, atol=1e-11, max_step=max_step, dense_output=True)
    ts1 = np.linspace(0.15, tof_max, n1)
    t1, _, _, _, _ = closest(sol.sol, ts1, orb["P"], orb["V"])
    h = (ts1[-1] - ts1[0]) / n1
    ts2 = np.linspace(max(0.15, t1 - h), min(tof_max, t1 + h), n2)
    t2, gap, dv2, arr, arrv = closest(sol.sol, ts2, orb["P"], orb["V"])
    j = int(np.argmin(((orb["P"] - arr)**2).sum(1)))
    v_t, v_o = np.linalg.norm(arrv), np.linalg.norm(orb["V"][j])
    return dict(orb=orb, theta=theta, dC=dC, u=u, dv1=u - V_LEO, dv2=dv2, gap=gap * KMD,
                tof=t2 * DAY, arr=arr, arrv=arrv, v_target=v_o, braking=v_t >= v_o,
                tot=(u - V_LEO + dv2) * V_U * 1000 + gap * KMD / 50.0)


def refine(c, tol=1e-3, must_brake=True, maxiter=120):
    """2-parameter local refinement (launch angle + energy offset) of one coarse seed."""
    r0 = minimize(lambda q: leg(c["orb"], q[0], q[1], must_brake=must_brake)["tot"],
                  [c["theta"], c["dC"]], method="Nelder-Mead",
                  options=dict(xatol=1e-5, fatol=tol, maxiter=maxiter))
    return leg(c["orb"], r0.x[0], r0.x[1], must_brake=must_brake)


def aim_transfer(fam, n_theta=16, tol=1e-3, pool=None, brake_only=True):
    """Coarse scan over the halo family and the launch epoch, then refinement of the
    cheapest candidates.  `tot` in the scan is dv1+dv2 plus a smooth gap/50 penalty: it
    is a SEED, never the criterion.  The design criterion is minimum refined dv among
    eclipse-free members whose refined leg actually lands on the orbit (gap <= GAP_OK_KM)
    and, since the assignment prescribes a BRAKING second impulse, arrives faster than
    the target orbit (braking=True; enforced by construction in leg(), dC >= 0)."""
    if brake_only:                       # re-scan with dC >= 0 so the seed respects the
        for orb in fam:                  # braking requirement and the ordering is honest
            orb["best"] = None
            for ith in range(n_theta):
                r = leg(orb, 2 * np.pi * ith / n_theta)
                if orb["best"] is None or r["tot"] < orb["best"]["tot"]:
                    orb["best"] = r
    tried = []
    for orb in sorted(pool or fam, key=lambda oo: oo["best"]["dv1"] + oo["best"]["dv2"]):
        tried.append(refine(orb["best"], tol, maxiter=30))      # быстрый скрининг цели
    br = [t for t in tried if t["braking"]]
    if not br:                    # тормозного входа в этой части семьи просто нет
        return min(tried, key=lambda x: x["tot"]), None, tried, False, 0
    brs = sorted(br, key=lambda x: x["dv1"] + x["dv2"])
    lead = [t for t in brs[:3] if t["dv1"] + t["dv2"] - brs[0]["dv1"] - brs[0]["dv2"]
            < 30.0 / (V_U * 1000)]        # лидеры: в пределах 30 м/с от минимума
    tried += [refine(t["orb"]["best"], tol) for t in lead]      # полное уточнение
    fin = tried[-len(lead):]
    ok = [t for t in fin if t["gap"] <= GAP_OK_KM and t["braking"]]
    if ok:
        best = min(ok, key=lambda t: t["dv1"] + t["dv2"])
        return best, best, tried, True, len(lead)
    return min(fin, key=lambda x: x["tot"]), None, tried, False, len(lead)


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
    s_b = dep_state(0.0, u)
    assert abs(jacobi(s_b) - C0) < 1e-12, jacobi(s_b) - C0   # импульс назначен точно
    r_ = s_b[0:3]                                           # член omega x r на месте:
    assert abs(s_b[3] - r_[1]) < 1e-9, s_b[3]                # u = 0 -> v_rot = ve - omega x r
    # тень: сама орбита внутри конуса полутени даёт затмение, снаружи - нет
    assert halo_family([0.5 * R_PEN_L2 / KMD], n=120)[0]["sh_pen"] > 0
    assert halo_family([2.0 * R_PEN_L2 / KMD], n=120)[0]["sh_pen"] == 0
    # тень: сам L2 лежит ВНУТРИ конуса полутени, но уже за концом конуса полной тени
    at_l2 = np.array([[X_L2, 0.0, 0.0]])
    assert shadow(at_l2)[1] == 1.0 and shadow(at_l2)[0] == 0.0
    assert shadow(at_l2)[2] == 0.0
    off = np.array([[X_L2, 2e5 / KMD, 0.0]])                 # 200 000 km от оси S-Z
    assert shadow(off)[1] == 0.0 and shadow(off)[0] == 0.0
    assert shadow(np.array([[1 - MU + 1e6 / KMD, 0.0, 0.0]]))[0] == 1.0   # в полной тени
    assert abs(RE / KM / ((696000.0 - RE / KM) / (AU / KM)) - 1.383e6) < 2e3  # конец umbra
    print("self_check OK: v_NOO, L2, C(L2), e=0 предельный случай, замыкание гало,")
    print("  интеграл Якоби, dv1")


def main():
    print("=" * 78)
    print("CR3BP Солнце-Земля:  mu = %.6e   1 TU = %.3f сут   1 VU = %.4f km/s"
          % (MU, DAY, V_U))
    print("  1 а.е. = %.0f km" % (AU / KM))
    print("Точка L2: x = %.8f -> расстояние Земля-L2 = %.0f km   C(L2) = %.7f"
          % (X_L2, RHO * KMD, C_L2))
    print("   контроль: a*(mu/3)^(1/3) = %.0f km; линеаризация: lambda = %.4f,"
          % ((MU / 3)**(1 / 3) * KMD, LAMBDA))
    print("   omega_z = %.4f (вертикальный период %.1f сут)"
          % (W_VERT, 2 * np.pi / W_VERT * DAY))
    print("НОО: r = %.0f km, v_krug = %.4f km/s, v_II косм = %.4f km/s" %
          ((RE + H_PARK) / KM, V_LEO * V_U, V_LEO * np.sqrt(2) * V_U))
    print("     площадка %s: i = %.1f deg (прямой старт), плоскость трассы наклонена"
          % (SITE, I_LEO))
    print("     к эклиптике не более чем на %.1f deg (i - eps_эклиптики)" %
          np.degrees(I_ECL))
    print("=" * 78)

    z_free = halo_zmin()                      # порог по самой орбите, не по геометрии
    zs = np.sort(np.append(np.linspace(3e-5, 9e-4, 10), 2.0 * z_free))
    fam = halo_family(zs, n=480)
    free = [o for o in fam if o["sh_pen"] == 0.0]      # без затмений вовсе
    print("Семейство гало-орбит у L2 (тип орбиты выбирается расчётом):")
    print("    Az, km      Ax, km   период,сут        C      min r(Земля), km  v, km/s"
          "   тень, % орбиты")
    for o in fam:
        print("  %8.0f %10.0f %10.1f  %.7f %16.0f     %6.3f   полн %5.1f  полутень %5.1f"
              % (o["z"] * KMD, o["Ax"] * KMD, o["T"] * DAY, o["C"],
                 o["rmin"] * KMD, np.linalg.norm(o["s0"][3:6]) * V_U,
                 100 * o["sh_umb"], 100 * o["sh_pen"]))
    print("   конус тени: полная тень кончается в %.0f km за Землёй (L2 - %.0f km),"
          % (RE / KM / ((696000.0 - RE / KM) / (AU / KM)), RHO * KMD))
    print("   радиус полутени на расстоянии L2 - %.0f km (угловое условие: разделение"
          % R_PEN_L2)
    print("   центров Солнца и Земли > %.3f deg): в неё попадает любая орбита, у которой"
          % np.degrees((696000.0 + RE / KM) / (AU / KM) + RE / KM / (RHO * KMD)))
    print("   min расстояние до оси Солнце-Земля меньше %.0f km (rho_min ~ Az)."
          % R_PEN_L2)
    print("   порог по расчитанной орбите (бинарный поиск): Az_min = %.0f km."
          % (z_free * KMD))
    print("   за год радиус полутени у L2 меняется на +/- %.0f km (эллиптичность орбиты"
          % ((696000.0 + RE / KM) / (AU / KM) * RHO * KMD * 0.016710))
    print("   Земли): двукратный запас по порогу перекрывает это более чем в 100 раз.")
    print("   без затмений в семье: %s" % (", ".join("%.0f" % (oo["z"] * KMD) for oo in free)
                                           or "нет - самая крупная Az = %.0f km"
                                               % (fam[-1]["z"] * KMD)))
    print("   кандидатом в сетку добавлена орбита Az = 2x Az_min (%.0f km): двукратный"
          % (2 * z_free * KMD))
    print("   запас по тени (год меняет радиус полутени лишь на ±%.0f km)."
          % ((696000.0 + RE / KM) / (AU / KM) * RHO * KMD * 0.016710))
    print("   выбор цели - минимум dv среди тех беззатменных, чья УТОЧНЁННАЯ трасса")
    print("   попадает на орбиту (промах <= %.0f km); суррогат dv+промах/50 служит только"
          % GAP_OK_KM)
    print("   стартом для уточнения, критерием он не является.")
    pool = free or fam          # если затмений не избежать - выбирать по одному dv

    r, coarse, tried, landed, n_fin = aim_transfer(fam, pool=pool)
    o = r["orb"]
    print("-" * 78)
    print("Уточнение кандидатов (скрининг всех беззатменных, затем полное уточнение лидеров):")
    print("  целевая Az, km   dv1+dv2, m/s   промах, km   тоф, сут   тормозн.?  статус")
    for n_t, t in enumerate(tried):
        st = "ВЫБРАНА" if t is r else ("полное" if n_t >= len(tried) - n_fin else "скрининг")
        print("  %12.0f %12.1f %11.0f %9.1f      %-6s     %s"
              % (t["orb"]["z"] * KMD, (t["dv1"] + t["dv2"]) * V_U * 1000, t["gap"], t["tof"],
                 "да" if t["braking"] else "нет", st))
    print("  схема задания (тормозной вход с промахом <= %.0f km) выполнена: %s"
          % (GAP_OK_KM, "да" if landed else "НЕТ"))
    print("-" * 78)
    print("Грубый перебор (лучший по dv ход среди 16 углов старта):")
    print("  целевая Az, km   dv1, m/s   dv2, m/s   промах, km   тень")
    for oo in fam:
        b = oo["best"]
        print("  %14.0f %9.1f %9.1f %12.0f   %s"
              % (oo["z"] * KMD, b["dv1"] * V_U * 1000, b["dv2"] * V_U * 1000, b["gap"],
                 "нет" if oo["sh_pen"] == 0 else "есть"))
    cmin = min(fam, key=lambda o: o["best"]["tot"])
    print("  минимум грубого перебора (dv + штраф за промах, dC >= 0) по всей семье:"
          " Az = %.0f km, dv = %.1f m/s (промах %.0f km)"
          % (cmin["z"] * KMD, (cmin["best"]["dv1"] + cmin["best"]["dv2"]) * V_U * 1000,
             cmin["best"]["gap"]))
    print("-" * 78)
    print("ПЕРЕЛЁТ НОО -> гало у L2 (два импульса):")
    print("  целевая орбита: Az = %.0f km, Ax = %.0f km, период = %.1f сут, C = %.7f"
          % (o["z"] * KMD, o["Ax"] * KMD, o["T"] * DAY, o["C"]))
    print("  затмение на целевой орбите: полная тень %.1f%%, полутень %.1f%% орбиты,"
          " min rho до оси S-Z = %.0f km"
          % (100 * o["sh_umb"], 100 * o["sh_pen"], o["rho_min"]))
    print("  угол старта в плоскости НОО       = %6.1f deg (плоскость содержит"
          " направление на L2)" % np.degrees(r["theta"]))
    print("  скорость после 1-го импульса      = %.4f km/s (НОО %.4f)"
          % (r["u"] * V_U, V_LEO * V_U))
    print("  dv1 (разгонный)                   = %8.1f m/s" % (r["dv1"] * V_U * 1000))
    print("  время активного перелёта           = %8.1f сут" % r["tof"])
    print("  расстояние в момент включения 2-го импульса от Земли = %.0f km"
          % (dist_earth(r["arr"]) * KMD))
    print("  промах по месту прицеливания        = %6.0f km (после уточнения)" % r["gap"])
    print("  избыток энергии перелёта над C(орбиты): dC = %.2e" % r["dC"])
    print("  dv2 (тормозной)                   = %8.1f m/s" % (r["dv2"] * V_U * 1000))
    j_in = int(np.argmin(((o["P"] - r["arr"])**2).sum(1)))
    v_arr, v_orb = np.linalg.norm(r["arrv"]), np.linalg.norm(o["V"][j_in])
    res = r["arrv"] - o["V"][j_in]
    print("  |v| трассы в точке входа = %.4f km/s, |v| орбиты = %.4f km/s -> по модулям %s;"
          " по невязке векторов dv2 = %.1f m/s"
          % (v_arr * V_U, v_orb * V_U,
             "тормозной" if v_arr > v_orb else "разгонный", r["dv2"] * V_U * 1000))
    print("  доля невязки вне эклиптики в dv2 = %.1f%% (dv2 - невязка ВЕКТОРОВ скоростей)"
          % (100 * abs(res[2]) / np.linalg.norm(res)))
    unc = refine(o["best"], must_brake=False)          # что стоит требование "тормозной"
    print("  если требуется только минимум dv (второй импульс может стать разгонным): dv = %.1f m/s (%+.1f), тоф %.1f сут,"
          % ((unc["dv1"] + unc["dv2"]) * V_U * 1000,
             (unc["dv1"] + unc["dv2"] - r["dv1"] - r["dv2"]) * V_U * 1000, unc["tof"]))
    print("  |v| трассы %.4f против |v| орбиты %.4f km/s -> импульс %s."
          % (np.linalg.norm(unc["arrv"]) * V_U, unc["v_target"] * V_U,
             "тормозной" if unc["braking"] else "РАЗГОННЫЙ - схеме задания не отвечает"))
    print("  В проект идёт вариант, удовлетворяющий заданию (второй импульс %s)."
          % ("тормозной" if r["braking"] else "НЕ тормозной - см. предупреждение"))
    if not landed:
        print("  ВНИМАНИЕ: тормозного входа с промахом <= %.0f km найти не удалось; ниже -"
              % GAP_OK_KM)
        print("  лучшая из найденных трасс, но схему задания она НЕ удовлетворяет целиком.")
    print("  СУММАРНАЯ характеристическая скорость dv = %.1f m/s"
          % ((r["dv1"] + r["dv2"]) * V_U * 1000))
    print("  критерий выбора цели: минимум уточнённого dv среди гало БЕЗ ЗАТМЕНИЙ, чья")
    print("  трасса попадает на орбиту (промах %.0f km <= %.0f km) И прибывает быстрее"
          " орбиты (второй импульс - тормозной, dC >= 0): выполнено %s."
          % (r["gap"], GAP_OK_KM, "да" if r["braking"] and r["gap"] <= GAP_OK_KM else "НЕТ"))
    print("  В грубом переборе штраф промах/50 м/с на 1 km - только старт для уточнения.")

    print("-" * 78)
    dv_rb = (r["dv1"] + r["dv2"]) * V_U * 1000      # перелёт целиком делает разгонный блок
    print("МАССА (по заданию: сухая масса КА %.0f kg; перелёт - РБ %s, удержание - %s):"
          % (M_DRY, RB["name"], SKT["name"]))
    print("  dv разгонного блока = dv1 + dv2 = %.0f m/s; с запасом 5%% на бюджет: %.0f m/s"
          % (dv_rb, 1.05 * dv_rb))
    print("  удержание: m_кс = m_сух*(1 - exp(-dv_sk/(Isp_СПД*g0))), Isp_СПД = %.0f s,"
          " тяга %.0f mN:" % (SKT["ISP"], 1000 * SKT["F"]))
    for sk_dv in (10.0, SKT["CAP_Y"], SKT["CAP_Y"] * SKT["LIFE_Y"]):
        m_x = M_DRY * (1 - np.exp(-sk_dv / (SKT["ISP"] * G0)))
        print("    dv_удержания %5.0f m/s -> ксенон %5.1f kg, часы работы %5.0f ч,"
              " импульс %.2f MN*s"
              % (sk_dv, m_x, M_DRY * sk_dv / SKT["F"] / 3600.0,
                 M_DRY * sk_dv / 1e6))
    m_x = M_DRY * (1 - np.exp(-(SKT["LIFE_Y"] * 10.0) / (SKT["ISP"] * G0)))
    m_sc = M_DRY + m_x                              # масса КА на момент включения dv1
    print("  принята заправка ксенонном %.1f kg (удержание ~10 m/s за миссию, см."
          " perturb.txt) -> масса КА %6.0f kg" % (m_x, m_sc))
    mp_nom = (m_sc + RB["M_DRY"]) * (np.exp(dv_rb / (RB["ISP"] * G0)) - 1)
    print("  РБ: m_топл = (m_КА + m_сух_РБ)*(exp(dv/(Isp*g0)) - 1), m_сух_РБ = %.0f kg"
          " (АССУМPTION, ТТД КВТК не подтверждены):" % RB["M_DRY"])
    for isp in (RB["ISP"] - 10, RB["ISP"], RB["ISP"] + 10):
        for dv, tag in ((dv_rb, "расчёт"), (1.05 * dv_rb, "запас 5%")):
            mp = (m_sc + RB["M_DRY"]) * (np.exp(dv / (isp * G0)) - 1)
            print("    Isp_РБ = %3.0f s, dv = %4.0f m/s (%s): топливо РБ %5.0f kg,"
                  " ввести на НОО %5.0f kg"
                  % (isp, dv, tag, mp, m_sc + RB["M_DRY"] + mp))

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
    print("  C coast vs C(орбиты) - dC: %.10f vs %.10f (разность %.1e)"
          % (Cs.mean(), o["C"] - r["dC"],
             Cs.mean() - (o["C"] - r["dC"])))
    print("  min r(Земля) на перелёте = %.0f km, min r(L2) = %.0f km"
          % (np.min(dist_earth(tr[0:3])) * KMD,
             np.min(np.hypot(np.hypot(tr[0] - X_L2, tr[1]), tr[2]) * KMD)))
    print("  оскулирующая орбита отлёта в геоцентрической СК: a = %.0f km, e = %.3f,"
          " апогей = %.0f km" % (a_km, ecc, a_km * (1 + ecc)))
    print("  замыкание гало-орбиты за период: невязка %.1e AU"
          % np.linalg.norm(fly(o["s0"], o["T"], 1e-12)[:3] - o["s0"][:3]))
    print("-" * 78)
    print("ОТВЕТ: dv1 = %.0f m/s (разгонный с НОО), dv2 = %.0f m/s (%s, выход на"
          % (r["dv1"] * V_U * 1000, r["dv2"] * V_U * 1000,
             "тормозной" if r["braking"] else "НЕ тормозной"))
    print("  гало у L2), суммарно dv = %.0f m/s; перелёт %.0f сут; целевая орбита"
          " Az = %.0f km (без затмений)"
          % ((r["dv1"] + r["dv2"]) * V_U * 1000, r["tof"], o["z"] * KMD))
    print("  масса: сухая КА %.0f kg, ксенон %.1f kg, топливо РБ %.0f kg (Isp %.0f s),"
          % (M_DRY, m_x, mp_nom, RB["ISP"]))
    print("  ввести на НОО %.0f kg; dv удержания и проверка лимита 100 m/s/год - в"
          % (m_sc + RB["M_DRY"] + mp_nom))
    print("  perturb.txt, раздел 6 (там же ресурс СПД-100ВМ и допуск, который лимиту равен).")
    print("  Давление Солнца, Луна (круговая орбита в эклиптике - форма и прецессии")
    print("  исключены по заданию), эксцентриситет орбиты Земли, наклонённая плоскость")
    print("  выведения и удержание на гало посчитаны в l2_perturb.py на этой же трассе.")


if __name__ == "__main__":
    import sys
    self_check() if "--check" in sys.argv else main()
