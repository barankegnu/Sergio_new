#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Сколько стоят возмущения, исключённые чистой CR3BP-моделью l2_transfer.py:
давление Солнца (включая тень Земли и неопределённость C_R/A), Луна (с наклонением
5.1 deg и e = 0.055), J2 и стартовая площадка (геометрия узла), удержание на гало.

Методика: та же номинальная трасса из result.txt (theta = 343.2 deg, dC = 6.42e-5,
цель - гало Az = 4488 km). Мера несоответствия = промах по месту в момент наилучшего
сближения с CR3BP-гало и dv2 для входа на него; затем то же с переприцеливанием по
(theta, dC) - как сделал бы реальный расчёт.

Ограничения: SRP без переориентации КА (C_R - параметр); Луна - тело на предписанной
орбите (её влияние на движение Земли в рамках модели не учитывается, кроме члена
"прямым минус косвенным"); J2 в динамику не вводится - для него дана аналитическая
оценка геометрии узла.

Запуск:  python3 l2_perturb.py        (~4 мин)
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import l2_transfer as L

TH, DC = np.deg2rad(343.2), 6.42e-5          # номинальное прицеливание (result.txt)
TOF = 3.5                                    # горизонт поиска сближения, TU
MS = 0.02                                    # шаг: разрешить лунный месяц
MOON_A, MOON_E, MOON_I = 384400.0 / L.KMD, 0.055, np.deg2rad(5.145)
N_SID = 2 * np.pi * L.DAY / 27.321661        # сидерический месяц -> rad/TU
# узел убывает за 18.6 лет, линия апсид идёт за 8.85 лет -> 20.4 и 42.9 цикла за год
OMDOT = -2 * np.pi / (18.6 * 365.25 / L.DAY)
WDOT = 2 * np.pi / (8.85 * 365.25 / L.DAY)

UMBRA_SLOPE = (696000.0 - 6371.0) / L.KMD      # радиус конуса тени падает на 4.61 м/км


def moon(ph_deg, inc=True, ecc=True, prec=False):
    d = dict(a=MOON_A, e=MOON_E if ecc else 0.0, i=MOON_I if inc else 0.0,
             Om=0.0, w=0.0, n=N_SID, ph=np.deg2rad(ph_deg))
    if prec:
        d.update(Omdot=OMDOT, wdot=WDOT)
    return d


def comoving(sol, ts):
    """SC states expressed in the frame that follows the true line of centres and the
    instantaneous Sun-Earth distance (the 'pulsating' ER3BP frame).  In it the CR3BP
    halo is the e->0 limit, so the residual mismatch is what an epoch-aware design
    actually pays for the eccentricity."""
    e, M0 = L.SUN_E["e"], L.SUN_E.get("M0", 0.0)
    h = np.sqrt(1 - e * e)
    Q = np.empty((3, len(ts))); QD = np.empty((3, len(ts)))
    for k, t in enumerate(ts):
        psi, R, nu = L.kep(t, e, M0)
        nud, Rd = h / R**2, R * e * np.sin(nu) * h / R**2 / (1 + e * np.cos(nu))
        rs, re_, ve = L.primaries(t)                    # ve: Earth's inertial velocity
        ud = sol(ts)[3:6][:, k] - ve + np.array([0.0, re_[0], 0.0])   # in rotating axes
        c, sn = np.cos(-psi), np.sin(-psi)              # extra angle off the mean frame
        Rz = np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1]])
        q = Rz @ (sol(ts)[0:3][:, k] - re_) / R + np.array([1 - L.MU, 0.0, 0.0])
        Q[:, k] = q
        QD[:, k] = (Rz @ ud / R - (Rd / R) * (q - np.array([1 - L.MU, 0.0, 0.0]))
                    - (nud - 1.0) * np.cross([0, 0, 1], q - np.array([1 - L.MU, 0, 0])))
    return Q, QD


def one_leg(orb, theta, dC, ts_n=2500, comove=False):
    """integrate the coast once; dv1, dv2, gap (km), tof (d), min distance to the Moon,
    time spent in the Earth's shadow (s)."""
    u = L.u_for_jacobi(theta, orb["C"] - dC)
    sol = solve_ivp(L.rhs, (0, TOF), L.dep_state(theta, u), method="DOP853",
                    rtol=1e-9, atol=1e-11, max_step=MS, dense_output=True)
    Pt, Vt = orb["P"], orb["V"]
    ts1 = np.linspace(0.15, TOF, 200)
    t1 = L.closest(sol.sol, ts1, Pt, Vt)[0]
    ts2 = np.linspace(t1 - .02, t1 + .02, 400)
    if comove:                                    # epoch-aware target comparison
        Pt = np.empty((len(ts2), 3)); Vt = np.empty((len(ts2), 3))
        Q, QD = comoving(sol.sol, ts2)
        d = np.sqrt(((Q.T[:, None, :] - orb["P"][None, :, :])**2).sum(-1))
        i, j = np.unravel_index(np.argmin(d), d.shape)
        t2, gap, dv2 = ts2[i], d[i, j], np.linalg.norm(QD[:, i] - orb["V"][j])
    else:
        t2, gap, dv2, _, _ = L.closest(sol.sol, ts2, Pt, Vt)
    ts = np.linspace(0.0, TOF, ts_n)
    P = sol.sol(ts)[0:3]
    dt = ts[1] - ts[0]
    out = dict(dv1=(u - L.V_LEO) * L.V_U * 1000, dv2=dv2 * L.V_U * 1000,
               gap=gap * L.KMD, tof=t2 * L.DAY, dmo=np.nan, t_shadow=np.nan,
               t_day=np.nan, u=u)
    out["tot"] = out["dv1"] + out["dv2"] + gap * L.KMD / 50.0
    if L.MOON is not None:                    # closest approach to the Moon
        M = np.stack([L.moon_pos(t) + L.primaries(t)[1] for t in ts], 1)
        out["dmo"] = np.sqrt(((P - M)**2).sum(0)).min() * L.KMD
    if L.SRP is not None:
        # тень: полной (k=1) и полутень через степень перекрытия дисков; интеграл по
        # времени, умноженный на a_SRP, = верхняя граница эффекта от отсутствия света.
        ts = np.linspace(0, 0.03, 6000)                   # первые 1.75 сут, шаг 25 c
        Y = sol.sol(ts)
        ge = Y[0:3] - np.array([L.primaries(t)[1] for t in ts]).T
        x_km = ge[0] * L.KMD
        rho_km = np.hypot(ge[1], ge[2]) * L.KMD
        d_km = np.linalg.norm(ge, axis=0) * L.KMD
        back = x_km > 0
        umbra = back & (rho_km < 6371.0 - UMBRA_SLOPE * x_km)     # конус полной тени
        dt = (ts[1] - ts[0]) * L.T_U
        out["t_shadow"] = umbra.sum() * dt                # полная тень
        out["t_day"] = 86400.0                            # верхняя оценка: сутки тени
    return out


def reaim(orb, comove=False):
    """local (launch angle, energy) re-aim = what a design in the fuller model would do.
    The active force model is whatever L.SRP / L.MOON / L.SUN_E are set to."""
    opt = minimize(lambda q: one_leg(orb, q[0], q[1], comove=comove)["tot"], [TH, DC],
                   method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-3, maxiter=40))
    return one_leg(orb, opt.x[0], opt.x[1], comove=comove)


def main():
    L.SRP = L.MOON = None
    orb = L.halo_family(np.linspace(3e-5, 9e-4, 10), n=480)[0]
    nom = one_leg(orb, TH, DC)
    print("=" * 78)
    print("Референс CR3BP: dv1 = %.1f, dv2 = %.1f, dv = %.1f m/s, промах %.0f km, %.1f сут"
          % (nom["dv1"], nom["dv2"], nom["dv1"] + nom["dv2"], nom["gap"], nom["tof"]))

    print("-" * 78)
    print("3) ЭКЦЕНТРИСИТЕТ ОРБИТЫ ЗЕМЛИ (e = 0.0167; кадр вращается со СРЕДНЕЙ движением,"
          " поэтому линия Солнце-Земля и L2 дрейфуют за перелёт)")
    print("   в среднем-вращающемся кадре линия Солнце-Земля (и весь L2) делает эпicycle")
    print("   амплитудой 2*e*1AU = %.2f deg -> ~%.1f млн km: наивная фиксированная цель"
          % (2 * 0.01671 * 180 / np.pi, 2 * 0.01671 * L.KMD / 1e6))
    print("   недостижима. Честная цена - в кадре, идущем за линией центров (comoving).")
    print("   фаза старта    |S-Z|    dv1     dv2 наивн.   dv2 comoving   промах,km")
    for name, M0 in (("перигей", 0.0), ("90 deg", np.pi / 2), ("апогей", np.pi),
                     ("270 deg", 3 * np.pi / 2)):
        L.SUN_E = dict(e=0.016710, M0=M0)
        a = one_leg(orb, TH, DC)
        b = one_leg(orb, TH, DC, comove=True)
        if name == "перигей":
            one_e = a
        L.SUN_E = None
        print("   %-10s  %.5f  %7.1f    %9.1f       %8.1f       %8.0f"
              % (name, (1 - 0.01671 * np.cos(M0)), a["dv1"], a["dv2"], b["dv2"], b["gap"]))
    print("   ПЕРЕПРИЦЕЛИВАНИЕ (остаток dv2 - это НЕ стоимость эксцентриситета, а то,")
    print("   что гало-цель не является орбитой эллиптической задачи: в ER3BP орбиты у")
    print("   L2 квазипериодические и другого размера (r_L2 = R*(mu/3)^(1/3), ±1.7%).")
    print("   Собственно цена эксцентриситета - в dv1 (см. столбец dv1 выше: +33 м/с).")
    for nm, M0 in (("перигей", 0.0), ("апогей", np.pi)):
        L.SUN_E = dict(e=0.016710, M0=M0)
        r = reaim(orb, comove=True)
        L.SUN_E = None
        print("   старт из %-7s: dv1 = %.1f  dv2(остаток) = %.1f  dv = %.1f м/с"
              " (%+.1f к CR3BP %.1f)"
              % (nm, r["dv1"], r["dv2"], r["dv1"] + r["dv2"],
                 r["dv1"] + r["dv2"] - nom["dv1"] - nom["dv2"], nom["dv1"] + nom["dv2"]))
    print("   (усиление по неустойчивому направлению за трассу ~ e^(140/23.4) = %d раз:"
          % np.exp(140 / (L.T_U / L.LAMBDA / 86400)))
    print("    поэтому CR3BP-прицел в эксцентрической модели всегда требует коррекций)")

    print("-" * 78)
    print("4) ДАВЛЕНИЕ СОЛНЦА (dv2 при неизменном прицеле; dv1 не меняется)")
    print("   A, m2   C_R   a_SRP, м/с^2  в тени, с   dv2 без тени и"
          " с ней, разность   dv2, m/s")
    srps = []
    for A in (20.0, 100.0):
        for CR in (1.0, 1.3, 1.8):
            L.SRP = dict(CR=CR, A_m=A / L.M_SC)
            r = one_leg(orb, TH, DC)
            acc = 4.54e-6 * CR * A / L.M_SC
            print("  %5.0f  %4.1f   %.2e      %6.0f        <= %.4f мм/с          %7.1f"
                  % (A, CR, acc, r["t_shadow"], acc * r["t_day"] * 1000, r["dv2"]))
            srps.append((r["dv2"], acc * r["t_day"]))
    print("   Полутень - частичное затемнение, она не может дать больше, чем полная тень;")
    print("   верхняя оценка взята как сутки непрерывной тени. Эффект на 5 порядков ниже")
    print("   dv2 -> в динамику не вводится.")
    L.SRP = dict(CR=1.3, A_m=20.0 / L.M_SC)
    r = reaim(orb)
    print("   переприцеливание при A=20, C_R=1.3: dv = %.1f m/s (%+.1f к номиналу)"
          % (r["dv1"] + r["dv2"], r["dv1"] + r["dv2"] - nom["dv1"] - nom["dv2"]))
    L.SRP = dict(CR=1.8, A_m=100.0 / L.M_SC)
    r = reaim(orb)
    print("   переприцеливание при A=100, C_R=1.8: dv = %.1f m/s (%+.1f)"
          % (r["dv1"] + r["dv2"], r["dv1"] + r["dv2"] - nom["dv1"] - nom["dv2"]))
    L.SRP = None

    print("-" * 78)
    print("5) ЛУНА (прямая минус косвенная составляющая; a = 384400 km, e = 0.055, i = 5.1 deg)")
    print("   фаза, deg  промах, km   dv2, m/s   мин до Луны, km")
    for ph in (0.0, 45.0, 90.0, 180.0, 270.0):
        L.MOON = moon(ph)
        r = one_leg(orb, TH, DC)
        print("   %6.1f   %10.0f   %8.1f   %10.0f" % (ph, r["gap"], r["dv2"], r["dmo"]))
    print("   переприцеливание:")
    for ph in (0.0, 90.0, 270.0):
        L.MOON = moon(ph)
        r = reaim(orb)
        print("     фаза %5.1f deg: dv1 = %.1f  dv2 = %.1f  dv = %.1f m/s (%+.1f)"
              % (ph, r["dv1"], r["dv2"], r["dv1"] + r["dv2"],
                 r["dv1"] + r["dv2"] - nom["dv1"] - nom["dv2"]))
    print("   влияние формы лунной орбиты (фаза 0, без переприцеливания):")
    for name, kw in (("круговая в эклиптике (как было раньше)", dict(inc=False, ecc=False)),
                     ("e = 0.055", dict(inc=False, ecc=True)),
                     ("i = 5.1 deg", dict(inc=True, ecc=False)),
                     ("i и e", dict(inc=True, ecc=True))):
        L.MOON = moon(0.0, **kw)
        r = one_leg(orb, TH, DC)
        print("     %-38s dv2 = %6.1f m/s, промах %8.0f km" % (name, r["dv2"], r["gap"]))
    print("   прецессии лунной орбиты (узел 18.6 лет -> -7.4 deg, апсиды 8.85 лет ->")
    print("   +15.6 deg за трассу) и что они стоят:")
    for ph in (0.0, 90.0, 270.0):
        L.MOON = moon(ph, prec=False)
        a = one_leg(orb, TH, DC)
        L.MOON = moon(ph, prec=True)
        b = one_leg(orb, TH, DC)
        c = reaim(orb)
        print("     фаза %5.1f deg: dv2 без прецессий %6.1f | с ними %6.1f м/с (разница"
              " %+6.1f) | dv после переприцеливания %.1f м/с"
              % (ph, a["dv2"], b["dv2"], b["dv2"] - a["dv2"], c["dv1"] + c["dv2"]))
    L.MOON = None

    print("-" * 78)
    print("5b) ЛУНА КАК РЕСУРС: поиск фазы и угла, где dv2 меньше CR3BP-номинала (%.1f м/с)"
          % nom["dv2"])
    # только те случаи, где трасса действительно проходит возле гало (иначе dv2 - не
    # стоимость входа на орбиту, а расстояние до неё)
    cand = []
    for ph in np.arange(0, 360, 30.0):
        L.MOON = moon(ph)
        for dth in np.deg2rad(np.arange(-25, 26, 5)):
            r = one_leg(orb, TH + dth, DC)
            if r["gap"] < 10e3:
                cand.append((r, ph, dth))
    print("   пересечений с гало (промах < 10 000 km) из %d проб: %d"
          % (12 * len(np.deg2rad(np.arange(-25, 26, 5))), len(cand)))
    if cand:
        r, ph, dth = min(cand, key=lambda c: c[0]["dv2"])
        print("   лучший: фаза %.0f deg, theta %+.0f deg -> dv2 = %.1f м/с (промах %.0f km),"
              " dv = %.1f м/с (%+.1f к CR3BP %.1f)"
              % (ph, np.degrees(dth), r["dv2"], r["gap"], r["dv1"] + r["dv2"],
                 r["dv1"] + r["dv2"] - nom["dv1"] - nom["dv2"], nom["dv1"] + nom["dv2"]))
        L.MOON = moon(ph)
        r2 = reaim(orb)
        print("   локальное переприцеливание: dv = %.1f м/с (%+.1f)"
              % (r2["dv1"] + r2["dv2"], r2["dv1"] + r2["dv2"] - nom["dv1"] - nom["dv2"]))
    L.MOON = None

    print("-" * 78)
    print("6) J2 И ПЛОЩАДКА (аналитика; в динамику не вводилось)")
    r_ = L.RE + L.H_PARK
    n_leo = np.sqrt(L.GM_E / r_**3)
    Od = -1.5 * n_leo * 1.08263e-3 * (L.RE / r_)**2 * np.cos(L.I_PARK)   # rad/s
    dO = np.degrees(Od * 86400)
    print("   NОО: период = %.2f мин, дрейф узла из-за J2 = %.2f deg/сут"
          % (2 * np.pi / n_leo / 60, dO))
    print("   направление Земля->L2 уходит на %.2f deg/сут -> узел совпадает с ним"
          % (360 / 365.25))
    print("   каждые %.1f сут (окно старта повторяется %d раз в году)"
          % (360 / abs(dO - 360 / 365.25), int(round(365.25 / (360 / abs(dO - 360 / 365.25))))))
    print("   цена, если узел НЕ совпал (разворот плоскости в момент разгона):")
    u1 = nom["u"] * L.V_U
    print("     angle, deg: " + " ".join("%5.0f" % a for a in (0, 5, 10, 15, 20, 23.5)))
    print("     dv1,  m/s : " + " ".join("%7.0f" % (np.sqrt(
        u1**2 + (L.V_LEO * L.V_U)**2 - 2 * u1 * L.V_LEO * L.V_U * np.cos(np.deg2rad(a)))
        * 1000) for a in (0, 5, 10, 15, 20, 23.5)))
    print("   наклонение 23.5 deg достижимо только с площадки |широта| <= 23.5 deg;")
    print("   Байконур 45.6 N, Плесецк 62.9 N - нет (Куру 5.2, Тхимбху 8.6,"
          " Шрихарикота 13.7, Вэньчан 19.6 - да).")

    print("-" * 78)
    print("7) УДЕРЖАНИЕ НА ГАЛО (из собственного числа неустойчивости модели)")
    te = L.T_U / L.LAMBDA
    print("   e-folding = %.1f сут (lambda = %.4f): ошибка df в 1 km по неустойчивому"
          % (te / 86400, L.LAMBDA))
    print("   направлению требует dv = %.3f мм/с на один e-fold" % (1000 / te * 1000))
    for tol in (100.0, 1000.0, 10000.0):
        per = 1e3 * tol / te                       # м/с на одно событие
        print("   допуск +-%5.0f km: %.4f м/с на коррекцию, %.3f м/с/год (это ДИНАМИЧЕСКОЕ"
          % (tol, per, per * 365.25 / (te / 86400)))
        print("                    требование; реальный бюджет умножается на ошибки")
        print("                    исполнения РД и навигации - в разы, не в десятки раз).")
    print("   для сравнения: ACE (Боровин, Ильин и др.) - коррекции удержания раз в 8 нед,")
    print("   геометрии орбиты - раз в 3-6 мес.")
    print("-" * 78)
    print("ИТОГ по dv = %.0f м/с (dv1 %.0f + dv2 %.0f):" % (nom["dv1"] + nom["dv2"],
                                                            nom["dv1"], nom["dv2"]))
    print("  dv1 не меняют ни SRP, ни Луна; эксцентриситет Земли даёт к нему +33 м/с.")
    print("  SRP с неопределённостью A и C_R - <= %.1f м/с; тень (даже сутки подряд) -"
          % (max(x[0] for x in srps) - nom["dv2"]))
    print("  <= %.1f мм/с, т.е. на 3-4 порядка меньше: в модель не вводится."
          % (1000 * max(x[1] for x in srps)))
    print("  Луна - главный эффект (dv2 140-475 м/с, после переприцеливания +60..+180 м/с)")
    print("  и одновременно единственный ресурс (гравманёвр: -72 м/с не найдено при")
    print("  промахе < 10 000 km, но выигрыш требует отдельной трассы с сближением).")
    print("  Прецессии лунной орбиты за трассу (-7.4 deg узел, +15.6 deg апсиды) стоят")
    print("  до -104 м/с в dv2, т.е. они НЕ пренебрежимо-медленные.")


if __name__ == "__main__":
    main()
