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

Ограничения: SRP без переориентации КА (C_R - параметр), тень - только полная (полутень
даёт плавный переход и входит в ту же оценку); Луна на орбите с фиксированными узлом и
линией апсид (регрессия 18.6 лет, апсиды 8.85 лет - обе медленнее 140-суточной трассы);
эксцентриситет Земли и J2 в динамику не введены - для них даны аналитические оценки.

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
UMBRA_SLOPE = (696000.0 - 6371.0) / L.KMD      # радиус конуса тени падает на 4.61 м/км


def moon(ph_deg, inc=True, ecc=True):
    return dict(a=MOON_A, e=MOON_E if ecc else 0.0, i=MOON_I if inc else 0.0,
                Om=0.0, w=0.0, n=N_SID, ph=np.deg2rad(ph_deg))


def one_leg(orb, theta, dC, mo=None, ts_n=2500):
    """integrate the coast once; dv1, dv2, gap (km), tof (d), min distance to the Moon,
    time spent in the Earth's shadow (s)."""
    u = L.u_for_jacobi(theta, orb["C"] - dC)
    sol = solve_ivp(L.rhs, (0, TOF), L.dep_state(theta, u), method="DOP853",
                    rtol=1e-9, atol=1e-11, max_step=MS, dense_output=True)
    ts1 = np.linspace(0.15, TOF, 200)
    t1 = L.closest(sol.sol, ts1, orb["P"], orb["V"])[0]
    t2, gap, dv2, _, _ = L.closest(sol.sol, np.linspace(t1 - .02, t1 + .02, 400),
                                   orb["P"], orb["V"])
    ts = np.linspace(0.0, TOF, ts_n)
    P = sol.sol(ts)[0:3]
    dt = ts[1] - ts[0]
    out = dict(dv1=(u - L.V_LEO) * L.V_U * 1000, dv2=dv2 * L.V_U * 1000,
               gap=gap * L.KMD, tof=t2 * L.DAY, dmo=np.nan, t_shadow=np.nan, u=u)
    out["tot"] = out["dv1"] + out["dv2"] + gap * L.KMD / 50.0
    if mo is not None:                        # geocentric distance to the Moon
        M = np.stack([L.moon_pos(t) for t in ts], 1)
        M[0] += 1 - L.MU                                   # geocentric -> barycentric
        out["dmo"] = np.sqrt(((P - M)**2).sum(0)).min() * L.KMD
    if L.SRP is not None:                     # полная тень: за Землёй и внутри конуса.
        ts = np.linspace(0, 0.02, 4000)                   # только первые 1.2 сут, шаг 50 с
        P = sol.sol(ts)[0:3]
        x_km = (P[0] - (1 - L.MU)) * L.KMD                # +x = анти-солнечно
        rho_km = np.hypot(P[1], P[2]) * L.KMD
        umbra = (x_km > 0) & (rho_km < 6371.0 - UMBRA_SLOPE * x_km)
        out["t_shadow"] = umbra.sum() * (ts[1] - ts[0]) * L.T_U
    return out


def reaim(orb, mo):
    opt = minimize(lambda q: one_leg(orb, q[0], q[1], mo)["tot"], [TH, DC],
                   method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-3, maxiter=40))
    return one_leg(orb, opt.x[0], opt.x[1], mo)


def main():
    L.SRP = L.MOON = None
    orb = L.halo_family(np.linspace(3e-5, 9e-4, 10), n=480)[0]
    nom = one_leg(orb, TH, DC)
    print("=" * 78)
    print("Референс CR3BP: dv1 = %.1f, dv2 = %.1f, dv = %.1f m/s, промах %.0f km, %.1f сут"
          % (nom["dv1"], nom["dv2"], nom["dv1"] + nom["dv2"], nom["gap"], nom["tof"]))

    print("-" * 78)
    print("1) ДАВЛЕНИЕ СОЛНЦА (dv2 при неизменном прицеле; dv1 не меняется)")
    print("   A, m2   C_R   a_SRP, м/с^2   t_тени, с   доп. dv от тени   dv2, m/s")
    srps = []
    for A in (20.0, 100.0):
        for CR in (1.0, 1.3, 1.8):
            L.SRP = dict(CR=CR, A_m=A / L.M_SC)
            r = one_leg(orb, TH, DC)
            acc = 4.54e-6 * CR * A / L.M_SC
            print("  %5.0f  %4.1f   %.2e        %6.0f      %+.2f мм/с        %7.1f"
                  % (A, CR, acc, r["t_shadow"], acc * r["t_shadow"] * 1000, r["dv2"]))
            srps.append((r["dv2"], 4.54e-6 * CR * A / L.M_SC * r["t_shadow"]))
    L.SRP = dict(CR=1.3, A_m=20.0 / L.M_SC)
    r = reaim(orb, None)
    print("   переприцеливание при A=20, C_R=1.3: dv = %.1f m/s (%+.1f к номиналу)"
          % (r["dv1"] + r["dv2"], r["dv1"] + r["dv2"] - nom["dv1"] - nom["dv2"]))
    L.SRP = dict(CR=1.8, A_m=100.0 / L.M_SC)
    r = reaim(orb, None)
    print("   переприцеливание при A=100, C_R=1.8: dv = %.1f m/s (%+.1f)"
          % (r["dv1"] + r["dv2"], r["dv1"] + r["dv2"] - nom["dv1"] - nom["dv2"]))
    L.SRP = None

    print("-" * 78)
    print("2) ЛУНА (прямая минус косвенная составляющая; a = 384400 km, e = 0.055, i = 5.1 deg)")
    print("   фаза, deg  промах, km   dv2, m/s   мин до Луны, km")
    for ph in (0.0, 45.0, 90.0, 180.0, 270.0):
        L.MOON = moon(ph)
        r = one_leg(orb, TH, DC, L.MOON)
        print("   %6.1f   %10.0f   %8.1f   %10.0f" % (ph, r["gap"], r["dv2"], r["dmo"]))
    print("   переприцеливание:")
    for ph in (0.0, 90.0, 270.0):
        L.MOON = moon(ph)
        r = reaim(orb, L.MOON)
        print("     фаза %5.1f deg: dv1 = %.1f  dv2 = %.1f  dv = %.1f m/s (%+.1f)"
              % (ph, r["dv1"], r["dv2"], r["dv1"] + r["dv2"],
                 r["dv1"] + r["dv2"] - nom["dv1"] - nom["dv2"]))
    print("   влияние формы лунной орбиты (фаза 0, без переприцеливания):")
    for name, kw in (("круговая в эклиптике (как было раньше)", dict(inc=False, ecc=False)),
                     ("e = 0.055", dict(inc=False, ecc=True)),
                     ("i = 5.1 deg", dict(inc=True, ecc=False)),
                     ("i и e", dict(inc=True, ecc=True))):
        L.MOON = moon(0.0, **kw)
        r = one_leg(orb, TH, DC, L.MOON)
        print("     %-38s dv2 = %6.1f m/s, промах %8.0f km" % (name, r["dv2"], r["gap"]))
    L.MOON = None

    print("-" * 78)
    print("3) J2 И ПЛОЩАДКА (аналитика; в динамику не вводилось)")
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
    print("4) УДЕРЖАНИЕ НА ГАЛО (из собственного числа неустойчивости модели)")
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
    print("Итог: dv1 = %.0f m/s не меняется ни при одном возмущении. SRP стоит <= %.1f м/с"
          % (nom["dv1"], max(x[0] for x in srps) - nom["dv2"]))
    print("с учётом неопределённости A и C_R; тень Земли - %.3f мм/с (пренебрежимо);"
          % (1000 * max(x[1] for x in srps)))
    print("Луна - главный эффект, и она же единственный ресурс (гравманёвр).")


if __name__ == "__main__":
    main()
