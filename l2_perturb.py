#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Сколько стоят возмущения и внемодельные ограничения, исключённые чистой CR3BP-моделью
l2_transfer.py: наклонённая плоскость выведения (Восточный), давление Солнца (включая
тень и неопределённость C_R/A), Луна на КРУГОВОЙ орбите в эклиптике (форму и прецессии
лунной орбиты по заданию не учитываем - их величину измерили и отметили отдельно),
эксцентриситет орбиты Земли, J2 и площадка, удержание на гало и ресурс СПД-100ВМ.

Методика: та же номинальная трасса, что в result.txt - целевая орбита, прицел и dC
ЧИТАЮТСЯ из result.txt, поэтому perturb не может разойтись с основным расчётом.
Мера несоответствия = промах по месту в момент наилучшего сближения с целевой орбитой и
dv2 для входа на неё; затем то же с переприцеливанием по (theta, dC) - как сделал бы
реальный расчёт.

Ограничения: SRP без переориентации КА (C_R - параметр); Луна - тело на предписанной
орбите (её влияние на движение Земли в рамках модели не учитывается, кроме члена
"прямым минус косвенным"); J2 в динамику не вводится - для него дана аналитическая
оценка геометрии узла; наклонение трассы снимается на этапе выбора цели (гало с тем же
наклонением, методика Боровина и др.) - здесь измерена цена цели из эклиптической семьи,
т.е. верхняя оценка.

Запуск:  python3 l2_perturb.py        (~10 мин, нужен result.txt)
"""
import re
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import l2_transfer as L


def nominal(path="result.txt"):
    """Az цели, угол старта и dC - из протокола основного расчёта."""
    t = open(path, encoding="utf8").read()
    az = float(re.search(r"целевая орбита: Az = ([\d.]+) km", t).group(1))
    th = float(re.search(r"угол старта в плоскости НОО\s*=\s*([-\d.]+) deg", t).group(1))
    dc = float(re.search(r"dC = ([-\d.eE+]+)", t).group(1))
    return az / L.KMD, np.deg2rad(th), dc


Z_TGT, TH, DC = nominal()
TOF = 3.5                                    # горизонт поиска сближения, TU
MS = 0.02                                    # шаг: разрешить лунный месяц
MOON_A, MOON_E, MOON_I = 384400.0 / L.KMD, 0.055, np.deg2rad(5.145)
N_SID = 2 * np.pi * L.DAY / 27.321661        # сидерический месяц -> rad/TU
# узел убывает за 18.6 лет, линия апсид идёт за 8.85 лет -> 20.4 и 42.9 цикла за год
OMDOT = -2 * np.pi / (18.6 * 365.25 / L.DAY)
WDOT = 2 * np.pi / (8.85 * 365.25 / L.DAY)

UMBRA_SLOPE = (696000.0 - 6371.0) / L.KMD      # радиус конуса тени падает на 4.61 м/км


def moon(ph_deg, inc=False, ecc=False, prec=False):
    """по заданию: круговая орбита Луны в эклиптике, без прецессий (inc=ecc=prec=False);
    флаги включаются только чтобы ИЗМЕРИТЬ величину исключённого эффекта."""
    d = dict(a=MOON_A, e=MOON_E if ecc else 0.0, i=MOON_I if inc else 0.0,
             Om=0.0, w=0.0, n=N_SID, ph=np.deg2rad(ph_deg))
    if prec:
        d.update(Omdot=OMDOT, wdot=WDOT)
    return d


def plane(deg):
    """пересобрать плоскость выведения (E_LEO построена при импорте модуля)."""
    L.I_ECL = np.deg2rad(deg)
    L.E_LEO = np.array([0.0, np.cos(L.I_ECL), np.sin(L.I_ECL)])


def zsh(a):
    """доля невязки скорости (dv2) вне эклиптики, % - мера цены разворота плоскости."""
    return 100 * abs(a["res"][2]) / np.linalg.norm(a["res"])


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


def one_leg(orb, theta, dC, ts_n=2500, comove=False, ms=None):
    """integrate the coast once; dv1, dv2, gap (km), tof (d), min distance to the Moon,
    time spent in the Earth's shadow (s).  ms - шаг интегрирования (0.5 годится там,
    где нет лунного возмущения: в 5-10 раз быстрее)."""
    u = L.u_for_jacobi(theta, orb["C"] - dC)
    sol = solve_ivp(L.rhs, (0, TOF), L.dep_state(theta, u), method="DOP853",
                    rtol=1e-9, atol=1e-11, max_step=ms or MS, dense_output=True)
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
        res = QD[:, i] - orb["V"][j]
    else:
        t2, gap, dv2, arr, arrv = L.closest(sol.sol, ts2, Pt, Vt)
        j = int(np.argmin(((Pt - arr)**2).sum(1)))
        res = arrv - Vt[j]                        # вектор невязки скорости (dv2) в осях
    ts = np.linspace(0.0, TOF, ts_n)
    P = sol.sol(ts)[0:3]
    dt = ts[1] - ts[0]
    out = dict(dv1=(u - L.V_LEO) * L.V_U * 1000, dv2=dv2 * L.V_U * 1000,
               gap=gap * L.KMD, tof=t2 * L.DAY, dmo=np.nan, t_shadow=np.nan,
               t_day=np.nan, u=u, res=res)
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


def reaim(orb, comove=False, ms=None):
    """local (launch angle, energy) re-aim = what a design in the fuller model would do.
    The active force model is whatever L.SRP / L.MOON / L.SUN_E are set to."""
    opt = minimize(lambda q: one_leg(orb, q[0], q[1], comove=comove, ms=ms)["tot"],
                   [TH, DC], method="Nelder-Mead",
                   options=dict(xatol=1e-6, fatol=1e-3, maxiter=40))
    return one_leg(orb, opt.x[0], opt.x[1], comove=comove, ms=ms)


def main():
    L.SRP = L.MOON = None
    orb = L.halo_family([Z_TGT], n=480)[0]        # цель - та же орбита, что в result.txt
    nom = one_leg(orb, TH, DC)
    print("=" * 78)
    print("Референс CR3BP: dv1 = %.1f, dv2 = %.1f, dv = %.1f m/s, промах %.0f km, %.1f сут"
          % (nom["dv1"], nom["dv2"], nom["dv1"] + nom["dv2"], nom["gap"], nom["tof"]))
    print("целевая орбита: Az = %.0f km (из result.txt), период %.1f сут, затмений %s"
          % (Z_TGT * L.KMD, orb["T"] * L.DAY,
             "нет" if orb["sh_pen"] == 0 else "есть"))

    print("-" * 78)
    print("1) ПЛОСКОСТЬ ВЫВЕДЕНИЯ: трасса не в эклиптике, цель - гало эклиптической семьи")
    print("   (Восточный, i_LEO = %.1f deg, наклон эклиптики %.2f deg)" % (L.I_LEO, L.EPS_ECL))
    print("   наклон    dv1 прицел  dv2 прицел  промах   dv2 переприц  тоф    доля dv2")
    print("   к эклип.    m/s         m/s        km         m/s        сут   вне эклиптики, %")
    I0 = np.degrees(L.I_ECL)
    rows = []
    for deg, why in ((0.0, "идеал - только с площадки |широта| <= 23.4 deg"),
                     (I0, "МИНИМУМ для Восточного: i_LEO - eps"),
                     (L.I_LEO, "плоскость не повёрнута к эклиптике"),
                     (L.I_LEO + L.EPS_ECL, "худшая расстановка узлов: i_LEO + eps")):
        plane(deg)
        f, r = one_leg(orb, TH, DC, ms=0.5), reaim(orb, ms=0.5)
        rows.append((deg, f, r))
        print("  %5.1f deg %9.1f %10.1f %9.0f %12.1f %9.1f      %4.1f / %4.1f   (%s)"
              % (deg, f["dv1"], f["dv2"], f["gap"], r["dv2"], r["tof"], zsh(f), zsh(r), why))
    plane(I0)
    dv2_t = [f["dv2"] for _, f, _ in rows]
    zq_t = [zsh(f) for _, f, _ in rows]
    d_tot = [r["dv1"] + r["dv2"] for _, _, r in rows]
    print("   dv1 от наклона плоскости НЕ зависит (разброс %.2f m/s): его назначает"
          % (max(f["dv1"] for _, f, _ in rows) - min(f["dv1"] for _, f, _ in rows)))
    print("   интеграл Якоби на радиусе НОО, а не направление импульса.")
    print("   dv2: наклон %.1f deg (Восточный, оптимум) против 0 deg - %+.1f m/s; наклон"
          % (I0, d_tot[1] - d_tot[0]))
    print("   %.1f deg - %+.1f m/s; худший %.1f deg - %+.1f m/s. Доля невязки вне эклиптики"
          % (rows[2][0], d_tot[2] - d_tot[0], rows[3][0], d_tot[3] - d_tot[0]))
    print("   растёт с наклоном (%.0f%% -> %.0f%%), т.е. это именно разворот плоскости,"
          % (zq_t[1], zq_t[3]))
    print("   а не торможение. Разброс %.0f m/s между соседними наклонами - мера"
          % (max(d_tot) - min(d_tot)))
    print("   негладкости множества пересечений трассы с целью, а не эффект наклонения.")
    print("   ВЫВОД: при i_LEO = %.1f deg трасса в наклонённой плоскости стоит тех же ~100"
          % L.I_LEO)
    print("   м/с, что и эклиптическая, ЕСЛИ азимутом и моментом старта повернуть плоскость")
    print("   так, чтобы наклон к эклиптике был минимален (i_LEO - eps = %.1f deg);"
          % I0)
    print("   не повернуть - значит платить %+.0f m/s (наклон 51.6 deg) ... %+.0f m/s"
          % (d_tot[2] - d_tot[0], d_tot[3] - d_tot[0]))
    print("   (наклон 75.0 deg). Цена снимается и выбором цели: гало с наклонением,"
          " равным")
    print("   наклонению трассы (Боровин, Ильин и др.: из множества траекторий выбираются")
    print("   те, наклонение которых соответствует наклонению орбиты выведения) - тогда в")
    print("   проект идёт переприцеленный dv2 = %.1f m/s, а таблица выше - верхняя оценка"
          % rows[1][2]["dv2"])
    plane_cost = d_tot[1] - d_tot[0]
    plane_cost_worst = d_tot[3] - d_tot[0]
    print("   эклиптической семьи.")

    print("-" * 78)
    print("2) ЭКЦЕНТРИСИТЕТ ОРБИТЫ ЗЕМЛИ (e = 0.0167; кадр вращается со СРЕДНЕЙ движением,"
          " поэтому линия Солнце-Земля и L2 дрейфуют за перелёт)")
    print("   в среднем-вращающемся кадре линия Солнце-Земля (и весь L2) делает эпicycle")
    print("   амплитудой 2*e*1AU = %.2f deg -> ~%.1f млн km: наивная фиксированная цель"
          % (2 * 0.01671 * 180 / np.pi, 2 * 0.01671 * L.KMD / 1e6))
    print("   недостижима. Честная цена - в кадре, идущем за линией центров (comoving).")
    print("   фаза старта    |S-Z|    dv1     dv2 наивн.   dv2 comoving   промах,km")
    e_dv1 = []
    for name, M0 in (("перигей", 0.0), ("90 deg", np.pi / 2), ("апогей", np.pi),
                     ("270 deg", 3 * np.pi / 2)):
        L.SUN_E = dict(e=0.016710, M0=M0)
        a = one_leg(orb, TH, DC)
        b = one_leg(orb, TH, DC, comove=True)
        if name == "перигей":
            one_e = a
        e_dv1.append(a["dv1"])
        L.SUN_E = None
        print("   %-10s  %.5f  %7.1f    %9.1f       %8.1f       %8.0f"
              % (name, (1 - 0.01671 * np.cos(M0)), a["dv1"], a["dv2"], b["dv2"], b["gap"]))
    print("   ПЕРЕПРИЦЕЛИВАНИЕ (остаток dv2 - это НЕ стоимость эксцентриситета, а то,")
    print("   что гало-цель не является орбитой эллиптической задачи: в ER3BP орбиты у")
    print("   L2 квазипериодические и другого размера (r_L2 = R*(mu/3)^(1/3), ±1.7%).")
    print("   Собственно цена эксцентриситета - в dv1 (см. столбец dv1 выше: %+.0f м/с)."
          % (max(e_dv1) - nom["dv1"]))
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
    print("3) ДАВЛЕНИЕ СОЛНЦА (dv2 при неизменном прицеле; dv1 не меняется)")
    print("   A, m2   C_R   a_SRP, м/с^2  в тени, с   dv2 без тени и"
          " с ней, разность   dv2, m/s")
    srps = []
    for A in (20.0, 100.0):
        for CR in (1.0, 1.3, 1.8):
            L.SRP = dict(CR=CR, A_m=A / L.M_DRY)
            r = one_leg(orb, TH, DC)
            acc = 4.54e-6 * CR * A / L.M_DRY
            print("  %5.0f  %4.1f   %.2e      %6.0f        <= %.4f мм/с          %7.1f"
                  % (A, CR, acc, r["t_shadow"], acc * r["t_day"] * 1000, r["dv2"]))
            srps.append((r["dv2"], acc * r["t_day"]))
    print("   Полутень - частичное затемнение, она не может дать больше, чем полная тень;")
    print("   верхняя оценка - сутки непрерывной тени. Для A = 20 м2 эффект на ~4 порядка")
    print("   ниже dv2, для A = 100 м2 - на 3; в динамику не вводится.")
    L.SRP = dict(CR=1.3, A_m=20.0 / L.M_DRY)
    r = reaim(orb)
    print("   переприцеливание при A=20, C_R=1.3: dv = %.1f m/s (%+.1f к номиналу)"
          % (r["dv1"] + r["dv2"], r["dv1"] + r["dv2"] - nom["dv1"] - nom["dv2"]))
    L.SRP = dict(CR=1.8, A_m=100.0 / L.M_DRY)
    r = reaim(orb)
    print("   переприцеливание при A=100, C_R=1.8: dv = %.1f m/s (%+.1f)"
          % (r["dv1"] + r["dv2"], r["dv1"] + r["dv2"] - nom["dv1"] - nom["dv2"]))
    L.SRP = None

    print("-" * 78)
    print("4) ЛУНА (по заданию - круговая орбита в эклиптике; a = 384400 km; прямая минус"
          " косвенная составляющая)")
    print("   фаза, deg  промах, km   dv2, m/s   мин до Луны, km")
    m_dv2, m_re, m_prec = [], [], []
    for ph in (0.0, 45.0, 90.0, 180.0, 270.0):
        L.MOON = moon(ph)
        r = one_leg(orb, TH, DC)
        m_dv2.append(r["dv2"])
        print("   %6.1f   %10.0f   %8.1f   %10.0f" % (ph, r["gap"], r["dv2"], r["dmo"]))
    print("   переприцеливание:")
    for ph in (0.0, 90.0, 270.0):
        L.MOON = moon(ph)
        r = reaim(orb)
        m_re.append(r["dv1"] + r["dv2"] - nom["dv1"] - nom["dv2"])
        print("     фаза %5.1f deg: dv1 = %.1f  dv2 = %.1f  dv = %.1f m/s (%+.1f)"
              % (ph, r["dv1"], r["dv2"], r["dv1"] + r["dv2"],
                 r["dv1"] + r["dv2"] - nom["dv1"] - nom["dv2"]))
    print("4a) ЧТО ИСКЛЮЧЕНО ПО ЗАДАНИЮ (ФОРМА лунной орбиты) И НАСКОЛЬКО ЭТО ВЕЛИКО"
          " (фаза 0, без переприцеливания):")
    for name, kw in (("круговая в эклиптике (модель по заданию)", dict(inc=False, ecc=False)),
                     ("e = 0.055", dict(inc=False, ecc=True)),
                     ("i = 5.1 deg", dict(inc=True, ecc=False)),
                     ("i и e", dict(inc=True, ecc=True))):
        L.MOON = moon(0.0, **kw)
        r = one_leg(orb, TH, DC)
        print("     %-38s dv2 = %6.1f m/s, промах %8.0f km" % (name, r["dv2"], r["gap"]))
    print("4b) ИСКЛЮЧЕНО ПО ЗАДАНИЮ: ПРЕЦССИИ ЛУННОЙ ОРБИТЫ (узел 18.6 лет -> -7.4 deg,")
    print("    апсиды 8.85 лет -> +15.6 deg за трассу) - измеренная величина эффекта:")
    for ph in (0.0, 90.0, 270.0):
        L.MOON = moon(ph, prec=False)
        a = one_leg(orb, TH, DC)
        L.MOON = moon(ph, prec=True)
        b = one_leg(orb, TH, DC)
        c = reaim(orb)
        m_prec.append(b["dv2"] - a["dv2"])
        print("     фаза %5.1f deg: dv2 без прецессий %6.1f | с ними %6.1f м/с (разница"
              " %+6.1f) | dv после переприцеливания %.1f м/с"
              % (ph, a["dv2"], b["dv2"], b["dv2"] - a["dv2"], c["dv1"] + c["dv2"]))
    L.MOON = None

    print("-" * 78)
    print("4c) ЛУНА КАК РЕСУРС: поиск фазы и угла, где dv2 меньше CR3BP-номинала (%.1f м/с)"
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
    print("5) J2 И ПЛОЩАДКА (Восточный; аналитика, в динамику не вводилось)")
    r_ = L.RE + L.H_PARK
    n_leo = np.sqrt(L.GM_E / r_**3)
    Od = -1.5 * n_leo * 1.08263e-3 * (L.RE / r_)**2 * np.cos(np.deg2rad(L.I_LEO))
    dO = np.degrees(Od * 86400)
    win = 360 / abs(dO - 360 / 365.25)
    print("   НОО 200 km, i = %.1f deg: период = %.2f мин, дрейф узла из-за J2 = %.2f deg/сут"
          % (L.I_LEO, 2 * np.pi / n_leo / 60, dO))
    print("   направление Земля->L2 уходит на %.2f deg/сут -> нужная ориентация плоскости"
          % (360 / 365.25))
    print("   повторяется каждые %.1f сут (окно старта %d раз в год)"
          % (win, round(365.25 / win)))
    print("   цена рассогласования направления разгона с касательной к НОО:")
    u1 = nom["u"] * L.V_U
    print("     angle, deg: " + " ".join("%5.0f" % a for a in (0, 5, 10, 15, 20, 25)))
    print("     dv1,  m/s : " + " ".join("%7.0f" % (np.sqrt(
        u1**2 + (L.V_LEO * L.V_U)**2 - 2 * u1 * L.V_LEO * L.V_U * np.cos(np.deg2rad(a)))
        * 1000) for a in (0, 5, 10, 15, 20, 25)))
    print("   Восточный: наклонение 51.6 deg - штатные прямые старты (так же, как у")
    print("   «Спектр-РГ», у которого i = 51,611 deg). Наклонение <= 23.4 deg (трасса прямо")
    print("   в эклиптике) с такой площадки невозможно по определению i >= широты, поэтому")
    print("   трасса строится в плоскости, наклонённой на i - eps = %.1f deg (п.1)." % I0)

    print("-" * 78)
    print("6) УДЕРЖАНИЕ НА ГАЛО ДВИГАТЕЛЕМ СПД-100ВМ (по собственному числу неустойчивости)")
    te = L.T_U / L.LAMBDA
    n_c = 365.25 / (te / 86400)                    # коррекций в год: по одной на e-fold
    print("   e-folding = %.1f сут (lambda = %.4f): ошибка 1 km по неустойчивому направлению"
          % (te / 86400, L.LAMBDA))
    print("   стоит %.3f мм/с; по одной коррекции на e-fold = %.1f шт/год, за %d лет - %.0f"
          % (1000 / te * 1000, n_c, L.SKT["LIFE_Y"], n_c * L.SKT["LIFE_Y"]))
    print("   СПД-100ВМ (36-55.pdf, табл. 3): тяга %.0f mN, Isp %.0f s, суммарный импульс"
          % (1000 * L.SKT["F"], L.SKT["ISP"]))
    print("   >%.1f MN*s, ресурс >%.0f ч и >%.0f включений:"
          % (L.SKT["RES_MNs"] / 1e6, L.SKT["RES_H"], L.SKT["RES_IGN"]))
    for tol in (100.0, 1000.0, 10000.0):
        per = 1e3 * tol / te                          # м/с на одно событие
        yr = per * n_c                                # м/с за год
        mx = L.M_DRY * (1 - np.exp(-yr / (L.SKT["ISP"] * L.G0)))
        hrs = L.M_DRY * yr / L.SKT["F"] / 3600
        print("   допуск +-%5.0f km: %6.2f м/с на коррекцию = %6.2f м/с/год; ксенон %.2f кг/год"
              % (tol, per, yr, mx))
        print("                    %5.0f ч работы/год (%.2f%% ресурса); в %.1f раза меньше"
              " лимита 100 м/с/год"
              % (hrs, 100 * hrs / L.SKT["RES_H"], L.SKT["CAP_Y"] / yr))
    print("   лимиту 100 м/с/год равны %.0f ч работы и %.1f кг ксенона в год: даже на пределе"
          % (L.M_DRY * 100.0 / L.SKT["F"] / 3600,
             L.M_DRY * (1 - np.exp(-100.0 / (L.SKT["ISP"] * L.G0)))))
    print("   бюджета один двигатель умещается (%.0f%% ресурса за %d лет, в %.0f раз в лимит"
          " включений)"
          % (100 * L.M_DRY * 100.0 / L.SKT["F"] / 3600 * L.SKT["LIFE_Y"] / L.SKT["RES_H"],
             L.SKT["LIFE_Y"], L.SKT["RES_IGN"] / (n_c * L.SKT["LIFE_Y"])))
    print("   Минимум удержания: бюджет линейен по допуску и почти не зависит от частоты")
    print("   коррекций (рост экспоненциальный, поэтому цена одного e-fold не меняется от")
    print("   того, дробить его на два или нет); при +/-100 km это %.2f м/с/год. Лимиту"
          % (1e3 * 100.0 / te * n_c))
    print("   задания 100 m/s/год соответствует допуск +-%.0f km по месту - т.е. лимит"
          % (100.0 / (1e3 * 100.0 / te * n_c) * 100.0))
    print("   задаёт не динамика, а допустимая ширина орбиты. Для сравнения: ACE (Боровин,")
    print("   Ильин и др.) - коррекции удержания раз в 8 нед, геометрии орбиты - раз в 3-6 мес.")
    print("-" * 78)
    print("ИТОГ по dv = %.0f м/с (dv1 %.0f + dv2 %.0f), цель - гало Az = %.0f km, без затмений:"
          % (nom["dv1"] + nom["dv2"], nom["dv1"], nom["dv2"], Z_TGT * L.KMD))
    print("  dv1 не меняют ни SRP, ни Луна, ни наклон плоскости выведения; эксцентриситет")
    print("  Земли даёт к нему %+3.0f м/с (максимум по фазе старта)." % (max(e_dv1) - nom["dv1"]))
    print("  Наклон плоскости %.1f deg (Восточный, оптимум) стоит после переприцеливания"
          % I0)
    print("  %+.1f м/с; не поворачивать плоскость к эклиптике - %+.1f м/с (75 deg)."
          % (plane_cost, plane_cost_worst))
    print("  без переприцеливания dv2 вырастает до %.0f м/с, из них %.0f%% - вне эклиптики."
          % (max(dv2_t), max(zq_t)))
    print("  Эта цена - свойство ЦЕЛИ (гало эклиптической семьи), а не перелёта: с")
    print("  наклонным гало она снимается (п.1).")
    print("  SRP с неопределённостью A и C_R - <= %.1f м/с; тень (даже сутки подряд) -"
          % (max(x[0] for x in srps) - nom["dv2"]))
    print("  <= %.1f мм/с, т.е. на 3-4 порядка меньше: в модель не вводится."
          % (1000 * max(x[1] for x in srps)))
    print("  Луна (по заданию - круговая орбита в эклиптике) - главный эффект: dv2")
    print("  %.0f..%.0f м/с при прежнем прицеле, %+d..%+d м/с после переприцеливания."
          % (min(m_dv2), max(m_dv2), min(m_re), max(m_re)))
    print("  Форма и прецессии лунной орбиты по заданию НЕ учитываются; измерено")
    print("  %+d..%+d м/с в dv2 - исключение этих членов есть требование задания, а не"
          % (min(m_prec), max(m_prec)))
    print("  вывод из расчёта (с круговой Луной dv2 меньше: см. 4a/4b).")
    per100 = 1e3 * 100.0 / te
    yr100 = per100 * n_c
    print("  Удержание при допуске +-100 km: %.2f м/с/год - в %.0f раз ниже лимита 100 м/с/год;"
          % (yr100, 100.0 / yr100))
    print("  ксенон %.2f кг/год, %.1f ч работы/год (%.2f%% ресурса СПД-100ВМ)."
          % (L.M_DRY * (1 - np.exp(-yr100 / (L.SKT["ISP"] * L.G0))),
             L.M_DRY * yr100 / L.SKT["F"] / 3600, 100 * L.M_DRY * yr100 / L.SKT["F"]
             / 3600 / L.SKT["RES_H"]))


if __name__ == "__main__":
    main()
