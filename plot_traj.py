#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Построение трассы перелёта НОО -> гало у L2 по тем же числам, что и отчёт.

Целевая орбита, угол старта, избыток энергии dC и время перелёта ЧИТАЮТСЯ из
result.txt (как и в l2_perturb.py / make_docx.js), поэтому рисунок не может
разойтись с расчётом: трасса интегрируется той же моделью l2_transfer.py.

Рисует три вида: (a) вид сверху (плоскость эклиптики) - трасса, целевое гало,
орбита Луны для масштаба; (b) разрез по xz - наклон плоскости выведения и конус
полутени Земли с порогом Az_min; (c) та же трасса в пространстве.

Запуск:  python3 plot_traj.py        ->  traj.png   (~20 с)
"""
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import l2_transfer as L

RES = open("result.txt", encoding="utf8").read()


def num(pat, src=RES):
    return float(re.search(pat, src).group(1))


Z_TGT = num(r"целевая орбита: Az = (\d+) km") / L.KMD          # амплитуда по z, а.е.
TH = np.deg2rad(num(r"угол старта в плоскости НОО\s*=\s*([-\d.]+) deg"))
DC = num(r"dC = ([-\d.eE+]+)")
TOF = num(r"время активного перелёта\s*=\s*([-\d.]+) сут") * 86400.0 / L.T_U   # TU
AZ_MIN = num(r"Az_min = (\d+) km") / L.KMD

orb = L.halo_family([Z_TGT], n=900)[0]
u = L.u_for_jacobi(TH, orb["C"] - DC)
tr = L.fly_traj(L.dep_state(TH, u), TOF, n=3000, rtol=1e-11)

# геоцентрические координаты вращающегося кадра, км: x - от Земли к L2 (анти-Солнце)
def gc(P):
    return np.array([(P[0] - (1 - L.MU)) * L.KMD, P[1] * L.KMD, P[2] * L.KMD])


T, H = gc(tr[0:3]), gc(orb["P"].T)
LEO = gc(np.column_stack([L.R_P * (np.cos(a) * L.X_HAT + np.sin(a) * L.E_LEO)
                          for a in np.linspace(0, 2 * np.pi, 200)]))
xi = np.linspace(0, T[0].max() * 1.05, 200)
r_pen = L.RE / L.KM + (696000.0 + L.RE / L.KM) / (L.AU / L.KM) * xi     # км, радиус полутени

fig = plt.figure(figsize=(9.8, 4.4), dpi=200)
a = fig.add_subplot(1, 3, 1)
a.plot(H[0] / 1e3, H[1] / 1e3, "-", lw=1.4, color="tab:green", label="гало у L2 (цель)")
a.plot(T[0] / 1e3, T[1] / 1e3, "-", lw=1.5, color="tab:blue", label="трасса перелёта")
a.plot(LEO[0] / 1e3, LEO[1] / 1e3, "-", lw=1.0, color="k", label="НОО 200 км")
a.plot([0], [0], "o", ms=6, color="tab:orange", label="Земля")
a.plot([L.RHO * L.KMD], [0], "x", ms=8, color="tab:red", label="L2")
a.add_patch(plt.Circle((0, 0), 384400 / 1e3, fill=False, ls=":", lw=0.8, color="0.4"))
a.text(384400 / 1e3 * 0.72, -384400 / 1e3 * 0.72, "орбита Луны", fontsize=6.5, color="0.35")
a.set_aspect("equal"); a.set_xlabel("ξ от Земли, тыс. км"); a.set_ylabel("η", rotation=0)
a.set_title("(a) вид сверху (эклиптика)", fontsize=8)
a.grid(alpha=0.3); a.legend(fontsize=6.0, loc="upper left")

b = fig.add_subplot(1, 3, 2)
b.fill_between(xi / 1e3, -r_pen, r_pen, color="0.75", alpha=0.55, lw=0,
               label="конус полутени Земли")
b.plot(H[0] / 1e3, H[2] / 1e3, "-", lw=1.4, color="tab:green")
b.plot(T[0] / 1e3, T[2] / 1e3, "-", lw=1.5, color="tab:blue")
b.axhline(AZ_MIN * L.KMD / 1e3, ls="--", lw=0.9, color="k")
b.axhline(-AZ_MIN * L.KMD / 1e3, ls="--", lw=0.9, color="k")
b.text(0.05 * L.RHO * L.KMD / 1e3, AZ_MIN * L.KMD / 1e3 * 1.15,
       "порог Az$_{min}$ = %.0f тыс. км" % (AZ_MIN * L.KMD / 1e3), fontsize=6.5)
b.plot([0], [0], "o", ms=6, color="tab:orange")
b.set_aspect("equal"); b.set_xlabel("ξ от Земли, тыс. км"); b.set_ylabel("ζ", rotation=0)
b.set_title("(b) разрез xz: наклон плоскости %.1f°\nи выход гало из тени" %
            np.degrees(L.I_ECL), fontsize=8)
b.grid(alpha=0.3); b.legend(fontsize=6.0, loc="lower right")

c = fig.add_subplot(1, 3, 3, projection="3d")
c.plot(H[0] / 1e3, H[1] / 1e3, H[2] / 1e3, "-", lw=1.2, color="tab:green")
c.plot(T[0] / 1e3, T[1] / 1e3, T[2] / 1e3, "-", lw=1.4, color="tab:blue")
c.scatter([0], [0], [0], s=18, c="tab:orange")
c.scatter([L.RHO * L.KMD / 1e3], [0], [0], s=18, marker="x", c="tab:red")
c.set_xlabel("ξ, тыс. км", fontsize=6.5); c.set_ylabel("η, тыс. км", fontsize=6.5)
c.set_zlabel("ζ, тыс. км", fontsize=6.5)
c.tick_params(labelsize=6)
c.set_title("(c) трасса в пространстве", fontsize=8)
c.view_init(elev=16, azim=28)

fig.tight_layout(pad=0.4)
fig.savefig("traj.png")
print("traj.png: %d точек трассы, %d точек гало (Az = %.0f km), тоф %.1f сут, dv1 = %.1f м/с"
      % (tr.shape[1], H.shape[1], Z_TGT * L.KMD, TOF * L.DAY, (u - L.V_LEO) * L.V_U * 1000))
