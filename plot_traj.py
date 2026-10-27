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

# ---------------- рисунок 2: данные по трассе (временны́е ряды и проектные числа) ----
t_d = np.linspace(0.0, TOF, tr.shape[1]) * L.DAY                     # сут
v_rot = np.linalg.norm(tr[3:6], axis=0) * L.V_U                      # км/с в ротирующемся
xi = (tr[0] - (1 - L.MU)) * L.KMD                                     # км за Землёй (+k L2)
rho = np.hypot(tr[1], tr[2]) * L.KMD                                  # км до оси Солнце-Земля
r_pen_t = L.RE / L.KM + (696000.0 + L.RE / L.KM) / (L.AU / L.KM) * np.maximum(xi, 0.0)
in_pen = (xi > 0.0) & (rho < r_pen_t)
v_halo = np.linalg.norm(orb["V"], axis=1) * L.V_U
# сравнение dv2 надо вести с ТОЧКОЙ орбиты, где происходит касание, а не с последним
# выборкой периодической орбиты:
j_in = int(np.argmin(((orb["P"] - tr[0:3, -1])**2).sum(1)))
v_orb_in = np.linalg.norm(orb["V"][j_in]) * L.V_U
# знак второго импульса берём из протокола (там он считается на сетке прицеливания):
# запас всего ~мм/с, и пересчёт по своей сетке его меняет - это граница ограничения dC=0
dv_brake = num(r"превышение скорости трассы над орбитой = ([-\d.]+) мм/с")
BRK = "тормозной" if dv_brake >= 0 else "разгонный"

f2, ax2 = plt.subplots(2, 2, figsize=(7.2, 4.8), dpi=200)
a = ax2[0, 0]
a.plot(t_d, xi / 1e3, lw=1.4, color="tab:blue")
a.axhline(L.RHO * L.KMD / 1e3, ls=":", lw=.9, color="tab:red")
a.set_title("(a) расстояние за Землёй ξ(t)", fontsize=8)
a.set_xlabel("время, сут"); a.set_ylabel("ξ, тыс. км"); a.grid(alpha=.3)
a.annotate("L2", (0.97, L.RHO * L.KMD / 1e3), fontsize=6.5, color="tab:red", ha="right")

a = ax2[0, 1]
a.plot(t_d, v_rot, lw=1.4, color="tab:blue", label="скорость трассы (ротирующийся кадр)")
a.plot(np.linspace(0.0, orb["T"] * L.DAY, len(v_halo)), v_halo, lw=1.1, color="tab:green",
       label="скорость на целевой орбите")
a.axhline(v_orb_in, ls="-.", lw=.9, color="tab:green")
a.axvline(t_d[-1], ls="--", lw=.9, color="k")
a.set_title("(b) |v|: вход %s, превышение трассы над орбитой %.1f мм/с"
            % (BRK, dv_brake), fontsize=8)
a.set_xlabel("время, сут"); a.set_ylabel("|v|, км/с"); a.grid(alpha=.3)
a.legend(fontsize=5.8, loc="lower left")

a = ax2[1, 0]
a.plot(t_d, rho / 1e3, lw=1.4, color="tab:blue", label="ρ до оси Солнце—Земля")
a.plot(t_d, r_pen_t / 1e3, lw=1.1, color="0.35", ls=":", label="радиус полутени Земли")
a.fill_between(t_d, 0, np.where(in_pen, r_pen_t / 1e3, 0), color="0.8", lw=0)
a.set_title("(c) затмения на перелёте (серое — в полутени)", fontsize=8)
a.set_xlabel("время, сут"); a.set_ylabel("ρ, тыс. км"); a.grid(alpha=.3)
a.legend(fontsize=5.8)

a = ax2[1, 1]; a.axis("off")
lines = ["dv1 = %8.1f м/с   (разгонный с НОО, 200 км)" % ((u - L.V_LEO) * L.V_U * 1000),
         "dv2 = %8.1f м/с   (%s)" % (num(r"dv2 \(второй импульс\)\s*=\s*([\d.]+)"),
                                     BRK),
         "тоф = %8.1f сут  промах прицела %6.0f км"
         % (TOF * L.DAY, num(r"промах по месту прицеливания\s*=\s*(\d+) km")),
         "цель: гало Az = %6.0f км, Ax = %6.0f км" % (Z_TGT * L.KMD, orb["Ax"] * L.KMD),
         "      период %5.1f сут, C = %.7f" % (orb["T"] * L.DAY, orb["C"]),
         "в тени на перелёте: %.2f %% времени (%.0f из %d выборок)"
         % (100.0 * in_pen.mean(), in_pen.sum(), len(in_pen)),
         "ввести на НОО: %s кг (РБ + КА)" % f"{int(num(r'ввести на НОО (\d+) kg')):d}"]
a.text(0.0, 0.95, "\n".join(lines), fontsize=7.2, family="monospace", va="top")
a.set_title("(d) проектные числа (result.txt)", fontsize=8)

f2.tight_layout(pad=0.5)
f2.savefig("traj_data.png")
print("traj_data.png: тень на перелёте %.2f%% времени (%d выборок), |v| входа %.4f против"
      " %.4f км/с; вход %s (по протоколу %+0.1f мм/с)"
      % (100.0 * in_pen.mean(), in_pen.sum(), v_rot[-1], v_orb_in, BRK, dv_brake))
