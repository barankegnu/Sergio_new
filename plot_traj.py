#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Построение трассы перелёта НОО -> гало у L2 по тем же числам, что и отчёт.

Целевая орбита, угол старта, dC и время перелёта ЧИТАЮТСЯ из блока AIM в result.txt
(как и в l2_perturb.py / make_docx.js), поэтому рисунки не могут разойтись с расчётом:
трасса интегрируется той же моделью l2_transfer.py.

Рисунки разделены, чтобы каждый читался с A4 без лупы:
  fig1_plan.png    вид сверху (эклиптика): трасса, гало, НОО, Луна для масштаба
  fig1_xz.png      разрез xz: наклон плоскости выведения и конус полутени с Az_min
  fig1_3d.png      та же трасса в пространстве
  fig2_xi_v.png    данные по трассе: xi(t) и |v|(t) против скорости орбиты
  fig2_shadow.png  затенение на перелёте: rho(t) и радиус полутени r_pen(t)

Запуск:  python3 plot_traj.py        (-> пять .png, ~25 с)
"""
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import l2_transfer as L

RES = open("result.txt", encoding="utf8").read()
FS, FST, FSW = 9, 10, 8          # подписи осей, заголовки, деления


def num(pat, src=RES):
    return float(re.search(pat, src).group(1))


Z_TGT = num(r"целевая орбита: Az = (\d+) km") / L.KMD          # амплитуда по z, а.е.
TH = num(r"AIM z = [-\d.e+]+ theta = ([-\d.]+)")               # радианы
DC = num(r"AIM z = [-\d.e+]+ theta = [-\d.]+ dC = ([-\d.e+]+)")
TOF = num(r"AIM z = [-\d.e+]+ theta = [-\d.]+ dC = [-\d.e+]+ tof_TU = ([-\d.]+)")
AZ_MIN = num(r"Az_min = (\d+) km") / L.KMD

plt.rcParams.update({"font.size": FS, "axes.titlesize": FST, "axes.labelsize": FS,
                     "xtick.labelsize": FSW, "ytick.labelsize": FSW, "axes.linewidth": .8, "savefig.dpi": 200, "figure.dpi": 110})

orb = L.halo_family([Z_TGT], n=900)[0]
u = L.u_for_jacobi(TH, orb["C"] - DC)
tr = L.fly_traj(L.dep_state(TH, u), TOF, n=3000, rtol=1e-11)

# геоцентрические координаты вращающегося кадра, тыс. км: x - от Земли к L2
def gc(P, k=1e3):
    return np.array([(P[0] - (1 - L.MU)) * L.KMD / k, P[1] * L.KMD / k, P[2] * L.KMD / k])


T, H = gc(tr[0:3]), gc(orb["P"].T)
LEO = gc(np.column_stack([L.R_P * (np.cos(a) * L.X_HAT + np.sin(a) * L.E_LEO)
                          for a in np.linspace(0, 2 * np.pi, 200)]))
t_d = np.linspace(0.0, TOF, tr.shape[1]) * L.DAY
xi_km, rho_km = T[0] * 1e3, np.hypot(T[1], T[2]) * 1e3          # км (T в тыс. км)
r_pen = (L.RE / L.KM + (696000.0 + L.RE / L.KM) / (L.AU / L.KM) * np.maximum(xi_km, 0.0)) / 1e3
in_pen = (xi_km > 0) & (rho_km / 1e3 < r_pen)
v_rot = np.linalg.norm(tr[3:6], axis=0) * L.V_U
j_in = int(np.argmin(((orb["P"] - tr[0:3, -1])**2).sum(1)))
v_orb_in = np.linalg.norm(orb["V"][j_in]) * L.V_U
BRK = "тормозной" if num(r"превышение скорости трассы над орбитой = ([-\d.]+) мм/с") >= 0 else "разгонный"

# ---- рис. 1: вид сверху -------------------------------------------------------------
fig = plt.figure(figsize=(5.6, 4.2))
a = fig.add_axes([.10, .10, .87, .84]); a.set_aspect("equal")
a.plot(T[0], T[1], "-", lw=1.8, color="tab:blue", label="трасса перелёта")
a.plot(H[0], H[1], "-", lw=1.6, color="tab:green", label="гало у L2 (цель)")
a.plot(LEO[0], LEO[1], "-", lw=1.4, color="k", label="НОО 200 км")
a.add_patch(plt.Circle((0, 0), 384.4, fill=False, ls=":", lw=1.0, color="0.45"))
a.text(300, -230, "орбита Луны", fontsize=FST - 1, color="0.35", rotation=25)
a.plot(0, 0, "o", ms=8, color="tab:orange", label="Земля")
a.plot(L.RHO * L.KMD / 1e3, 0, "x", ms=9, mew=2, color="tab:red", label="L2")
a.annotate("начало трассы", (T[0][0], T[1][0]), xytext=(60, 34), fontsize=FST - 1,
           arrowprops=dict(arrowstyle="->", lw=.9))
a.annotate("вход на гало", (T[0][-1], T[1][-1]), xytext=(120, -95), fontsize=FST - 1,
           arrowprops=dict(arrowstyle="->", lw=.9))
a.set_xlabel("\u03be от Земли, тыс. км"); a.set_ylabel("\u03b7, тыс. км")
a.set_title("Рис. 1. Вид сверху на эклиптике")
a.grid(alpha=.3); a.legend(fontsize=FST - 1, loc="upper left", framealpha=.95)
fig.savefig("fig1_plan.png"); plt.close(fig)

# ---- рис. 2: разрез xz ---------------------------------------------------------------
fig = plt.figure(figsize=(5.6, 3.6))
a = fig.add_axes([.10, .13, .87, .80]); a.set_aspect("equal")
xi = np.linspace(0, max(T[0].max(), L.RHO * L.KMD / 1e3 * 1.05) * 1e3, 300) / 1e3
rp = (L.RE / L.KM + (696000.0 + L.RE / L.KM) / (L.AU / L.KM) * xi * 1e3) / 1e3
a.fill_between(xi, -rp, rp, color="0.78", alpha=.85, lw=0,
               label="конус полутени Земли")
a.axhspan(AZ_MIN * L.KMD / 1e3, 200, color="0.97", zorder=0)
a.plot(H[0], H[2], "-", lw=1.6, color="tab:green", label="гало: \u03b6 = \u00b1%.0f тыс. км"
       % (Z_TGT * L.KMD / 1e3))
a.plot(T[0], T[2], "-", lw=1.8, color="tab:blue", label="трасса")
for s in (1, -1):
    a.axhline(s * AZ_MIN * L.KMD / 1e3, ls="--", lw=1.2, color="k")
a.text(180, AZ_MIN * L.KMD / 1e3 * 1.12, "порог тени Az$_{min}$ = %.1f тыс. км"
       % (AZ_MIN * L.KMD / 1e3), fontsize=FST - 1)
a.plot(0, 0, "o", ms=8, color="tab:orange")
a.set_xlabel("\u03be от Земли, тыс. км"); a.set_ylabel("\u03b6, тыс. км")
a.set_title("Рис. 2. Разрез xz: трасса в наклонённой плоскости (%.1f\u00b0 к эклиптике)"
            % np.degrees(L.I_ECL))
a.grid(alpha=.3); a.legend(fontsize=FST - 1, loc="lower left", framealpha=.95)
fig.savefig("fig1_xz.png"); plt.close(fig)

# ---- рис. 3: трёхмерный вид ---------------------------------------------------------
fig = plt.figure(figsize=(5.6, 4.0))
a = fig.add_axes([.02, .02, .96, .92], projection="3d")
a.plot(H[0], H[1], H[2], "-", lw=1.5, color="tab:green")
a.plot(T[0], T[1], T[2], "-", lw=1.7, color="tab:blue")
a.scatter([0], [0], [0], s=42, c="tab:orange")
a.scatter([L.RHO * L.KMD / 1e3], [0], [0], s=42, marker="x", c="tab:red")
a.text2D(0.02, 0.92, "трасса", color="tab:blue", fontsize=FST)
a.text2D(0.02, 0.87, "гало у L2", color="tab:green", fontsize=FST)
a.set_xlabel("\u03be, тыс. км"); a.set_ylabel("\u03b7, тыс. км"); a.set_zlabel("\u03b6, тыс. км")
a.set_title("Рис. 3. Трасса и целевая орбита в пространстве")
a.view_init(elev=17, azim=27)
fig.savefig("fig1_3d.png"); plt.close(fig)

# ---- рис. 4: xi(t) и |v|(t) ---------------------------------------------------------
fig = plt.figure(figsize=(6.0, 5.0))
a = fig.add_subplot(2, 1, 1)
a.plot(t_d, xi_km / 1e3, lw=1.8, color="tab:blue")
a.axhline(L.RHO * L.KMD / 1e3, ls=":", lw=1.3, color="tab:red")
a.text(4, L.RHO * L.KMD / 1e3 * 1.03, "L2", color="tab:red", fontsize=FST)
a.set_ylabel("\u03be за Землёй, тыс. км")
a.set_title("Рис. 4. Ход перелёта: расстояние и скорость")
a.grid(alpha=.3)
a = fig.add_subplot(2, 1, 2)
a.plot(t_d, v_rot, lw=1.8, color="tab:blue", label="трасса (вращающийся кадр)")
a.plot(np.linspace(0.0, orb["T"] * L.DAY, len(orb["V"])),
       np.linalg.norm(orb["V"], axis=1) * L.V_U, lw=1.4, color="tab:green",
       label="целевая орбита")
a.axhline(v_orb_in, ls="-.", lw=1.3, color="tab:green")
a.axvline(t_d[-1], ls="--", lw=1.2, color="k")
a.text(t_d[-1] * .985, v_rot[-1] * .55, "вход\n%s" % BRK, fontsize=FST, ha="right")
a.set_xlabel("время, сут"); a.set_ylabel("|v|, км/с")
a.grid(alpha=.3); a.legend(fontsize=FST - 1, loc="center left")
fig.tight_layout(pad=0.6)
fig.savefig("fig2_xi_v.png"); plt.close(fig)

# ---- рис. 5: затенение на перелёте --------------------------------------------------
fig = plt.figure(figsize=(6.0, 3.4))
a = fig.add_axes([.09, .15, .89, .78])
a.plot(t_d, rho_km / 1e3, lw=1.8, color="tab:blue", label="\u03c1 до оси Солнце—Земля")
a.plot(t_d, r_pen, lw=1.5, color="0.35", ls=":", label="радиус полутени Земли r$_{pen}$(\u03be)")
a.fill_between(t_d, 0, np.where(in_pen, r_pen, 0), color="0.8", lw=0,
               label="тень: %.2f %% времени" % (100 * in_pen.mean()))
a.set_yscale("log")
a.set_xlabel("время, сут"); a.set_ylabel("\u03c1, тыс. км")
a.set_title("Рис. 5. Затенение на перелёте (стартовая точка — в тени Земли)")
a.grid(alpha=.3, which="both"); a.legend(fontsize=FST - 1, loc="upper right")
fig.savefig("fig2_shadow.png"); plt.close(fig)

print("fig1_plan.png fig1_xz.png fig1_3d.png fig2_xi_v.png fig2_shadow.png:")
print("  точек трассы %d, точек гало %d, тоф %.1f сут, dv1 %.1f м/с, dv2(по |v|) %s"
      % (tr.shape[1], H.shape[1], TOF * L.DAY, (u - L.V_LEO) * L.V_U * 1000, BRK))
print("  |v| входа %.4f км/с против %.4f на орбите; тень на перелёте %.2f%% (%d выборок)"
      % (v_rot[-1], v_orb_in, 100 * in_pen.mean(), in_pen.sum()))
