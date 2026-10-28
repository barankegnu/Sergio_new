#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Рисунки перелёта НОО -> гало у L2 по тем же числам, что и отчёт.

Цель, угол старта, dC и время перелёта ЧИТАЮТСЯ из блока AIM в result.txt (как
l2_perturb.py и make_docx.js): трасса интегрируется той же моделью l2_transfer.py,
поэтому рисунок не может разойтись с расчётом.

Раньше всё было на одном авто-масштабном кадре с equal aspect - и это был неправильный
с бюджетом кадра выбор: трасса 38 % времени проводит со стороны Солнца (xi до -997 тыс.
км), eta доходит до 961 тыс. км, тогда как НОО это 6,6 тыс. км, а полутень Земли у L2 -
13,4 тыс. км, т.е. доли процента кадра. Поэтому теперь семь односюжетных рисунков с прицельными
осями:

  fig1_overview  весь перелёт сверху;  fig2_start  старт (первые 150 тыс. км);
  fig3_entry   подлёт и точка входа;     fig4_xz    разрез xz с конусом полутени;
  fig5_shadow  rho(t) против границы полутени; fig6_xi, fig7_v - ход xi(t) и |v|(t)


Запуск:  python3 plot_traj.py        (-> семь .png, ~35 с)
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


Z_TGT = num(r"целевая орбита: Az = (\d+) km") / L.KMD
TH = num(r"AIM z = [-\d.e+]+ theta = ([-\d.]+)")
DC = num(r"AIM z = [-\d.e+]+ theta = [-\d.]+ dC = ([-\d.e+]+)")
TOF = num(r"AIM z = [-\d.e+]+ theta = [-\d.]+ dC = [-\d.e+]+ tof_TU = ([-\d.]+)")
AZ_MIN = num(r"Az_min = (\d+) km")                    # км
LEO_R = (L.RE + L.H_PARK) / L.KM                     # км

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                     "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.linewidth": .9,
                     "savefig.dpi": 200, "figure.dpi": 110, "legend.framealpha": .95})

orb = L.halo_family([Z_TGT], n=900)[0]
u = L.u_for_jacobi(TH, orb["C"] - DC)
tr = L.fly_traj(L.dep_state(TH, u), TOF, n=3000, rtol=1e-11)
t_d = np.linspace(0.0, TOF, tr.shape[1]) * L.DAY     # сут

X = (tr[0] - (1 - L.MU)) * L.KMD / 1e3
Y = tr[1] * L.KMD / 1e3
Z = tr[2] * L.KMD / 1e3
# стартовый участок равномерная выборка почти не показывает (в радиусе 40 тыс. км лежат
# 2 точки из 3000), поэтому первые ~150 тыс. км пути интегрируются отдельно и плотно:
# тот же участок той же трассы, но с мелким шагом по времени
R_ZOOM = 150.0                                        # тыс. км
r_geo = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)             # расстояние от Земли, тыс. км
t_early = TOF * int(np.argmax(r_geo > R_ZOOM)) / (tr.shape[1] - 1)
tr0 = L.fly_traj(L.dep_state(TH, u), max(t_early, 1e-3), n=900, rtol=1e-11)

HX = (orb["P"][:, 0] - (1 - L.MU)) * L.KMD / 1e3
HY = orb["P"][:, 1] * L.KMD / 1e3
HZ = orb["P"][:, 2] * L.KMD / 1e3
V = np.linalg.norm(tr[3:6], axis=0) * L.V_U          # км/с во вращающемся кадре
RHOkm = np.hypot(tr[1], tr[2]) * L.KMD              # км до оси Солнце-Земля
r_pen = (L.RE / L.KM + (696000.0 + L.RE / L.KM) / (L.AU / L.KM)
         * np.maximum(X * 1e3, 0.0)) / 1e3          # тыс. км
in_pen = (X > 0) & (RHOkm / 1e3 < r_pen)
j_in = int(np.argmin((HX - X[-1])**2 + (HY - Y[-1])**2 + (HZ - Z[-1])**2))
v_orb_in = np.linalg.norm(orb["V"][j_in]) * L.V_U
BRK = "тормозной" if num(r"превышение скорости трассы над орбитой = ([-\d.]+) мм/с") >= 0 \
    else "разгонный"
MOON = 384400.0 / 1e3
X0 = (tr0[0] - (1 - L.MU)) * L.KMD / 1e3
Y0 = tr0[1] * L.KMD / 1e3
k0 = int(np.argmax(X > 1150)) if (X > 1150).any() else 0


def frame(a, xlab, ylab, title):
    a.set_xlabel(xlab); a.set_ylabel(ylab); a.set_title(title); a.grid(alpha=.35)


def earth(a):
    a.plot(0, 0, "o", ms=9, mfc="tab:orange", mec="k", mew=.7, zorder=5)


def save(fig, name):
    fig.tight_layout(pad=0.55); fig.savefig(name); plt.close(fig)


# = рис. 1: обзор всего перелёта ======================================================
fig, a = plt.subplots(figsize=(6.4, 4.6))
a.plot(X, Y, "-", lw=2.0, color="tab:blue", zorder=3, label="трасса перелёта")
a.plot(HX, HY, "-", lw=1.8, color="tab:green", zorder=3, label="гало у L2 (цель)")
a.add_patch(plt.Circle((0, 0), MOON, fill=False, ls=":", lw=1.1, color="0.45"))
a.text(MOON * .55, -MOON * .82, "орбита Луны", fontsize=10, color="0.35", rotation=-25)
earth(a)
a.plot(L.RHO * L.KMD / 1e3, 0, "x", ms=11, mew=2.2, color="tab:red", zorder=5, label="L2")
a.annotate("Земля", (0, 0), xytext=(60, 190), fontsize=10, arrowprops=dict(arrowstyle="->", lw=1))
a.annotate("вход на гало", (X[-1], Y[-1]), xytext=(X[-1] - 520, Y[-1] - 330), fontsize=10,
           arrowprops=dict(arrowstyle="->", lw=1))
a.annotate("", xy=(-1150, 0), xytext=(-870, 0), arrowprops=dict(arrowstyle="->", lw=1.6, color="0.3"))
a.text(-1140, 60, "к Солнцу", fontsize=10, color="0.3")
a.set_xlim(-1250, 1800); a.set_ylim(-1150, 1150); a.set_aspect("equal")
frame(a, "\u03be от Земли, тыс. км", "\u03b7 (в эклиптике), тыс. км",
      "Рис. 1. Весь перелёт: вид сверху на эклиптику")
a.legend(fontsize=10, loc="lower left")
save(fig, "fig1_overview.png")

# = рис. 2: старт =====================================================================
fig, a = plt.subplots(figsize=(5.2, 4.6))
a.plot(X0, Y0, "-", lw=2.4, color="tab:blue", label="первые %.0f тыс. км пути" % R_ZOOM)
th = np.linspace(0, 2 * np.pi, 200)
a.plot(LEO_R / 1e3 * np.cos(th), LEO_R / 1e3 * np.sin(th), "-", lw=2.0, color="k",
       label="НОО 200 км (r = %.1f тыс. км)" % (LEO_R / 1e3))
a.add_patch(plt.Circle((0, 0), AZ_MIN / 2e3, fill=True, color="0.82", alpha=.9, lw=0,
                       label="полутень Земли у L2 (%.1f тыс. км)" % (AZ_MIN / 1e3)))
earth(a)
lim = R_ZOOM * 1.15
a.set_xlim(-lim, lim); a.set_ylim(-lim, lim); a.set_aspect("equal")
frame(a, "\u03be, тыс. км", "\u03b7, тыс. км", "Рис. 2. Старт: НОО и начало трассы")
a.legend(fontsize=9, loc="upper left")
TXT2 = (0.0, -lim * .93)                       # вниз по центру, внутри рамки
a.text(*TXT2, "включение 1-го импульса - с ночной стороны Земли", fontsize=9.5,
       color="0.25", ha="center", va="bottom")
save(fig, "fig2_start.png")

# = рис. 3: подлёт и вход на гало =====================================================
fig, a = plt.subplots(figsize=(6.0, 4.4))
a.plot(HX, HY, "-", lw=2.0, color="tab:green", label="гало (цель)")
a.plot(X[k0:], Y[k0:], "-", lw=2.4, color="tab:blue",
       label="подлёт (\u03be > 1,15 млн км)")
a.plot(X[-1], Y[-1], "*", ms=17, color="tab:red", mec="k", mew=.6, zorder=5,
       label="точка входа, промах %s км" % re.search(r"промах по месту прицеливания\s*=\s*(\d+) km", RES).group(1))
a.plot(L.RHO * L.KMD / 1e3, 0, "x", ms=11, mew=2.2, color="tab:red")
a.text(L.RHO * L.KMD / 1e3 - 45, 70, "L2", fontsize=10, color="tab:red")
XL3 = (min(X[k0], HX.min()) - 70, max(X[k0], HX.max()) + 70)
a.set_xlim(*XL3); a.set_ylim(-760, 760); a.set_aspect("equal")
frame(a, "\u03be, тыс. км", "\u03b7, тыс. км",
      "Рис. 3. Цель: гало Az = %.0f тыс. км и вход на него" % (Z_TGT * L.KMD / 1e3))
a.legend(fontsize=9, loc="lower right", framealpha=.95)
save(fig, "fig3_entry.png")

# = рис. 4: разрез xz с конусом полутени =============================================
fig, a = plt.subplots(figsize=(6.6, 3.9))
xi = np.linspace(0, 1750, 400)
rp = (L.RE / L.KM + (696000.0 + L.RE / L.KM) / (L.AU / L.KM) * xi * 1e3) / 1e3
a.fill_between(xi, -rp, rp, color="0.8", alpha=.95, lw=0, label="конус полутени Земли")
a.plot(HX, HZ, "-", lw=2.0, color="tab:green", label="гало")
a.plot(X, Z, "-", lw=2.1, color="tab:blue", label="трасса")
for n, sg in enumerate((1, -1)):
    a.axhline(sg * AZ_MIN / 1e3, ls="--", lw=1.4, color="k",
              label="порог выхода из тени Az_min = %.1f тыс. км" % (AZ_MIN / 1e3)
              if sg > 0 else "_nolegend_")
TXT4 = (1900, -252)                            # вниз-внутрь, справа
a.text(*TXT4, "у самой оси Солнце-Земля полутень\nЗемли: \u00b113,4 тыс. км у L2\n"
       "гало: min \u03c1 = %.1f тыс. км" % (orb["rho_min"] / 1e3),
       fontsize=9.5, color="0.15", ha="right", va="bottom")
earth(a)
a.set_xlim(-1100, 2000); a.set_ylim(-260, 260)
frame(a, "\u03be от Земли, тыс. км", "\u03b6 (вверх от эклиптики), тыс. км",
      "Рис. 4. Разрез xz (оси в масштабе между собой не выдержаны)")
a.legend(fontsize=9.5, loc="lower left")
save(fig, "fig4_xz.png")

# = рис. 5: затенение вдоль трассы ====================================================
fig, a = plt.subplots(figsize=(6.2, 3.6))
a.plot(t_d, RHOkm / 1e3, lw=2.2, color="tab:blue", label="\u03c1 до оси Солнце\u2014Земля")
a.plot(t_d, r_pen, lw=1.9, color="0.35", ls=":", label="r$_{pen}$(\u03be) — граница полутени")
a.fill_between(t_d, 1e-1, np.where(in_pen, r_pen, 1e-1), color="0.8", lw=0,
               label="тень: %.2f %% времени (%d выборок)" % (100 * in_pen.mean(), in_pen.sum()))
a.set_yscale("log"); a.set_ylim(1, 1200)
frame(a, "время, сут", "\u03c1, тыс. км", "Рис. 5. Затенение на перелёте (лог-шкала)")
a.legend(fontsize=9.5, loc="lower right", framealpha=.95)
save(fig, "fig5_shadow.png")

# = рис. 6: ход xi(t) ================================================================
fig, a = plt.subplots(figsize=(6.2, 3.4))
a.plot(t_d, X, lw=2.2, color="tab:blue")
a.axhline(L.RHO * L.KMD / 1e3, ls=":", lw=1.6, color="tab:red")
a.text(6, L.RHO * L.KMD / 1e3 * 1.04, "L2 (%.0f тыс. км)" % (L.RHO * L.KMD / 1e3),
       color="tab:red", fontsize=10)
a.axhline(0, lw=.9, color="0.4")
a.text(6, 60, "ниже нуля - трасса со стороны Солнца (%.0f %% времени)" % (100 * (X < 0).mean()),
       fontsize=9.5, color="0.25")
a.axvline(t_d[-1], ls="--", lw=1.3, color="k")
a.text(t_d[-1] * .97, X.min() * .55, "вход\nна гало", fontsize=9.5, ha="right")
frame(a, "время, сут", "\u03be за Землёй, тыс. км", "Рис. 6. Ход перелёта: расстояние \u03be(t)")
save(fig, "fig6_xi.png")

# = рис. 7: ход |v|(t) ==============================================================
fig, a = plt.subplots(figsize=(6.2, 3.6))
a.plot(t_d, V, lw=2.2, color="tab:blue", label="трасса (вращающийся кадр)")
a.plot(np.linspace(0.0, orb["T"] * L.DAY, len(HX)),
       np.linalg.norm(orb["V"], axis=1) * L.V_U, lw=1.8, color="tab:green",
       label="скорость на целевой орбите")
a.axhline(v_orb_in, ls="-.", lw=1.6, color="tab:green")
a.axvline(t_d[-1], ls="--", lw=1.4, color="k")
a.annotate("вход: %s\n|v| трассы %.3f против %.3f км/с" % (BRK, V[-1], v_orb_in),
           xy=(t_d[-1], V[-1]), xytext=(t_d[-1] * .30, max(V[-1] * .22, 0.09)), fontsize=10,
           arrowprops=dict(arrowstyle="->", lw=1))
a.set_yscale("log")
frame(a, "время, сут", "|v|, км/с (лог-шкала)", "Рис. 7. Скорость: |v|(t) и скорость орбиты в точке касания")
a.set_ylim(min(V) * .8, max(V) * 1.4)
a.legend(fontsize=9.5, loc="upper right")
save(fig, "fig7_v.png")

# ---- проверки: в каждый кадр данные должны попадать ================================
assert len(X0) > 50 and np.hypot(X0, Y0).max() <= R_ZOOM * 1.35, "рис. 2: старт пуст"
assert 1100 <= X[k0:].min() and X[k0:].max() <= 1800, "рис. 3: подлёт вышел за окно"
assert -260 < min(Z.min(), HZ.min()) and max(Z.max(), HZ.max()) < 260, "рис. 4: данные вне осей"
assert 1 < RHOkm.min() / 1e3 < 1200, "рис. 5: rho вне лог-шкалы"
assert min(V) < v_orb_in < max(V), "рис. 7: |v| цели вне диапазона трассы"
assert -lim < TXT2[1] < lim and abs(TXT2[0]) < lim, "рис. 2: подпись вне рамки"
assert XL3[0] < X[k0] and X[-1] < XL3[1], "рис. 3: вход вышел за расширенное окно"
assert -260 < TXT4[1] < 260 and 1700 < TXT4[0] < 2000, "рис. 4: подпись вне рамки"

print("рис. 1-7: fig1_overview fig2_start fig3_entry fig4_xz fig5_shadow fig6_xi fig7_v")
print("  тоф %.1f сут, dv1 %.1f м/с, dv2 %.1f м/с (%s); трасса xi %.0f..%.0f тыс. км,"
      " %d %% времени со стороны Солнца"
      % (TOF * L.DAY, (u - L.V_LEO) * L.V_U * 1000, num(r"dv2 \(второй импульс\)\s*=\s*([\d.]+)"),
         BRK, X.min(), X.max(), round(100 * (X < 0).mean())))
print("  тень на перелёте %.2f%% (%d выборок), rho_min %.0f км, порог Az_min %.0f км,"
      " точек старта %d" % (100 * in_pen.mean(), in_pen.sum(), RHOkm.min(), AZ_MIN, len(X0)))
