#!env/bin/python
"""
Расчетно-лабораторная работа 3 (тема лекций 3)
Расчет параметров парной линейной регрессии и корреляции

Данные:
  y  — расход воды р. Нарын – г. Нарын (м³/с)
  x  — климатические предикторы (температура / осадки)
       по МС Нарын и МС Тянь-Шань

Задание: для выбранного месяца рассчитать парную регрессию
ỹ = b0 + b1·x ± S и вывести результаты по форме Таблицы 1 (ЛИНЕЙН).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

# ─── Загрузка данных ──────────────────────────────────────────────────────────
csv_path = Path(__file__).parent / "naryn_data.csv"
df = pd.read_csv(csv_path)

MONTH_KEYS = ['I','II','III','IV','V','VI','VII','VIII','IX','X','XI','XII']
MONTH_NAMES = {
    'I':'Январь','II':'Февраль','III':'Март','IV':'Апрель',
    'V':'Май','VI':'Июнь','VII':'Июль','VIII':'Август',
    'IX':'Сентябрь','X':'Октябрь','XI':'Ноябрь','XII':'Декабрь',
}

PREDICTORS = {
    '1': ('naryn_t',  'Температура воздуха, МС Нарын'),
    '2': ('naryn_p',  'Осадки, МС Нарын'),
    '3': ('ts_t',     'Температура воздуха, МС Тянь-Шань'),
    '4': ('ts_p',     'Осадки, МС Тянь-Шань'),
}

# ─── Выбор параметров ─────────────────────────────────────────────────────────
print("=" * 55)
print("  РАБОТА 3 — Парная регрессия: сток р. Нарын")
print("=" * 55)

print("\nМесяцы:")
for k, name in MONTH_NAMES.items():
    print(f"  {k:>5} — {name}")
month = input("\nВведите месяц для y (расход воды): ").strip()
if month not in MONTH_KEYS:
    print(f"Месяц '{month}' не найден. Используется 'VII'.")
    month = 'VII'

print("\nМесяц для x (предиктор) — может отличаться от месяца y")
print("(например, осадки за апрель предсказывают сток за июль)")
month_x = input(f"Введите месяц для x [Enter = тот же '{month}']: ").strip()
if month_x not in MONTH_KEYS:
    month_x = month

print("\nПредикторы (x):")
for key, (_, desc) in PREDICTORS.items():
    print(f"  {key} — {desc}")
pred_key = input("\nВыберите предиктор (1–4): ").strip()
if pred_key not in PREDICTORS:
    pred_key = '1'

pred_col, pred_desc = PREDICTORS[pred_key]

# ─── Подготовка данных ────────────────────────────────────────────────────────
col_y = f'Q_{month}'
col_x = f'{pred_col}_{month_x}'

sub = df[['year', col_y, col_x]].dropna()
sub = sub[sub[col_y] > 0]   # убираем нулевые/отсутствующие расходы

y = sub[col_y].to_numpy()
x = sub[col_x].to_numpy()
n = len(y)

years = sub['year'].astype(int).tolist()

# ─── Вычисление статистик (аналог ЛИНЕЙН) ────────────────────────────────────
b1, b0, r, p_value, sb1 = stats.linregress(x, y)

y_pred    = b0 + b1 * x
residuals = y - y_pred

SS1 = np.sum((y_pred - y.mean()) ** 2)
SS2 = np.sum(residuals ** 2)
k2  = n - 2
s2  = SS2 / k2
s   = np.sqrt(s2)
r2  = r ** 2
F   = SS1 / s2

x_mean = x.mean()
sb0 = s * np.sqrt(1/n + x_mean**2 / np.sum((x - x_mean)**2))

SEP = "═" * 55

# ─── Вывод ───────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print(f"  y = расход воды, {MONTH_NAMES[month]}")
print(f"  x = {pred_desc}, {MONTH_NAMES[month_x]}")
print(f"  n = {n} лет ({years[0]}–{years[-1]})")
print(SEP)

print(f"""
  Таблица 1 — ЛИНЕЙН
  ┌{'─'*24}┬{'─'*24}┐
  │  {'A':<22}│  {'B':<22}│
  ├{'─'*24}┼{'─'*24}┤
  │  b1  = {b1:>14.3f}    │  b0  = {b0:>14.3f}    │
  │  Sb1 = {sb1:>14.3f}    │  Sb0 = {sb0:>14.3f}    │
  │  r²  = {r2:>14.3f}    │  S²  = {s2:>14.3f}    │
  │  F   = {F:>14.3f}    │  k2  = {k2:>14.3f}    │
  │  Σ1  = {SS1:>14.3f}    │  Σ2  = {SS2:>14.3f}    │
  └{'─'*24}┴{'─'*24}┘""")

sign = '+' if b0 >= 0 else '−'
print(f"\n  Уравнение регрессии:")
print(f"    Q = {b1:.4f}·x {sign} {abs(b0):.4f}")

strength = 'сильная' if abs(r) > 0.7 else ('умеренная' if abs(r) > 0.4 else 'слабая')
direction = 'обратная' if r < 0 else 'прямая'
signif = 'значима (p < 0.05)' if p_value < 0.05 else 'незначима'
print(f"\n  Интерпретация:")
print(f"    r  = {r:+.4f}  → связь {strength}, {direction}")
print(f"    r² = {r2:.4f}   → предиктор объясняет {r2*100:.1f}% дисперсии стока")
print(f"    S  = {s:.4f}  м³/с — стандартная ошибка регрессии")
print(f"    p  = {p_value:.2e}  → {signif}")

# ─── Таблица предсказаний ─────────────────────────────────────────────────────
print(f"\n{SEP}")
print(f"  ФАКТИЧЕСКИЕ И РАСЧЁТНЫЕ ЗНАЧЕНИЯ")
print(SEP)
print(f"\n  {'Год':>6}  {'x':>10}  {'Q факт':>10}  {'Q расч':>10}  {'Остаток':>10}")
print(f"  {'─'*52}")
for yr, xi, yi, yi_p in zip(years, x, y, y_pred):
    print(f"  {yr:>6}  {xi:>10.2f}  {yi:>10.1f}  {yi_p:>10.2f}  {yi - yi_p:>10.2f}")

rmse = np.sqrt(np.mean((y - y_pred)**2))
print(f"\n  RMSE = {rmse:.3f} м³/с")
print(f"\n{SEP}")
