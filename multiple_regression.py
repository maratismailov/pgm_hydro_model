#!env/bin/python
"""
Расчетно-лабораторная работа 2 (тема 4.4)
Множественная линейная корреляция и регрессия
Аналог программы ЛИНЕЙН Excel — три шага:
  Шаг 1: T = f(Z)
  Шаг 2: T = f(Z, φ)
  Шаг 3: T = f(Z, φ, λ)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

# ─── Загрузка данных ──────────────────────────────────────────────────────────
csv_path = Path(__file__).parent / "stations2.csv"
df = pd.read_csv(csv_path)

MONTHS = {
    "I": "Январь", "II": "Февраль", "III": "Март",
    "IV": "Апрель", "V": "Май", "VI": "Июнь",
    "VII": "Июль", "VIII": "Август", "IX": "Сентябрь",
    "X": "Октябрь", "XI": "Ноябрь", "XII": "Декабрь",
    "year": "Год"
}

# ─── Выбор месяца / периода ───────────────────────────────────────────────────
print("Доступные периоды:")
for key, name in MONTHS.items():
    print(f"  {key:>5} — {name}")

period = input("\nВведите обозначение периода (например, VII или year): ").strip()
if period not in MONTHS:
    print(f"Период '{period}' не найден. Используется 'year'.")
    period = "year"

print(f"\n>>> Выбран период: {MONTHS[period]}\n")

# ─── Данные ───────────────────────────────────────────────────────────────────
names = df["station"].tolist()
Z   = df["z_km"].to_numpy()
phi = df["lat"].to_numpy()
lam = df["lon"].to_numpy()
T   = df[period].to_numpy()
n   = len(T)

SEP  = "═" * 62
sep2 = "─" * 62


def lineyн(X_cols: list[np.ndarray], y: np.ndarray):
    """
    Вычисляет полный набор статистик множественной регрессии
    (аналог ЛИНЕЙН Excel).

    X_cols — список столбцов-предикторов (без константы).
    Возвращает словарь со всеми параметрами.
    """
    k = len(X_cols)          # число предикторов
    n = len(y)

    # Матрица X с константой (первый столбец — единицы)
    X = np.column_stack([np.ones(n)] + X_cols)

    # МНК: β = (XᵀX)⁻¹ Xᵀy
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y   # [b0, b1, b2, ...]

    y_pred = X @ beta
    residuals = y - y_pred

    SS_tot = np.sum((y - y.mean()) ** 2)   # общая сумма квадратов
    SS_res = np.sum(residuals ** 2)        # Σ2 — остаточная
    SS_reg = SS_tot - SS_res               # Σ1 — объяснённая

    df_res = n - k - 1                     # k2 = n - k - 1
    df_reg = k

    s2  = SS_res / df_res                  # дисперсия остатков
    s   = np.sqrt(s2)                      # стандартная ошибка регрессии
    R2  = SS_reg / SS_tot                  # коэф. детерминации
    R   = np.sqrt(R2)                      # множественный R
    F   = (SS_reg / df_reg) / s2          # F-критерий

    # Ошибки коэффициентов
    se_beta = np.sqrt(np.diag(XtX_inv) * s2)

    return {
        "beta":    beta,        # [b0, b1, ...]
        "se_beta": se_beta,     # ошибки коэффициентов
        "R2":      R2,
        "R":       R,
        "s2":      s2,
        "s":       s,
        "F":       F,
        "df_reg":  df_reg,
        "df_res":  df_res,
        "SS_reg":  SS_reg,
        "SS_res":  SS_res,
        "y_pred":  y_pred,
    }


def print_step(step_num: int, predictors: list[str],
               X_cols: list[np.ndarray], res: dict):
    """Выводит таблицу результатов одного шага."""
    beta    = res["beta"]
    se_beta = res["se_beta"]
    labels  = ["b0 (свободный член)"] + [f"b{i+1} ({p})" for i, p in enumerate(predictors)]

    print(f"\n{'─'*62}")
    print(f"  ШАГ {step_num}: предикторы — {', '.join(predictors)}")
    print(f"{'─'*62}")

    print(f"\n  {'Параметр':<30} {'Значение':>10}  {'Ошибка':>10}")
    print(f"  {'─'*52}")
    for label, b, se in zip(labels, beta, se_beta):
        print(f"  {label:<30} {b:>10.4f}  {se:>10.4f}")

    print(f"\n  {'R  (множественный коэф. корр.)':.<40} {res['R']:>8.5f}")
    print(f"  {'R² (коэф. детерминации)':.<40} {res['R2']:>8.5f}")
    print(f"  {'s² (дисперсия остатков)':.<40} {res['s2']:>8.4f}")
    print(f"  {'s  (ст. ошибка регрессии)':.<40} {res['s']:>8.4f}")
    print(f"  {'F  (критерий Фишера)':.<40} {res['F']:>8.4f}")
    print(f"  {'df_reg / df_res (степени свободы)':.<40} {res['df_reg']:>4} / {res['df_res']:<4}")
    print(f"  {'Σ1 (объяснённая сумма квадратов)':.<40} {res['SS_reg']:>8.3f}")
    print(f"  {'Σ2 (остаточная сумма квадратов)':.<40} {res['SS_res']:>8.3f}")

    # Уравнение регрессии
    terms = [f"{beta[0]:.4f}"]
    for i, (p, b) in enumerate(zip(predictors, beta[1:])):
        sign = "+" if b >= 0 else "−"
        terms.append(f"{sign} {abs(b):.4f}·{p}")
    print(f"\n  Уравнение: T = {' '.join(terms)}")


# ─── Три шага ─────────────────────────────────────────────────────────────────
print(SEP)
print(f"  МНОЖЕСТВЕННАЯ ЛИНЕЙНАЯ РЕГРЕССИЯ — {MONTHS[period].upper()}")
print(SEP)

steps = [
    (["Z"],         [Z]),
    (["Z", "φ"],    [Z, phi]),
    (["Z", "φ", "λ"], [Z, phi, lam]),
]

results = []
for step_num, (pred_names, X_cols) in enumerate(steps, start=1):
    res = lineyн(X_cols, T)
    results.append(res)
    print_step(step_num, pred_names, X_cols, res)

# ─── Сравнительная таблица шагов ──────────────────────────────────────────────
print(f"\n{SEP}")
print("  СРАВНЕНИЕ ТРЁХ ШАГОВ")
print(SEP)
print(f"\n  {'Показатель':<30} {'Шаг 1':>10} {'Шаг 2':>10} {'Шаг 3':>10}")
print(f"  {'─'*60}")
for label, key, fmt in [
    ("R (корреляция)", "R",  ".4f"),
    ("R²",             "R2", ".4f"),
    ("s (ст. ошибка)", "s",  ".4f"),
    ("F-критерий",     "F",  ".3f"),
]:
    vals = [f"{r[key]:{fmt}}" for r in results]
    print(f"  {label:<30} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10}")

# ─── Лучшая модель — таблица предсказаний ────────────────────────────────────
# Выбираем шаг с максимальным R²
best_step = int(np.argmax([r["R2"] for r in results]))
best_res  = results[best_step]
best_names = steps[best_step][0]

print(f"\n  → Наилучшая модель: Шаг {best_step + 1} "
      f"(предикторы: {', '.join(best_names)}, R² = {best_res['R2']:.4f})")

print(f"\n{SEP}")
print(f"  ПРЕДСКАЗАНИЯ ПО ЛУЧШЕЙ МОДЕЛИ (Шаг {best_step + 1})")
print(SEP)
print(f"\n  {'Станция':<22} {'T факт':>8} {'T расч':>8} {'Остаток':>9}")
print(f"  {'─'*50}")
for name, t_fact, t_calc in zip(names, T, best_res["y_pred"]):
    print(f"  {name:<22} {t_fact:>8.1f} {t_calc:>8.2f} {t_fact - t_calc:>9.2f}")

rmse = np.sqrt(np.mean((T - best_res["y_pred"]) ** 2))
print(f"\n  RMSE = {rmse:.4f}°C")
print(f"\n{SEP}")
