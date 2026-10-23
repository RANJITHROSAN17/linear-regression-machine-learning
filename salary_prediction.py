"""Salary vs. years-of-experience: simple linear regression with scikit-learn.

Usage:
    python salary_prediction.py            # prints metrics, writes plots to docs/
    python salary_prediction.py --show     # also opens the plots interactively
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "exp_vs_salary.csv"
DOCS = ROOT / "docs"


def load_data(path: Path = DATA) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    X = df.iloc[:, :-1].to_numpy(dtype=float)  # years of experience
    y = df.iloc[:, -1].to_numpy(dtype=float)   # salary
    return X, y


def train(X: np.ndarray, y: np.ndarray, test_size: float = 1 / 3, seed: int = 0):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "coef": float(model.coef_[0]),
        "intercept": float(model.intercept_),
        "r2_test": float(r2_score(y_test, y_pred)),
        "mae_test": float(mean_absolute_error(y_test, y_pred)),
    }
    return model, metrics, (X_train, X_test, y_train, y_test)


def plot(model, split, out_dir: Path = DOCS, show: bool = False) -> None:
    import matplotlib.pyplot as plt

    X_train, X_test, y_train, y_test = split
    out_dir.mkdir(exist_ok=True)
    line_x = np.linspace(X_train.min(), X_train.max(), 100).reshape(-1, 1)
    for name, X_, y_ in (("training", X_train, y_train), ("test", X_test, y_test)):
        plt.figure(figsize=(7, 4.5))
        plt.scatter(X_, y_, s=6, alpha=0.4, color="tab:red", label=f"{name} data")
        plt.plot(line_x, model.predict(line_x), color="tab:blue", label="fitted line")
        plt.title(f"Salary vs. experience ({name} set)")
        plt.xlabel("Years of experience")
        plt.ylabel("Salary")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{name}_set.png", dpi=120)
    if show:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--show", action="store_true", help="open plots interactively")
    args = parser.parse_args()
    if not args.show:
        matplotlib.use("Agg")

    X, y = load_data()
    model, metrics, split = train(X, y)
    print(f"salary ≈ {metrics['coef']:.2f} × years + {metrics['intercept']:.2f}")
    print(f"R² (test)  = {metrics['r2_test']:.4f}")
    print(f"MAE (test) = {metrics['mae_test']:.1f}")
    plot(model, split, show=args.show)
    print(f"plots written to {DOCS.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
