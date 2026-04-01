"""
eda.py
------
Exploratory Data Analysis for the Iris dataset.
Run this BEFORE training to understand the data.

Plots:
  1. Class distribution (bar chart)
  2. Feature distributions per class (violin plots)
  3. Pairplot — all features vs each other, colored by class
  4. Correlation heatmap
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


PALETTE = {"setosa": "#378ADD", "versicolor": "#1D9E75", "virginica": "#D85A30"}


def plot_class_distribution(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["species"].value_counts()
    bars = ax.bar(counts.index, counts.values,
                  color=[PALETTE[s] for s in counts.index],
                  edgecolor="white", width=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(val), ha="center", va="bottom", fontsize=11)
    ax.set_title("Class Distribution — Iris Dataset", fontsize=13, fontweight="bold")
    ax.set_ylabel("Sample count")
    ax.set_ylim(0, max(counts.values) + 8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("eda_class_distribution.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved: eda_class_distribution.png")


def plot_feature_violin(df):
    feature_cols = [c for c in df.columns if c != "species"]
    fig, axes = plt.subplots(1, len(feature_cols), figsize=(14, 5))
    fig.suptitle("Feature Distributions by Class", fontsize=13, fontweight="bold")
    for ax, feat in zip(axes, feature_cols):
        sns.violinplot(
            data=df, x="species", y=feat,
            palette=PALETTE, ax=ax, inner="quartile", linewidth=0.8
        )
        ax.set_title(feat, fontsize=10)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=20)
        ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("eda_violin.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved: eda_violin.png")


def plot_pairplot(df):
    g = sns.pairplot(
        df, hue="species",
        palette=PALETTE,
        plot_kws={"alpha": 0.65, "s": 30, "edgecolor": "none"},
        diag_kind="kde"
    )
    g.figure.suptitle("Pairplot — All Features vs Each Other", y=1.02,
                       fontsize=13, fontweight="bold")
    g.figure.savefig("eda_pairplot.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved: eda_pairplot.png")


def plot_correlation_heatmap(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    corr = df.drop(columns="species").corr()
    sns.heatmap(
        corr, annot=True, fmt=".2f",
        cmap="coolwarm", center=0,
        linewidths=0.5, linecolor="white",
        ax=ax, cbar_kws={"shrink": 0.8}
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("eda_correlation.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved: eda_correlation.png")


def run_eda(data):
    df = data["df"]
    print("\n── EDA Summary ──────────────────────────────")
    print(df.describe().round(3).to_string())
    print(f"\nMissing values: {df.isnull().sum().sum()}")

    plot_class_distribution(df)
    plot_feature_violin(df)
    plot_pairplot(df)
    plot_correlation_heatmap(df)


if __name__ == "__main__":
    from data_loader import load_data
    run_eda(load_data())
