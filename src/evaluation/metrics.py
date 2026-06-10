import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report, hamming_loss,
    label_ranking_average_precision_score,
    precision_score, recall_score,
)


def sentiment_report(y_true, y_pred, model_name: str = "") -> dict:
    labels = ["Negative", "Neutral", "Positive"]
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0, labels=labels)
    print(f"\n{'='*50}")
    if model_name:
        print(f"  {model_name}")
    print(f"  Accuracy:    {acc:.4f}")
    print(f"  F1 (macro):  {f1_macro:.4f}")
    print(f"  F1 (weighted): {f1_weighted:.4f}")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
    return {
        "model": model_name,
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,}
def plot_confusion_matrix(y_true, y_pred, labels, title: str = "", save_path: str = None):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title or "Confusion Matrix")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig

def plot_sentiment_model_comparison(results: list[dict], save_path: str = None):
    df = pd.DataFrame(results)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, metric in zip(axes, ["accuracy", "f1_macro", "f1_weighted"]):
        bars = ax.bar(df["model"], df[metric], color=sns.color_palette("Set2", len(df)))
        ax.set_ylim(0, 1)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylabel("Score")
        ax.tick_params(axis="x", rotation=15)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    plt.suptitle("Sentiment Model Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
def plot_train_vs_f1(results: list[dict], save_path: str = None):
    df = pd.DataFrame(results)
    fig, ax = plt.subplots(figsize=(7, 5))
    for _, row in df.iterrows():
        ax.scatter(row.get("train_time", 0), row["f1_macro"], s=100, label=row["model"])
        ax.annotate(row["model"], (row.get("train_time", 0), row["f1_macro"]),
                    textcoords="offset points", xytext=(5, 3), fontsize=8)
    ax.set_xlabel("Train Time (s)")
    ax.set_ylabel("F1 Macro")
    ax.set_title("Train Time vs F1 Macro (Pareto Frontier)")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig

def tag_report(Y_true, Y_pred, tag_names: list[str], model_name: str = "") -> dict:
    hl = hamming_loss(Y_true, Y_pred)
    lrap = label_ranking_average_precision_score(Y_true, Y_pred)
    subset_acc = accuracy_score(Y_true, Y_pred)
    per_label_f1 = f1_score(Y_true, Y_pred, average=None, zero_division=0)
    per_label_prec = precision_score(Y_true, Y_pred, average=None, zero_division=0)
    per_label_rec = recall_score(Y_true, Y_pred, average=None, zero_division=0)

    print(f"\n{'='*50}")
    if model_name:
        print(f"  {model_name}")
    print(f"  Hamming Loss:     {hl:.4f}")
    print(f"  LRAP:             {lrap:.4f}")
    print(f"  Subset Accuracy:  {subset_acc:.4f}")
    tag_df = pd.DataFrame({
        "tag": tag_names,
        "precision": per_label_prec,
        "recall": per_label_rec,
        "f1": per_label_f1,
    })
    print(tag_df.to_string(index=False))
    return {
        "model": model_name,
        "hamming_loss": hl,
        "lrap": lrap,
        "subset_accuracy": subset_acc,
        "per_label_f1": per_label_f1.tolist(),
        "tag_names": tag_names,
    }

def plot_per_label_f1(results: list[dict], tag_names: list[str], save_path: str = None):
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(tag_names))
    width = 0.8 / len(results)
    for i, r in enumerate(results):
        offset = (i - len(results) / 2) * width + width / 2
        ax.bar(x + offset, r["per_label_f1"], width=width * 0.9, label=r["model"])
    ax.set_xticks(x)
    ax.set_xticklabels(tag_names, rotation=25, ha="right")
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-label F1 Across Tag Classifiers")
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_tag_model_comparison(results: list[dict], save_path: str = None):
    df = pd.DataFrame(results)[["model", "hamming_loss", "lrap", "subset_accuracy"]]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, (col, title) in zip(axes, [
        ("hamming_loss", "Hamming Loss (↓)"),
        ("lrap", "LRAP (↑)"),
        ("subset_accuracy", "Subset Accuracy (↑)"),
    ]):
        bars = ax.bar(df["model"], df[col], color=sns.color_palette("Set1", len(df)))
        ax.set_title(title)
        ax.set_ylabel("Score")
        ax.tick_params(axis="x", rotation=15)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    plt.suptitle("Tag Classifier Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_match_score_distribution(results_df: pd.DataFrame, query: str = "", save_path: str = None):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(
        results_df["name"].str[:30] if "name" in results_df.columns else results_df["business_id"],
        results_df["match_score"],
        color=sns.color_palette("Blues_r", len(results_df)),
    )
    ax.set_xlabel("Match Score (0–10)")
    ax.set_title(f"Top Restaurants\n{query[:60]}")
    ax.invert_yaxis()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig

def plot_score_breakdown(row: pd.Series, save_path: str = None):
    components = {
        "Sentiment": row.get("score_sentiment", 0),
        "Tag Match": row.get("score_tag_match", 0),
        "Stars": row.get("score_stars", 0),
        "Volume": row.get("score_volume", 0),
        "Cuisine": row.get("score_cuisine", 0),
    }
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(components.keys(), components.values(), color=sns.color_palette("Set3", 5))
    ax.set_ylabel("Score contribution")
    ax.set_title(f"Score Breakdown: {row.get('name', '')}")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
def build_comparison_table(results: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(results).set_index("model")
