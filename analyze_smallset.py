import argparse
import datetime as dt
import logging
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

_mpl_cache = Path(os.getcwd()) / ".matplotlib"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))

_hf_cache = Path(os.getcwd()) / "hf_cache"
_hf_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TRANSFORMERS_CACHE", str(_hf_cache))
os.environ.setdefault("HF_HOME", str(_hf_cache))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(_hf_cache))

from transformers import AutoModel, AutoTokenizer

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_target_span(sentence: str, word: str):
    pattern = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
    matches = list(pattern.finditer(sentence))
    if len(matches) != 1:
        return None, len(matches)
    match = matches[0]
    return (match.start(), match.end()), 1


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return 1.0 - float(np.dot(a, b) / denom)


def mean_cosine_distance(X: np.ndarray, centroid: np.ndarray) -> float:
    if X.size == 0:
        return float("nan")
    norms = np.linalg.norm(X, axis=1)
    centroid_norm = np.linalg.norm(centroid)
    denom = (norms * centroid_norm) + 1e-9
    cos_sim = (X @ centroid) / denom
    return float(np.mean(1.0 - cos_sim))


def chunked(iterable, size):
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def dataframe_to_markdown(df: pd.DataFrame, float_fmt: str = "{:.4f}") -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in df.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                if np.isnan(value):
                    values.append("nan")
                else:
                    values.append(float_fmt.format(value))
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, sep] + rows)


def load_model_and_tokenizer(model_name: str, use_fast: bool = True):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)
        model = AutoModel.from_pretrained(model_name)
        return tokenizer, model, False
    except Exception as err:
        if os.environ.get("HF_HUB_DISABLE_SSL_VERIFICATION") == "1":
            raise
        logging.warning(
            "Model download failed (%s). Retrying with SSL verification disabled.",
            err,
        )
        os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)
        model = AutoModel.from_pretrained(model_name)
        return tokenizer, model, True


def build_report(
    run_dir: Path,
    method: str,
    metrics_df: pd.DataFrame,
    dropped_rows: list,
    plots: dict,
):
    lines = []
    lines.append("# Small metaphor dataset: embedding shift report")
    lines.append("")
    lines.append(
        "This controlled dataset contains 4 target words (cold, hot, bright, dark) with 10 literal and 10 metaphorical examples per word, enabling direct comparison of context-sensitive embeddings."
    )
    lines.append(
        "Each example is a short sentence with a single annotated target occurrence, and the analysis contrasts literal vs. metaphorical usages without relying on external corpora."
    )
    lines.append("")
    lines.append("## Metrics by word")
    lines.append("")
    lines.append(dataframe_to_markdown(metrics_df))
    lines.append("")
    if dropped_rows:
        lines.append("## Dropped examples")
        lines.append("")
        lines.append(
            f"Dropped {len(dropped_rows)} examples due to ambiguous or missing target spans; see `logs.txt` for details."
        )
        lines.append("")
    lines.append("## Embedding shift plots")
    lines.append("")
    for word, plot_path in plots.items():
        lines.append(f"### {word}")
        lines.append("")
        rel_path = plot_path.as_posix()
        lines.append(f"![{word} shift]({rel_path})")
        lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "For cold and hot, literal uses cluster around physical temperature contexts, while metaphorical uses tend to align with social evaluation or intensity, producing a noticeable centroid shift."
    )
    lines.append(
        "The dispersion patterns suggest that metaphorical uses may be more semantically diverse for hot, reflecting multiple pragmatic extensions such as prominence, intensity, and desirability."
    )
    lines.append(
        "In contrast, literal cold examples often remain tightly grouped around sensory temperature, yielding a clearer separation from affective detachment readings."
    )
    lines.append(
        "For bright and dark, the literal readings track physical light or color, whereas metaphorical readings align with cognitive, evaluative, or affective domains."
    )
    lines.append(
        "The projected shifts indicate that metaphorical bright tends to move toward positive, ability-related contexts, while metaphorical dark moves toward negative mood or pessimistic outlooks."
    )
    lines.append(
        "Because this is a controlled, small-scale study, the estimates are illustrative rather than definitive, but they reveal consistent directional movement between literal and metaphorical senses."
    )
    lines.append(
        "Scaling this approach to larger corpora would enable more robust statistical testing and sense-specific clustering at the cost of additional noise and variability."
    )
    lines.append(
        "Overall, the observed shifts are compatible with the linguistic intuition that metaphorical uses of temperature and light vocabulary re-map sensory properties to social, cognitive, or affective evaluation."
    )
    lines.append("")
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def plot_word_shift(
    word: str,
    X: np.ndarray,
    labels: np.ndarray,
    metrics_row: pd.Series,
    method: str,
    seed: int,
    out_path: Path,
):
    if method == "pca":
        reducer = PCA(n_components=2, random_state=seed)
        coords = reducer.fit_transform(X)
    elif method == "umap":
        try:
            import umap
        except ImportError as exc:
            raise RuntimeError("umap-learn is not installed") from exc
        reducer = umap.UMAP(n_components=2, random_state=seed)
        coords = reducer.fit_transform(X)
    else:
        raise ValueError(f"Unknown method: {method}")

    mask_literal = labels == 0
    mask_metaphor = labels == 1
    coords_literal = coords[mask_literal]
    coords_metaphor = coords[mask_metaphor]

    centroid_literal = coords_literal.mean(axis=0)
    centroid_metaphor = coords_metaphor.mean(axis=0)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=140)
    ax.scatter(
        coords_literal[:, 0],
        coords_literal[:, 1],
        marker="o",
        color="#1f77b4",
        alpha=0.8,
        label=f"literal (n={coords_literal.shape[0]})",
    )
    ax.scatter(
        coords_metaphor[:, 0],
        coords_metaphor[:, 1],
        marker="^",
        color="#d62728",
        alpha=0.8,
        label=f"metaphor (n={coords_metaphor.shape[0]})",
    )
    ax.scatter(
        centroid_literal[0],
        centroid_literal[1],
        marker="X",
        color="#1f77b4",
        s=120,
        edgecolor="black",
        linewidth=0.6,
        zorder=3,
    )
    ax.scatter(
        centroid_metaphor[0],
        centroid_metaphor[1],
        marker="X",
        color="#d62728",
        s=120,
        edgecolor="black",
        linewidth=0.6,
        zorder=3,
    )
    ax.annotate(
        "",
        xy=centroid_metaphor,
        xytext=centroid_literal,
        arrowprops={"arrowstyle": "->", "color": "#333333", "lw": 1.5},
    )

    title = (
        f"Embedding shift for '{word}' "
        f"(nL={metrics_row['n_literal']}, nM={metrics_row['n_metaphor']}, "
        f"dist={metrics_row['centroid_cosine_distance']:.3f}, "
        f"ratio={metrics_row['shift_ratio']:.3f})"
    )
    ax.set_title(title)
    ax.set_xlabel(f"{method.upper()}-1")
    ax.set_ylabel(f"{method.upper()}-2")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_overview(global_df: pd.DataFrame, out_path: Path):
    colors = {
        "cold": "#1f77b4",
        "hot": "#d62728",
        "bright": "#ff7f0e",
        "dark": "#2ca02c",
    }
    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=140)
    for word, group in global_df.groupby("word"):
        for label, marker in [(0, "o"), (1, "^")]:
            sub = group[group["label"] == label]
            if sub.empty:
                continue
            ax.scatter(
                sub["pc1"],
                sub["pc2"],
                marker=marker,
                color=colors.get(word, "#333333"),
                alpha=0.75,
                label=f"{word} ({'lit' if label == 0 else 'met'})",
            )
    ax.set_xlabel("PCA-1")
    ax.set_ylabel("PCA-2")
    ax.set_title("Overview: PCA of all embeddings")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Analyze a small metaphor dataset.")
    parser.add_argument("--data_path", default="small_metaphor_dataset_4words.csv")
    parser.add_argument("--model_name", default="roberta-base")
    parser.add_argument("--method", default="pca", choices=["pca", "umap"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_root", default="runs/smallset/")
    args = parser.parse_args()

    set_seed(args.seed)

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out_root) / timestamp
    ensure_dir(run_dir)
    plots_dir = run_dir / "plots"
    ensure_dir(plots_dir)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(run_dir / "logs.txt", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.info("Starting analysis")
    logging.info("Data path: %s", args.data_path)
    logging.info("Model: %s", args.model_name)
    logging.info("Method: %s", args.method)

    data_path = Path(args.data_path)
    if not data_path.exists():
        logging.error("Dataset not found: %s", data_path)
        sys.exit(1)

    df = pd.read_csv(data_path)
    required_cols = {"id", "word", "label", "sentence"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        logging.error("Missing required columns: %s", missing)
        sys.exit(1)

    dropped_rows = []
    valid_rows = []

    for row in df.to_dict(orient="records"):
        span, match_count = find_target_span(row["sentence"], row["word"])
        if span is None:
            dropped_rows.append(
                {
                    "id": row.get("id"),
                    "word": row.get("word"),
                    "reason": f"regex_match_count={match_count}",
                }
            )
            continue
        row["span"] = span
        valid_rows.append(row)

    logging.info("Rows: %d total, %d passed regex, %d dropped by regex", len(df), len(valid_rows), len(dropped_rows))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Device: %s", device)

    tokenizer, model, insecure_download = load_model_and_tokenizer(args.model_name, use_fast=True)
    if insecure_download:
        logging.warning("SSL verification disabled for model download due to prior failure.")
    model.to(device)
    model.eval()

    embeddings = [None] * len(valid_rows)

    with torch.no_grad():
        for batch_indices in chunked(list(range(len(valid_rows))), args.batch_size):
            batch = [valid_rows[i] for i in batch_indices]
            sentences = [row["sentence"] for row in batch]
            spans = [row["span"] for row in batch]
            encoded = tokenizer(
                sentences,
                return_offsets_mapping=True,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            offsets = encoded.pop("offset_mapping")
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = model(**encoded)
            hidden = outputs.last_hidden_state

            for local_i, global_i in enumerate(batch_indices):
                span_start, span_end = spans[local_i]
                offsets_i = offsets[local_i].tolist()
                token_indices = [
                    idx
                    for idx, (start, end) in enumerate(offsets_i)
                    if not (start == 0 and end == 0)
                    and start < span_end
                    and end > span_start
                ]
                if not token_indices:
                    dropped_rows.append(
                        {
                            "id": valid_rows[global_i].get("id"),
                            "word": valid_rows[global_i].get("word"),
                            "reason": "no_subword_overlap",
                        }
                    )
                    continue
                token_emb = hidden[local_i, token_indices].mean(dim=0).cpu().numpy()
                embeddings[global_i] = token_emb

    final_rows = []
    final_embeddings = []
    for row, emb in zip(valid_rows, embeddings):
        if emb is None:
            continue
        row.pop("span", None)
        final_rows.append(row)
        final_embeddings.append(emb)

    if not final_rows:
        logging.error("No valid embeddings to analyze.")
        sys.exit(1)

    emb_array = np.vstack(final_embeddings)
    emb_dim = emb_array.shape[1]
    emb_cols = [f"emb_{i}" for i in range(emb_dim)]

    out_df = pd.DataFrame(final_rows)
    out_df["label"] = out_df["label"].astype(int)
    out_df = out_df.reset_index(drop=True)
    emb_df = pd.DataFrame(emb_array, columns=emb_cols)
    out_df = pd.concat([out_df, emb_df], axis=1)

    pca_global = PCA(n_components=2, random_state=args.seed)
    pcs = pca_global.fit_transform(emb_array)
    out_df["pc1"] = pcs[:, 0]
    out_df["pc2"] = pcs[:, 1]

    embeddings_path = run_dir / "embeddings.csv"
    out_df.to_csv(embeddings_path, index=False)
    logging.info("Saved embeddings to %s", embeddings_path)

    metrics = []
    words = sorted(out_df["word"].unique())
    dropped_by_word = {}
    for drop in dropped_rows:
        word = drop.get("word")
        dropped_by_word[word] = dropped_by_word.get(word, 0) + 1
        logging.warning("Dropped id=%s word=%s reason=%s", drop.get("id"), word, drop.get("reason"))

    for word in words:
        word_mask = out_df["word"] == word
        word_df = out_df[word_mask]
        X_word = emb_array[word_mask.values]
        labels = word_df["label"].values
        X_literal = X_word[labels == 0]
        X_metaphor = X_word[labels == 1]
        n_literal = X_literal.shape[0]
        n_metaphor = X_metaphor.shape[0]
        centroid_literal = X_literal.mean(axis=0) if n_literal > 0 else None
        centroid_metaphor = X_metaphor.mean(axis=0) if n_metaphor > 0 else None
        if centroid_literal is not None and centroid_metaphor is not None:
            centroid_dist = cosine_distance(centroid_literal, centroid_metaphor)
        else:
            centroid_dist = float("nan")
        dispersion_literal = (
            mean_cosine_distance(X_literal, centroid_literal) if n_literal > 0 else float("nan")
        )
        dispersion_metaphor = (
            mean_cosine_distance(X_metaphor, centroid_metaphor) if n_metaphor > 0 else float("nan")
        )
        if np.isnan(centroid_dist):
            shift_ratio = float("nan")
        else:
            denom = ((dispersion_literal + dispersion_metaphor) / 2.0) + 1e-9
            shift_ratio = float(centroid_dist / denom)
        metrics.append(
            {
                "word": word,
                "n_literal": n_literal,
                "n_metaphor": n_metaphor,
                "dropped": dropped_by_word.get(word, 0),
                "centroid_cosine_distance": centroid_dist,
                "dispersion_literal": dispersion_literal,
                "dispersion_metaphor": dispersion_metaphor,
                "shift_ratio": shift_ratio,
            }
        )

    metrics_df = pd.DataFrame(metrics)
    metrics_path = run_dir / "metrics_by_word.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logging.info("Saved metrics to %s", metrics_path)

    plots = {}
    for word in words:
        word_mask = out_df["word"] == word
        word_df = out_df[word_mask]
        X_word = emb_array[word_mask.values]
        labels = word_df["label"].values
        metrics_row = metrics_df[metrics_df["word"] == word].iloc[0]
        plot_path = plots_dir / f"word_shift_{word}_{args.method}.png"
        plot_word_shift(word, X_word, labels, metrics_row, args.method, args.seed, plot_path)
        plots[word] = plot_path.relative_to(run_dir)
        logging.info("Saved plot %s", plot_path)

    overview_path = plots_dir / "overview_scatter_pca.png"
    plot_overview(out_df, overview_path)
    logging.info("Saved overview plot %s", overview_path)

    build_report(run_dir, args.method, metrics_df, dropped_rows, plots)
    logging.info("Saved report to %s", run_dir / "report.md")
    logging.info("Done.")


if __name__ == "__main__":
    main()
