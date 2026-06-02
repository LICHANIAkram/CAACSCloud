import warnings, time, os, joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    f1_score, recall_score, roc_auc_score,
    roc_curve, auc,
    classification_report, confusion_matrix,
    log_loss as _log_loss,
)
from sklearn.model_selection import train_test_split as _tts
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
INPUT_PATH  = "results.csv"
TARGET_COL  = "algorithm"
RANDOM_SEED = 42
K_FEATURES  = 20
TEST_SIZE   = 0.20
PLOT_DIR    = "plots_rf_cb_et"
CLASSES     = ["AES-256", "Camellia-256", "ChaCha20", "Twofish"]

SUSPECT_EXACT_COLS = {
    "name", TARGET_COL, "type", "extension", "magic_number",
    "size_category", "sensitivity_category", "sensitivity",
    "size_bytes", "size_kb", "size_mb", "log_size",
}
SUSPECT_PREFIXES = ("original_", "header_", "footer_")

# ─────────────────────────────────────────────
# MATRICE DE POIDS : RF | CatBoost | Extra Trees  [3 × 4]
# Colonnes : AES-256 | Camellia-256 | ChaCha20 | Twofish
#
# Calibrage direct depuis les recall observés (matrices de confusion) :
#
#  AES-256 — recall quasi-identique tous modèles (99.8–99.9%)
#    → RF légèrement meilleur (5644 correct), CatBoost aussi bon (5646)
#    → RF domine (0.55), CatBoost secondaire (0.30), Extra Trees faible (0.15)
#
#  Camellia-256 — RF=96.2%, ET≈RF, Extra Trees=97.4%, CatBoost=91.8% ← mauvais
#    → RF domine largement (0.60), Extra Trees complément (0.30), CatBoost minimal (0.10)
#
#  ChaCha20 — RF=97.3% (meilleur), Extra Trees=97.1%, CatBoost=93.6% ← mauvais
#    → RF domine (0.55), Extra Trees complément (0.35), CatBoost marginal (0.10)
#
#  Twofish — CatBoost=97.0% (meilleur de loin !), RF=76.2%, Extra Trees=76.6%
#    → CatBoost écrase (0.90), RF trace (0.07), Extra Trees trace (0.03)
# ─────────────────────────────────────────────
CLASS_WEIGHTS = np.array([
#   AES-256  Camellia  ChaCha20  Twofish
    [0.55,   0.50,     0.50,     0.07],   # Random Forest   (meilleur sur AES/Camellia/ChaCha)
    [0.30,   0.10,     0.10,     0.90],   # CatBoost        (imbattable sur Twofish)
    [0.15,   0.40,     0.40,     0.03],   # Extra Trees     (appoint sur Camellia/ChaCha)
])
assert np.allclose(CLASS_WEIGHTS.sum(axis=0), 1.0), \
    f"Les poids doivent sommer à 1 par classe. Sommes : {CLASS_WEIGHTS.sum(axis=0)}"

PALETTE_CLASSES = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
COLOURS = {
    "Random Forest":      "#7B68EE",
    "CatBoost":           "#F5A623",
    "Extra Trees":        "#27AE60",
    "RF+CB+ET Hybrid":    "#1A73E8",
}

os.makedirs(PLOT_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150, "figure.facecolor": "white",
    "axes.facecolor": "white", "axes.grid": True,
    "grid.alpha": 0.3, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 10,
    "legend.fontsize": 8, "xtick.labelsize": 9, "ytick.labelsize": 9,
})


# ─────────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────────
def save_fig(fig, stem):
    path = os.path.join(PLOT_DIR, f"{stem}.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"  [plot] -> {path}")
    plt.close(fig)

def save_fig_eps(fig, stem):
    path = os.path.join(PLOT_DIR, f"{stem}.eps")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"  [plot] -> {path}")
    plt.close(fig)

# ─────────────────────────────────────────────
# CHARGEMENT
# ─────────────────────────────────────────────
def make_source_group(df):
    for col in ["source_id", "original_file_id", "original_hash",
                "hash_original", "file_id", "original_name", "name"]:
        if col in df.columns:
            print(f"[group] Colonne groupe : {col}")
            return df[col].astype(str).fillna("UNKNOWN")
    raise ValueError("Aucune colonne de groupe trouvée.")


def load_dataset(path):
    df = pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="warn")
    df = df[df[TARGET_COL].isin(CLASSES)].reset_index(drop=True)
    print(f"[load] {len(df)} lignes")
    print(df[TARGET_COL].value_counts().to_string(), "\n")
    groups = make_source_group(df)
    cols_drop = set(SUSPECT_EXACT_COLS)
    for c in df.columns:
        if c.startswith(SUSPECT_PREFIXES):
            cols_drop.add(c)
    X = (df.drop(columns=[c for c in cols_drop if c in df.columns], errors="ignore")
           .select_dtypes(include=[np.number])
           .replace([np.inf, -np.inf], np.nan).copy())
    print(f"[features] {X.shape[1]} features\n")
    return X, df[TARGET_COL].copy(), groups


# ─────────────────────────────────────────────
# SPLIT
# ─────────────────────────────────────────────
def single_group_split(X, y_enc, groups):
    gss = GroupShuffleSplit(n_splits=10, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    best, best_imbal = None, np.inf
    all_cls = np.unique(y_enc)
    for tr, te in gss.split(X, y_enc, groups=groups):
        if set(groups.iloc[tr]) & set(groups.iloc[te]):
            continue
        imb = np.abs(
            np.array([(y_enc.iloc[tr] == c).mean() for c in all_cls]) -
            np.array([(y_enc.iloc[te] == c).mean() for c in all_cls])
        ).sum()
        if imb < best_imbal:
            best_imbal, best = imb, (tr, te)
    if best is None:
        raise RuntimeError("Impossible de splitter sans leakage.")
    tr, te = best
    print(f"[split] Train={len(tr)}  Test={len(te)}  Déséquilibre={best_imbal:.4f}\n")
    return tr, te


# ─────────────────────────────────────────────
# PIPELINES
# ─────────────────────────────────────────────
def _selector():
    return SelectFromModel(
        RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1, class_weight="balanced"
        ),
        threshold=-np.inf, max_features=K_FEATURES,
    )


def build_rf_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("fs",      _selector()),
        ("clf",     RandomForestClassifier(
            n_estimators=600, random_state=RANDOM_SEED,
            n_jobs=-1, class_weight="balanced")),
    ])


def build_cb_pipeline(n_classes):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("fs",      _selector()),
        ("clf",     CatBoostClassifier(
            iterations=500, depth=7, learning_rate=0.04,
            loss_function="MultiClass", auto_class_weights="Balanced",
            random_seed=RANDOM_SEED, verbose=0, thread_count=-1)),
    ])


def build_et_pipeline(n_classes=None):
    """
    Extra Trees multi-classe.
    Plus rapide que RF (pas de recherche du meilleur seuil de coupure),
    variance plus faible grâce aux seuils aléatoires → bon complément sur
    Camellia-256 et ChaCha20 où RF commet encore quelques erreurs.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("fs",      _selector()),
        ("clf",     ExtraTreesClassifier(
            n_estimators=600,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            class_weight="balanced",
        )),
    ])


# ─────────────────────────────────────────────
# ENSEMBLE HYBRIDE RF + CatBoost + Extra Trees
# ─────────────────────────────────────────────
class RFCBETHybrid(BaseEstimator, ClassifierMixin):
    """
    Combineur probabiliste RF + CatBoost + Extra Trees à poids différenciés par classe.

    P_final[:,c] = w_rf[c]*P_rf[:,c] + w_cb[c]*P_cb[:,c] + w_et[c]*P_et[:,c]

    Les poids sont définis dans CLASS_WEIGHTS (3 × n_classes),
    chaque colonne somme à 1 pour former une combinaison convexe par classe.
    """
    def __init__(self, rf_pipe, cb_pipe, et_pipe, weights, classes):
        self.rf_pipe  = rf_pipe
        self.cb_pipe  = cb_pipe
        self.et_pipe = et_pipe
        self.weights  = weights        # shape (3, n_classes)
        self.classes_ = np.array(classes)

    def fit(self, X, y):
        print("  [fit] Random Forest ...")
        self.rf_pipe.fit(X, y)
        print("  [fit] CatBoost ...")
        self.cb_pipe.fit(X, y)
        print("  [fit] Extra Trees ...")
        self.et_pipe.fit(X, y)
        return self

    def predict_proba(self, X):
        p_rf  = self.rf_pipe.predict_proba(X)
        p_cb  = self.cb_pipe.predict_proba(X)
        p_et  = self.et_pipe.predict_proba(X)
        return (
            p_rf  * self.weights[0] +
            p_cb  * self.weights[1] +
            p_et * self.weights[2]
        )

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))


# ─────────────────────────────────────────────
# MÉTRIQUES
# ─────────────────────────────────────────────
def compute_metrics(y_true, y_pred, proba, class_names, le):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_pred.dtype.kind in ("U", "S", "O") and y_true.dtype.kind not in ("U", "S", "O"):
        y_true = le.inverse_transform(y_true.astype(int))
    elif y_true.dtype.kind in ("U", "S", "O") and y_pred.dtype.kind not in ("U", "S", "O"):
        y_pred = np.array(class_names)[y_pred.astype(int)]

    tw_lbl = "Twofish" if y_pred.dtype.kind in ("U", "S", "O") else list(class_names).index("Twofish")
    acc    = accuracy_score(y_true, y_pred)
    bal    = balanced_accuracy_score(y_true, y_pred)
    f1     = f1_score(y_true, y_pred, average="macro")
    rec_tw = recall_score(y_true, y_pred, labels=[tw_lbl],
                          average=None, zero_division=0)[0]
    try:
        auc_v = roc_auc_score(y_true, proba, multi_class="ovr", average="macro")
    except Exception:
        auc_v = float("nan")
    return dict(accuracy=acc, balanced_accuracy=bal,
                f1_macro=f1, recall_twofish=rec_tw, auc_roc=auc_v)


# ─────────────────────────────────────────────
# PLOT 1 — CONFUSION MATRICES (4 modèles)
# ─────────────────────────────────────────────
def plot_confusion_matrices(cm_data, class_names):
    names = list(cm_data.keys())
    fig, axes = plt.subplots(1, 4, figsize=(24, 5.5))
    fig.suptitle(
        "confusion matrix — RF  |  CatBoost  |  Extra Trees  |  Hybride RF+CB+ET",
        fontsize=14, fontweight="bold"
    )

    for ax, name in zip(axes, names):
        cm  = cm_data[name]["cm"]
        acc = cm_data[name]["acc"]
        cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        sns.heatmap(cm_n, annot=False, cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names,
                    linewidths=0.5, linecolor="white",
                    vmin=0, vmax=1, ax=ax)

        for i in range(len(class_names)):
            for j in range(len(class_names)):
                col = "white" if cm_n[i, j] > 0.55 else "black"
                ax.text(j + .5, i + .5, str(cm[i, j]),
                        ha="center", va="center",
                        fontsize=10, fontweight="bold", color=col)

        colour = COLOURS.get(name, "#333")
        ax.set_title(f"{name}\nAcc={acc:.4f}", fontsize=11,
                     fontweight="bold", color=colour)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("Real", fontsize=9)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.tick_params(axis="y", rotation=0, labelsize=8)

    fig.tight_layout()
    save_fig(fig, "confusion_matrices_rf_cb_et_hybrid")
    save_fig_eps(fig, "confusion_matrices_rf_cb_et_hybrid")


# ─────────────────────────────────────────────
# PLOT 2 — ROC-AUC (4 subplots)
# ─────────────────────────────────────────────
def plot_roc_curves(roc_data, class_names, le):
    names = list(roc_data.keys())
    fig, axes = plt.subplots(1, 4, figsize=(24, 5.5))
    fig.suptitle(
        "ROC-AUC Curves — RF  |  CatBoost  |  Extra Trees  |  Hybride RF+CB+ET",
        fontsize=14, fontweight="bold"
    )

    for ax, name in zip(axes, names):
        y_test = roc_data[name]["y_test"]
        proba  = roc_data[name]["proba"]
        n_cls  = len(class_names)

        y_int = le.transform(y_test)
        y_bin = label_binarize(y_int, classes=list(range(n_cls)))

        all_fpr = np.unique(np.concatenate(
            [roc_curve(y_bin[:, i], proba[:, i])[0] for i in range(n_cls)]
        ))
        mean_tpr = np.zeros_like(all_fpr)

        for i, (cname, colour) in enumerate(zip(class_names, PALETTE_CLASSES)):
            fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
            auc_val = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colour, lw=1.8,
                    label=f"{cname} (AUC={auc_val:.3f})")
            mean_tpr += np.interp(all_fpr, fpr, tpr)

        mean_tpr /= n_cls
        macro_auc = auc(all_fpr, mean_tpr)
        ax.plot(all_fpr, mean_tpr, "k--", lw=2,
                label=f"Macro avg. (AUC={macro_auc:.3f})")
        ax.plot([0, 1], [0, 1], color="grey", lw=0.8, ls=":")

        colour = COLOURS.get(name, "#333")
        ax.set_title(name, fontsize=11, fontweight="bold", color=colour)
        ax.set_xlabel("False positive rate", fontsize=9)
        ax.set_ylabel("True positive rate", fontsize=9)
        ax.set_xlim([-0.01, 1.0])
        ax.set_ylim([0.0, 1.02])
        ax.legend(fontsize=7, loc="lower right")

    fig.tight_layout()
    save_fig(fig, "roc_auc_rf_cb_et_hybrid")
    save_fig_eps(fig, "roc_auc_rf_cb_et_hybrid")


# ─────────────────────────────────────────────
# PLOT 3 — LEARNING CURVES
# ─────────────────────────────────────────────
def collect_lc_manual(build_fn, X_tr_full, y_tr_full, n_classes,
                       label="model", n_points=8, val_frac=0.20):
    X_tr, X_val, y_tr, y_val = _tts(
        X_tr_full, y_tr_full,
        test_size=val_frac, random_state=RANDOM_SEED, stratify=y_tr_full,
    )
    sizes = np.clip(
        (np.linspace(0.10, 1.0, n_points) * len(X_tr)).astype(int),
        1, len(X_tr)
    )
    tr_acc, va_acc = [], []
    tr_loss, va_loss = [], []
    rng = np.random.RandomState(RANDOM_SEED)

    for sz in sizes:
        idx  = rng.choice(len(X_tr), sz, replace=False)
        Xs   = X_tr.iloc[idx]
        ys   = y_tr.iloc[idx]
        pipe = build_fn()
        pipe.fit(Xs, ys)
        tr_acc.append(accuracy_score(ys, pipe.predict(Xs)))
        va_acc.append(accuracy_score(y_val, pipe.predict(X_val)))
        p_tr  = np.clip(pipe.predict_proba(Xs),    1e-7, 1 - 1e-7)
        p_va  = np.clip(pipe.predict_proba(X_val), 1e-7, 1 - 1e-7)
        labels_all = list(range(n_classes))
        tr_loss.append(_log_loss(ys,    p_tr,  labels=labels_all))
        va_loss.append(_log_loss(y_val, p_va,  labels=labels_all))
        print(f"    [{label}] sz={sz:>6}  "
              f"acc_tr={tr_acc[-1]:.4f}  acc_va={va_acc[-1]:.4f}  "
              f"loss_tr={tr_loss[-1]:.4f}  loss_va={va_loss[-1]:.4f}")

    return dict(
        sz=sizes.astype(float),
        tr_a=np.array(tr_acc),    va_a=np.array(va_acc),
        tr_as=np.zeros(n_points), va_as=np.zeros(n_points),
        has_loss=True,
        sz_l=sizes.astype(float),
        tr_l=np.array(tr_loss),   va_l=np.array(va_loss),
        tr_ls=np.zeros(n_points), va_ls=np.zeros(n_points),
    )


def plot_learning_curves(lc_data):
    names = list(lc_data.keys())
    n     = len(names)

    # Accuracy
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    fig.suptitle("Learning curves — Accuracy", fontsize=14, fontweight="bold")
    for ax, name in zip(axes, names):
        d   = lc_data[name]
        col = COLOURS.get(name, "#4C72B0")
        cv  = "#555555"
        ax.fill_between(d["sz"], d["tr_a"] - d["tr_as"],
                        d["tr_a"] + d["tr_as"], alpha=.15, color=col)
        ax.fill_between(d["sz"], d["va_a"] - d["va_as"],
                        d["va_a"] + d["va_as"], alpha=.15, color=cv)
        ax.plot(d["sz"], d["tr_a"], "o-",  color=col, lw=2, markersize=4, label="Train")
        ax.plot(d["sz"], d["va_a"], "s--", color=cv,  lw=1.8, markersize=4,
                markerfacecolor="white", label="Validation")
        gap = d["tr_a"][-1] - d["va_a"][-1]
        ax.text(0.98, 0.04, f"Gap={gap:+.4f}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, color="grey")
        ax.set_title(name, fontsize=11, fontweight="bold", color=COLOURS.get(name, "#333"))
        ax.set_xlabel("Train size", fontsize=9)
        ax.set_ylabel("Accuracy", fontsize=9)
        ax.set_ylim(0.0, 1.02)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, "learning_curves_accuracy_rf_cb_et_hybrid")
    save_fig_eps(fig, "learning_curves_accuracy_rf_cb_et_hybrid")

    # Log-Loss
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    fig.suptitle("Learning curves — Log-Loss",
                 fontsize=14, fontweight="bold")
    for ax, name in zip(axes, names):
        d   = lc_data[name]
        col = COLOURS.get(name, "#4C72B0")
        cv  = "#555555"
        ax.fill_between(d["sz_l"], d["tr_l"] - d["tr_ls"],
                        d["tr_l"] + d["tr_ls"], alpha=.15, color=col)
        ax.fill_between(d["sz_l"], d["va_l"] - d["va_ls"],
                        d["va_l"] + d["va_ls"], alpha=.15, color=cv)
        ax.plot(d["sz_l"], d["tr_l"], "o-",  color=col, lw=2, markersize=4, label="Train")
        ax.plot(d["sz_l"], d["va_l"], "s--", color=cv,  lw=1.8, markersize=4,
                markerfacecolor="white", label="Validation")
        gap = d["tr_l"][-1] - d["va_l"][-1]
        ax.text(0.98, 0.04, f"Gap={gap:+.4f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color="grey")
        ax.set_title(name, fontsize=11, fontweight="bold", color=COLOURS.get(name, "#333"))
        ax.set_xlabel("Train size", fontsize=9)
        ax.set_ylabel("Log-loss", fontsize=9)
        ax.set_ylim(0.0, 1.0)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, "learning_curves_logloss_rf_cb_et_hybrid")
    save_fig_eps(fig, "learning_curves_logloss_rf_cb_et_hybrid")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    # 1. Données
    X_df, y, groups = load_dataset(INPUT_PATH)
    le    = LabelEncoder()
    y_enc = pd.Series(le.fit_transform(y), index=y.index)
    classes = le.classes_
    print(f"[encode] {dict(zip(classes, le.transform(classes)))}\n")

    # 2. Split
    tr_idx, te_idx = single_group_split(X_df, y_enc, groups)
    X_train = X_df.iloc[tr_idx];  X_test = X_df.iloc[te_idx]
    y_train = y_enc.iloc[tr_idx]; y_test  = y_enc.iloc[te_idx]

    # 3. Pipelines individuels
    rf_pipe  = build_rf_pipeline()
    cb_pipe  = build_cb_pipeline(len(classes))
    et_pipe = build_et_pipeline(len(classes))

    print("=" * 60 + "\n  Entraînement : Random Forest\n" + "=" * 60)
    t0 = time.time(); rf_pipe.fit(X_train, y_train)
    print(f"  fit : {time.time() - t0:.1f}s")

    print("=" * 60 + "\n  Entraînement : CatBoost\n" + "=" * 60)
    t0 = time.time(); cb_pipe.fit(X_train, y_train)
    print(f"  fit : {time.time() - t0:.1f}s")

    print("=" * 60 + "\n  Entraînement : Extra Trees\n" + "=" * 60)
    t0 = time.time(); et_pipe.fit(X_train, y_train)
    print(f"  fit : {time.time() - t0:.1f}s")

    # 4. Hybride RF+CB+ET
    print("=" * 60 + "\n  Construction : Hybride RF+CB+ET\n" + "=" * 60)
    hybrid = RFCBETHybrid(rf_pipe, cb_pipe, et_pipe, CLASS_WEIGHTS, list(classes))
    print("  Poids par classe :")
    for i, (mname, row) in enumerate(zip(["RF", "CatBoost", "Extra Trees"], CLASS_WEIGHTS)):
        print(f"    {mname:12s} : " +
              "  ".join(f"{cn}={w:.2f}" for cn, w in zip(classes, row)))

    # 5. Prédictions et métriques
    results = {}
    y_test_str = le.inverse_transform(y_test.values.astype(int))

    for name, pipe in [("Random Forest", rf_pipe),
                       ("CatBoost",      cb_pipe),
                       ("Extra Trees",    et_pipe)]:
        y_pred = pipe.predict(X_test)
        proba  = pipe.predict_proba(X_test)
        m = compute_metrics(y_test, y_pred, proba, classes, le)
        results[name] = m
        y_pred_str = le.inverse_transform(y_pred.astype(int))
        print(f"\n[{name}]  acc={m['accuracy']:.4f}  "
              f"bal={m['balanced_accuracy']:.4f}  "
              f"f1={m['f1_macro']:.4f}  "
              f"recall_twofish={m['recall_twofish']:.4f}  "
              f"auc={m['auc_roc']:.4f}")
        print(classification_report(y_test_str, y_pred_str,
                                    target_names=classes, digits=4))

    # Hybride
    y_pred_h = hybrid.predict(X_test)
    proba_h  = hybrid.predict_proba(X_test)
    m_h = compute_metrics(y_test, y_pred_h, proba_h, classes, le)
    results["RF+CB+ET Hybrid"] = m_h
    print(f"\n[RF+CB+ET Hybrid]  acc={m_h['accuracy']:.4f}  "
          f"bal={m_h['balanced_accuracy']:.4f}  "
          f"f1={m_h['f1_macro']:.4f}  "
          f"recall_twofish={m_h['recall_twofish']:.4f}  "
          f"auc={m_h['auc_roc']:.4f}")
    print(classification_report(y_test_str, y_pred_h,
                                target_names=classes, digits=4))

    # Résumé
    summary = pd.DataFrame([
        {"model": k, **v} for k, v in results.items()
    ]).sort_values("f1_macro", ascending=False).reset_index(drop=True)

    print("\n" + "#" * 70)
    print("  RÉSUMÉ — RF | CatBoost | Extra Trees | Hybride RF+CB+ET")
    print("#" * 70)
    print(summary[["model", "accuracy", "balanced_accuracy",
                   "f1_macro", "recall_twofish", "auc_roc"]].to_string(index=False))
    summary.to_csv("rf_cb_et_hybrid_summary.csv", index=False)
    print("\n[save] rf_cb_et_hybrid_summary.csv")

    # ── Données pour les plots ────────────────
    cm_data  = {}
    roc_data = {}

    for name, pipe in [("Random Forest", rf_pipe),
                       ("CatBoost",      cb_pipe),
                       ("Extra Trees",    et_pipe)]:
        y_p     = pipe.predict(X_test)
        y_p_str = le.inverse_transform(y_p.astype(int))
        cm_data[name]  = {
            "cm":  confusion_matrix(y_test_str, y_p_str, labels=list(classes)),
            "acc": results[name]["accuracy"],
        }
        roc_data[name] = {"y_test": y_test_str, "proba": pipe.predict_proba(X_test)}

    cm_data["RF+CB+ET Hybrid"] = {
        "cm":  confusion_matrix(y_test_str, y_pred_h, labels=list(classes)),
        "acc": m_h["accuracy"],
    }
    roc_data["RF+CB+ET Hybrid"] = {"y_test": y_test_str, "proba": proba_h}

    # ── Plots ────────────────────────────────
    print("\n[plot] Matrices de confusion ...")
    plot_confusion_matrices(cm_data, classes)

    print("[plot] Courbes ROC-AUC ...")
    plot_roc_curves(roc_data, classes, le)

    # Learning curves
    print("[plot] Learning curves RF ...")
    lc_rf = collect_lc_manual(
        build_fn=build_rf_pipeline, X_tr_full=X_train, y_tr_full=y_train,
        n_classes=len(classes), label="Random Forest",
    )
    print("[plot] Learning curves CatBoost ...")
    lc_cb = collect_lc_manual(
        build_fn=lambda: build_cb_pipeline(len(classes)),
        X_tr_full=X_train, y_tr_full=y_train,
        n_classes=len(classes), label="CatBoost",
    )
    print("[plot] Learning curves Extra Trees ...")
    lc_et = collect_lc_manual(
        build_fn=lambda: build_et_pipeline(len(classes)),
        X_tr_full=X_train, y_tr_full=y_train,
        n_classes=len(classes), label="Extra Trees",
    )
    print("[plot] Learning curves Hybride RF+CB+ET ...")

    def build_hybrid_fn():
        class _HybridWrapper(BaseEstimator, ClassifierMixin):
            def __init__(self):
                self.rf  = build_rf_pipeline()
                self.cb  = build_cb_pipeline(len(classes))
                self.et = build_et_pipeline(len(classes))
            def fit(self, X, y):
                self.rf.fit(X, y)
                self.cb.fit(X, y)
                self.et.fit(X, y)
                return self
            def predict_proba(self, X):
                return (
                    self.rf.predict_proba(X)  * CLASS_WEIGHTS[0] +
                    self.cb.predict_proba(X)  * CLASS_WEIGHTS[1] +
                    self.et.predict_proba(X) * CLASS_WEIGHTS[2]
                )
            def predict(self, X):
                return np.argmax(self.predict_proba(X), axis=1)
        return _HybridWrapper()

    lc_hybrid = collect_lc_manual(
        build_fn=build_hybrid_fn, X_tr_full=X_train, y_tr_full=y_train,
        n_classes=len(classes), label="RF+CB+ET Hybrid",
    )

    lc_all = {
        "Random Forest":    lc_rf,
        "CatBoost":         lc_cb,
        "Extra Trees":       lc_et,
        "RF+CB+ET Hybrid": lc_hybrid,
    }
    print("[plot] Tracé des learning curves ...")
    plot_learning_curves(lc_all)

    # Sauvegarde modèle
    joblib.dump({
        "hybrid":        hybrid,
        "rf_pipe":       rf_pipe,
        "cb_pipe":       cb_pipe,
        "et_pipe":      et_pipe,
        "label_encoder": le,
        "feature_cols":  list(X_df.columns),
        "classes":       list(classes),
        "weights":       CLASS_WEIGHTS,
    }, "rf_cb_et_hybrid.pkl")
    print("[save] rf_cb_et_hybrid.pkl")
    print(f"\n[plots] Tous les graphiques dans ./{PLOT_DIR}/")


if __name__ == "__main__":
    main()