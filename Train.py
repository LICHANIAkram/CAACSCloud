# ==============================================================
# Context-Aware Cryptographic Selection — Model Training v8 RAW FEATURES
# Optuna sur tous les modèles, sans data augmentation et sans feature engineering
#
# Changements principaux vs v6 :
#   1. Correction de l'ordre du pipeline : features puis scaling puis modèles
#   2. Aucune data augmentation : aucune génération artificielle d'échantillons
#   3. Gestion du déséquilibre par class_weight et sample_weight
#   4. Ajout CatBoost + XGBoost + ExtraTrees + RandomForest
#   5. Ensemble par probabilités pondérées
#   6. Recherche automatique des meilleurs poids d'ensemble
#   7. Optimisation optionnelle des seuils par classe
#   8. Sauvegarde des modèles, métriques et figures
#
# Installation si nécessaire :
#   pip install pandas numpy scikit-learn xgboost optuna catboost joblib matplotlib
# ==============================================================

import os
import json
import warnings
from itertools import cycle

import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve
from sklearn.preprocessing import LabelEncoder, RobustScaler, label_binarize
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    log_loss,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

warnings.filterwarnings("ignore")


# ==============================================================
# CONFIGURATION
# ==============================================================

DATA_PATH = "Data_Set.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.20

TARGET_SCORE = 0.95
USE_OPTUNA = True

# Réglages de recherche Optuna. Augmentez si vous avez plus de temps de calcul.
OPTUNA_TRIALS_RF = 30
OPTUNA_TRIALS_ET = 30
OPTUNA_TRIALS_XGB = 50
OPTUNA_TRIALS_CAT = 50

# Version sans augmentation : on conserve exactement les échantillons originaux.
USE_DATA_AUGMENTATION = False

OUTPUT_DIR = "outputs_v8_raw_features_no_engineering"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================================================
# UTILITAIRES GENERAUX
# ==============================================================

def out_path(filename):
    return os.path.join(OUTPUT_DIR, filename)


def print_header(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def safe_bincount(y):
    counts = np.bincount(np.asarray(y, dtype=int))
    return counts.tolist()


# ==============================================================
# 1. CHARGEMENT DU DATASET
# ==============================================================

print_header("CHARGEMENT DU DATASET")

df = pd.read_csv(
    DATA_PATH,
    sep=",",
    header=None,
    encoding="utf-8-sig",
    skiprows=1,
    names=["name", "size", "type", "extension", "sensitivity", "algorithm"],
    dtype=str,
)

df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
df["size"] = pd.to_numeric(df["size"], errors="coerce")
df["sensitivity"] = pd.to_numeric(df["sensitivity"], errors="coerce")

df.dropna(subset=["size", "sensitivity", "algorithm", "type", "extension"], inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"Shape : {df.shape}")
print("\nDistribution des classes :")
print(df["algorithm"].value_counts())


# ==============================================================
# 2. ENCODAGE
# ==============================================================

print_header("ENCODAGE")

df.drop(columns=["name"], inplace=True, errors="ignore")

label_encoders = {}

for col in ["type", "extension"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

target_encoder = LabelEncoder()
df["algorithm"] = target_encoder.fit_transform(df["algorithm"].astype(str))

CLASS_NAMES = list(target_encoder.classes_)
n_classes = len(CLASS_NAMES)

print(f"Classes : {CLASS_NAMES}")
print(f"Nombre de classes : {n_classes}")

BASE_COLS = ["type", "extension", "size", "sensitivity"]
X_base = df[BASE_COLS].to_numpy()
y = df["algorithm"].to_numpy()


# ==============================================================
# 3. SPLIT STRATIFIE
# ==============================================================

print_header("SPLIT STRATIFIE")

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_base,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

X_tr_raw, X_val_raw, y_tr, y_val = train_test_split(
    X_train_raw,
    y_train,
    test_size=VAL_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_train,
)

print(f"Train : {X_tr_raw.shape}")
print(f"Val   : {X_val_raw.shape}")
print(f"Test  : {X_test_raw.shape}")


# ==============================================================
# 4. SANS FEATURE ENGINEERING
# ==============================================================

print_header("SANS FEATURE ENGINEERING — VARIABLES BRUTES")

# Dans cette version, on n'ajoute aucune variable artificielle.
# Le modèle reçoit uniquement les colonnes originales :
#   1. type
#   2. extension
#   3. size
#   4. sensitivity
#
# Après encodage de type/extension, X_base contient déjà ces 4 variables.
# Le nombre de colonnes reste donc 4 avant et après cette étape.

FEATURE_COLS = BASE_COLS.copy()

X_tr_feat = X_tr_raw.astype(float)
X_val_feat = X_val_raw.astype(float)
X_test_feat = X_test_raw.astype(float)

# Gardé pour compatibilité avec la sauvegarde et les scripts externes.
train_stats = {
    "mode": "raw_features_no_engineering",
    "feature_cols": FEATURE_COLS,
    "n_features": len(FEATURE_COLS),
}

print(f"Variables utilisées : {FEATURE_COLS}")
print(f"Features utilisées : {X_tr_feat.shape[1]}")

# ==============================================================
# 5. SCALING
# ==============================================================

print_header("SCALING")

scaler = RobustScaler()
X_tr_s = scaler.fit_transform(X_tr_feat)
X_val_s = scaler.transform(X_val_feat)
X_test_s = scaler.transform(X_test_feat)

y_tr_res = np.asarray(y_tr)
y_val = np.asarray(y_val)
y_test = np.asarray(y_test)

print(f"Train scaled : {X_tr_s.shape}")
print(f"Val scaled   : {X_val_s.shape}")
print(f"Test scaled  : {X_test_s.shape}")


# ==============================================================
# 6. POIDS DE CLASSES
# ==============================================================

print_header("POIDS DE CLASSES")

classes_tr = np.unique(y_tr_res)
cw = compute_class_weight("balanced", classes=classes_tr, y=y_tr_res)
cw_dict = {int(c): float(w) for c, w in zip(classes_tr, cw)}
sw_tr = np.array([cw_dict[int(c)] for c in y_tr_res])

for c, w in cw_dict.items():
    print(f"{CLASS_NAMES[c]:20s}: {w:.4f}")


# ==============================================================
# 7. SANS DATA AUGMENTATION
# ==============================================================

print_header("SANS DATA AUGMENTATION — POIDS DE CLASSES UNIQUEMENT")

# Dans cette version, on ne modifie pas la distribution réelle du dataset.
# Aucun échantillon synthétique n'est créé.
# Les modèles utilisent seulement :
#   - class_weight pour RandomForest / ExtraTrees
#   - sample_weight pour XGBoost / CatBoost
#   - calibration + ensemble pondéré pour stabiliser les probabilités

X_tr_bal = X_tr_s
y_tr_bal = y_tr_res
sample_weights_bal = sw_tr
balancing_method = "No augmentation - class_weight/sample_weight only"

print(f"Méthode utilisée : {balancing_method}")
print("Distribution train conservée :", safe_bincount(y_tr_res))


# ==============================================================
# 8. EVALUATION ET FIGURES
# ==============================================================

def evaluate_model(model, X_t, y_t, name):
    y_pred = np.asarray(model.predict(X_t)).flatten()

    acc = accuracy_score(y_t, y_pred)
    prec = precision_score(y_t, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_t, y_pred, average="macro", zero_division=0)
    f1m = f1_score(y_t, y_pred, average="macro", zero_division=0)
    f1w = f1_score(y_t, y_pred, average="weighted", zero_division=0)

    print_header(f"RESULTATS — {name}")
    print(f"Accuracy          : {acc:.4f}")
    print(f"Precision macro   : {prec:.4f}")
    print(f"Recall macro      : {rec:.4f}")
    print(f"F1 macro          : {f1m:.4f}")
    print(f"F1 weighted       : {f1w:.4f}")
    print()
    print(classification_report(y_t, y_pred, target_names=CLASS_NAMES, zero_division=0))

    return y_pred, {
        "accuracy": float(acc),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_macro": float(f1m),
        "f1_weighted": float(f1w),
    }


def evaluate_predictions(y_t, y_pred, name):
    acc = accuracy_score(y_t, y_pred)
    prec = precision_score(y_t, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_t, y_pred, average="macro", zero_division=0)
    f1m = f1_score(y_t, y_pred, average="macro", zero_division=0)
    f1w = f1_score(y_t, y_pred, average="weighted", zero_division=0)

    print_header(f"RESULTATS — {name}")
    print(f"Accuracy          : {acc:.4f}")
    print(f"Precision macro   : {prec:.4f}")
    print(f"Recall macro      : {rec:.4f}")
    print(f"F1 macro          : {f1m:.4f}")
    print(f"F1 weighted       : {f1w:.4f}")
    print()
    print(classification_report(y_t, y_pred, target_names=CLASS_NAMES, zero_division=0))

    return {
        "accuracy": float(acc),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_macro": float(f1m),
        "f1_weighted": float(f1w),
    }


def save_confusion_matrix(y_t, y_pred, name, filename_base, cmap="Blues"):
    cm = confusion_matrix(y_t, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(cmap=cmap, ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title(f"Confusion Matrix — {name}\nAccuracy={accuracy_score(y_t, y_pred):.3f}")
    plt.tight_layout()
    plt.savefig(out_path(f"{filename_base}.png"), dpi=160, bbox_inches="tight")
    plt.savefig(out_path(f"{filename_base}.eps"), format="eps", bbox_inches="tight")
    plt.close()

    print(f"Matrice sauvegardée : {out_path(filename_base + '.png')}")


def analyse_errors(y_t, y_pred, name):
    cm = confusion_matrix(y_t, y_pred)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    print_header(f"ANALYSE DES ERREURS — {name}")

    print("Matrice normalisée par vraie classe :")
    print(f"{'':20s}" + " ".join(f"{c[:10]:>10s}" for c in CLASS_NAMES))
    for i, row in enumerate(cm_norm):
        print(f"{CLASS_NAMES[i]:20s}" + " ".join(f"{v:10.3f}" for v in row))

    errors = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm[i, j] > 0:
                errors.append((int(cm[i, j]), CLASS_NAMES[i], CLASS_NAMES[j]))

    errors.sort(reverse=True)

    print("\nTop erreurs :")
    for n, true_name, pred_name in errors[:10]:
        print(f"  {true_name} -> {pred_name} : {n}")


def save_feature_importance(model, name, filename_base):
    if not hasattr(model, "feature_importances_"):
        return

    imp = model.feature_importances_
    idx = np.argsort(imp)[-25:]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(range(len(idx)), imp[idx])
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([FEATURE_COLS[i] if i < len(FEATURE_COLS) else f"f{i}" for i in idx], fontsize=8)
    ax.set_xlabel("Importance")
    ax.set_title(f"Top 25 Features — {name}")
    plt.tight_layout()
    plt.savefig(out_path(f"{filename_base}.png"), dpi=160, bbox_inches="tight")
    plt.savefig(out_path(f"{filename_base}.eps"), format="eps", bbox_inches="tight")
    plt.close()



def _plot_learning_curve_metric(
    train_sizes_abs,
    train_scores,
    val_scores,
    name,
    filename_base,
    metric_label,
    title_suffix,
    y_min=0.0,
    y_max=1.0,
):
    tr_mean = train_scores.mean(axis=1)
    tr_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(7.5, 5.2))

    ax.plot(train_sizes_abs, tr_mean, "o-", label="Train")
    ax.fill_between(
        train_sizes_abs,
        np.maximum(y_min, tr_mean - tr_std),
        np.minimum(y_max, tr_mean + tr_std),
        alpha=0.15,
    )

    ax.plot(train_sizes_abs, val_mean, "o-", label="Validation CV")
    ax.fill_between(
        train_sizes_abs,
        np.maximum(y_min, val_mean - val_std),
        np.minimum(y_max, val_mean + val_std),
        alpha=0.15,
    )

    ax.set_xlabel("Taille du training set")
    ax.set_ylabel(metric_label)
    ax.set_title(f"Learning Curve — {name} — {title_suffix}")
    ax.set_ylim(y_min, y_max)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path(f"{filename_base}.png"), dpi=170, bbox_inches="tight")
    plt.savefig(out_path(f"{filename_base}.eps"), format="eps", bbox_inches="tight")
    plt.close()

    print(f"  Courbe sauvegardée : {out_path(filename_base + '.png')}")


def save_learning_curves(model, X, y, name, filename_base):
    """
    Génère trois learning curves :
      1. F1 macro : axe Y de 0 à 1
      2. Accuracy : axe Y de 0 à 1
      3. Log loss : axe Y à partir de 0, limite haute adaptée

    Remarque :
      - Pour F1 et Accuracy, fixer l'axe Y à 0→1 évite une lecture trop optimiste.
      - Pour Log loss, l'axe commence à 0. La borne haute est automatique,
        car une loss multi-classe peut dépasser 1.
    """
    try:
        train_sizes = np.linspace(0.1, 1.0, 7)
        cv = StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE)

        # F1 macro
        tr_sizes_f1, tr_f1, val_f1 = learning_curve(
            model,
            X,
            y,
            cv=cv,
            scoring="f1_macro",
            train_sizes=train_sizes,
            n_jobs=-1,
        )

        _plot_learning_curve_metric(
            tr_sizes_f1,
            tr_f1,
            val_f1,
            name,
            f"{filename_base}_f1_macro",
            "F1 macro",
            "F1 macro",
            y_min=0.0,
            y_max=1.0,
        )

        # Accuracy
        tr_sizes_acc, tr_acc, val_acc = learning_curve(
            model,
            X,
            y,
            cv=cv,
            scoring="accuracy",
            train_sizes=train_sizes,
            n_jobs=-1,
        )

        _plot_learning_curve_metric(
            tr_sizes_acc,
            tr_acc,
            val_acc,
            name,
            f"{filename_base}_accuracy",
            "Accuracy",
            "Accuracy",
            y_min=0.0,
            y_max=1.0,
        )

        # Log loss : sklearn retourne neg_log_loss
        tr_sizes_loss, tr_neg_loss, val_neg_loss = learning_curve(
            model,
            X,
            y,
            cv=cv,
            scoring="neg_log_loss",
            train_sizes=train_sizes,
            n_jobs=-1,
        )

        tr_loss = -tr_neg_loss
        val_loss = -val_neg_loss

        loss_upper = float(
            max(
                1.0,
                np.nanmax(tr_loss) * 1.08,
                np.nanmax(val_loss) * 1.08,
            )
        )

        _plot_learning_curve_metric(
            tr_sizes_loss,
            tr_loss,
            val_loss,
            name,
            f"{filename_base}_log_loss",
            "Log loss",
            "Log loss",
            y_min=0.0,
            y_max=loss_upper,
        )

    except Exception as e:
        print(f"Learning curves ignorées pour {name}. Erreur : {e}")


def save_learning_curve_fig(model, X, y, name, filename_base):
    # Compatibilité avec les anciens appels éventuels.
    save_learning_curves(model, X, y, name, filename_base)



def compute_roc_data(probas, y_bin):
    fpr, tpr, roc_auc = {}, {}, {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], probas[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)

    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(all_fpr, mean_tpr)

    return fpr, tpr, roc_auc


def save_roc_figure(probas, y_t, name, filename_base):
    try:
        y_bin = label_binarize(y_t, classes=list(range(n_classes)))
        fpr, tpr, roc_auc = compute_roc_data(probas, y_bin)

        fig, ax = plt.subplots(figsize=(8, 6))

        for i in range(n_classes):
            ax.plot(
                fpr[i],
                tpr[i],
                lw=1.5,
                label=f"{CLASS_NAMES[i]} AUC={roc_auc[i]:.3f}",
            )

        ax.plot(
            fpr["macro"],
            tpr["macro"],
            "k--",
            lw=2.0,
            label=f"Macro AUC={roc_auc['macro']:.3f}",
        )

        ax.plot([0, 1], [0, 1], "k:", lw=1)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC — {name}")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_path(f"{filename_base}.png"), dpi=160, bbox_inches="tight")
        plt.savefig(out_path(f"{filename_base}.eps"), format="eps", bbox_inches="tight")
        plt.close()

        return float(roc_auc["macro"])

    except Exception as e:
        print(f"ROC ignoré pour {name}. Erreur : {e}")
        return None


# ==============================================================
# 9. OPTUNA — TOUS LES MODELES
# ==============================================================

def optuna_randomforest_search(X, y, n_trials=30):
    def objective(trial):
        bootstrap = trial.suggest_categorical("bootstrap", [True, False])
        class_weight = trial.suggest_categorical(
            "class_weight",
            ["balanced", "balanced_subsample"],
        )

        if not bootstrap and class_weight == "balanced_subsample":
            class_weight = "balanced"

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 500, 2200),
            "max_depth": trial.suggest_categorical(
                "max_depth",
                [None, 10, 15, 20, 30, 40, 60, 80],
            ),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical(
                "max_features",
                ["sqrt", "log2", None],
            ),
            "bootstrap": bootstrap,
            "class_weight": class_weight,
        }

        scores = []
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

        for tr_idx, vl_idx in cv.split(X, y):
            model = RandomForestClassifier(
                **params,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
            model.fit(X[tr_idx], y[tr_idx])
            preds = model.predict(X[vl_idx])
            scores.append(f1_score(y[vl_idx], preds, average="macro", zero_division=0))

        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"Meilleur F1 CV RandomForest : {study.best_value:.4f}")
    print(f"Meilleurs paramètres RandomForest : {study.best_params}")
    return study.best_params


def optuna_extratrees_search(X, y, n_trials=30):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 500, 2500),
            "max_depth": trial.suggest_categorical(
                "max_depth",
                [None, 10, 15, 20, 30, 40, 60, 80],
            ),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical(
                "max_features",
                ["sqrt", "log2", None],
            ),
            "class_weight": trial.suggest_categorical(
                "class_weight",
                ["balanced", "balanced_subsample"],
            ),
            "bootstrap": trial.suggest_categorical("bootstrap", [False, True]),
        }

        scores = []
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

        for tr_idx, vl_idx in cv.split(X, y):
            model = ExtraTreesClassifier(
                **params,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
            model.fit(X[tr_idx], y[tr_idx])
            preds = model.predict(X[vl_idx])
            scores.append(f1_score(y[vl_idx], preds, average="macro", zero_division=0))

        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"Meilleur F1 CV ExtraTrees : {study.best_value:.4f}")
    print(f"Meilleurs paramètres ExtraTrees : {study.best_params}")
    return study.best_params


def optuna_xgb_search(X, y, sample_weight, n_trials=50):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 500, 2200),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
            "subsample": trial.suggest_float("subsample", 0.60, 1.00),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 1.00),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.60, 1.00),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 12),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "max_delta_step": trial.suggest_int("max_delta_step", 0, 8),
        }

        scores = []
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

        for tr_idx, vl_idx in cv.split(X, y):
            model = XGBClassifier(
                **params,
                objective="multi:softprob",
                num_class=n_classes,
                eval_metric="mlogloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=0,
            )
            model.fit(X[tr_idx], y[tr_idx], sample_weight=sample_weight[tr_idx])
            preds = model.predict(X[vl_idx])
            scores.append(f1_score(y[vl_idx], preds, average="macro", zero_division=0))

        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"Meilleur F1 CV XGBoost : {study.best_value:.4f}")
    print(f"Meilleurs paramètres XGBoost : {study.best_params}")
    return study.best_params


def optuna_catboost_search(X, y, X_val, y_val, sample_weight, n_trials=50):
    def objective(trial):
        bootstrap_type = trial.suggest_categorical(
            "bootstrap_type",
            ["Bayesian", "Bernoulli", "MVS"],
        )

        params = {
            "iterations": trial.suggest_int("iterations", 700, 3500),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.0, 3.0),
            "bootstrap_type": bootstrap_type,
        }

        if bootstrap_type == "Bayesian":
            params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 2.5)
        else:
            params["subsample"] = trial.suggest_float("subsample", 0.60, 1.00)

        model = CatBoostClassifier(
            **params,
            loss_function="MultiClass",
            eval_metric="TotalF1",
            random_seed=RANDOM_STATE,
            verbose=False,
            early_stopping_rounds=150,
            allow_writing_files=False,
        )
        model.fit(
            X,
            y,
            sample_weight=sample_weight,
            eval_set=(X_val, y_val),
            use_best_model=True,
        )
        preds = np.asarray(model.predict(X_val)).flatten()
        return float(f1_score(y_val, preds, average="macro", zero_division=0))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"Meilleur F1 validation CatBoost : {study.best_value:.4f}")
    print(f"Meilleurs paramètres CatBoost : {study.best_params}")
    return study.best_params


if USE_OPTUNA:
    print_header(f"OPTUNA RANDOM FOREST — {OPTUNA_TRIALS_RF} TRIALS")
    best_rf_params = optuna_randomforest_search(X_tr_bal, y_tr_bal, n_trials=OPTUNA_TRIALS_RF)

    print_header(f"OPTUNA EXTRATREES — {OPTUNA_TRIALS_ET} TRIALS")
    best_et_params = optuna_extratrees_search(X_tr_bal, y_tr_bal, n_trials=OPTUNA_TRIALS_ET)

    print_header(f"OPTUNA XGBOOST — {OPTUNA_TRIALS_XGB} TRIALS")
    best_xgb_params = optuna_xgb_search(
        X_tr_bal,
        y_tr_bal,
        sample_weights_bal,
        n_trials=OPTUNA_TRIALS_XGB,
    )

    print_header(f"OPTUNA CATBOOST — {OPTUNA_TRIALS_CAT} TRIALS")
    best_cat_params = optuna_catboost_search(
        X_tr_bal,
        y_tr_bal,
        X_val_s,
        y_val,
        sample_weights_bal,
        n_trials=OPTUNA_TRIALS_CAT,
    )
else:
    best_rf_params = {
        "n_estimators": 1400,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "class_weight": "balanced_subsample",
        "bootstrap": True,
    }
    best_et_params = {
        "n_estimators": 1800,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "bootstrap": False,
    }
    best_xgb_params = {
        "n_estimators": 1600,
        "max_depth": 8,
        "learning_rate": 0.025,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "colsample_bylevel": 0.85,
        "min_child_weight": 2,
        "gamma": 0.2,
        "reg_alpha": 0.05,
        "reg_lambda": 2.0,
        "max_delta_step": 0,
    }
    best_cat_params = {
        "iterations": 2500,
        "depth": 8,
        "learning_rate": 0.025,
        "l2_leaf_reg": 5,
        "bootstrap_type": "Bayesian",
        "bagging_temperature": 0.8,
        "random_strength": 1.0,
    }

# ==============================================================
# 10. MODELES INDIVIDUELS
# ==============================================================

results = {}
models = {}
predictions = {}
probas = {}
auc_summary = {}


# ------------------------------
# RandomForest
# ------------------------------

print_header("MODELE 1 — RANDOM FOREST")

rf_model = RandomForestClassifier(
    **best_rf_params,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

rf_model.fit(X_tr_bal, y_tr_bal)

y_pred_rf, results["RandomForest"] = evaluate_model(rf_model, X_test_s, y_test, "RandomForest")
predictions["RandomForest"] = y_pred_rf
models["RandomForest"] = rf_model

probas["RandomForest"] = rf_model.predict_proba(X_test_s)
auc_summary["RandomForest"] = save_roc_figure(probas["RandomForest"], y_test, "RandomForest", "roc_randomforest")
save_confusion_matrix(y_test, y_pred_rf, "RandomForest", "confusion_matrix_randomforest", "Greens")
save_feature_importance(rf_model, "RandomForest", "feature_importance_randomforest")
analyse_errors(y_test, y_pred_rf, "RandomForest")
joblib.dump(rf_model, out_path("randomforest_model.pkl"))


# ------------------------------
# ExtraTrees
# ------------------------------

print_header("MODELE 2 — EXTRATREES")

et_model = ExtraTreesClassifier(
    **best_et_params,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

et_model.fit(X_tr_bal, y_tr_bal)

y_pred_et, results["ExtraTrees"] = evaluate_model(et_model, X_test_s, y_test, "ExtraTrees")
predictions["ExtraTrees"] = y_pred_et
models["ExtraTrees"] = et_model

probas["ExtraTrees"] = et_model.predict_proba(X_test_s)
auc_summary["ExtraTrees"] = save_roc_figure(probas["ExtraTrees"], y_test, "ExtraTrees", "roc_extratrees")
save_confusion_matrix(y_test, y_pred_et, "ExtraTrees", "confusion_matrix_extratrees", "Oranges")
save_feature_importance(et_model, "ExtraTrees", "feature_importance_extratrees")
analyse_errors(y_test, y_pred_et, "ExtraTrees")
joblib.dump(et_model, out_path("extratrees_model.pkl"))


# ------------------------------
# XGBoost
# ------------------------------

print_header("MODELE 3 — XGBOOST")

xgb_model = XGBClassifier(
    **best_xgb_params,
    objective="multi:softprob",
    num_class=n_classes,
    eval_metric="mlogloss",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=0,
)

xgb_model.fit(
    X_tr_bal,
    y_tr_bal,
    sample_weight=sample_weights_bal,
)

y_pred_xgb, results["XGBoost"] = evaluate_model(xgb_model, X_test_s, y_test, "XGBoost")
predictions["XGBoost"] = y_pred_xgb
models["XGBoost"] = xgb_model

probas["XGBoost"] = xgb_model.predict_proba(X_test_s)
auc_summary["XGBoost"] = save_roc_figure(probas["XGBoost"], y_test, "XGBoost", "roc_xgboost")
save_confusion_matrix(y_test, y_pred_xgb, "XGBoost", "confusion_matrix_xgboost", "Blues")
save_feature_importance(xgb_model, "XGBoost", "feature_importance_xgboost")
analyse_errors(y_test, y_pred_xgb, "XGBoost")
joblib.dump(xgb_model, out_path("xgboost_model.pkl"))


# ------------------------------
# CatBoost
# ------------------------------

print_header("MODELE 4 — CATBOOST")

cat_model = CatBoostClassifier(
    **best_cat_params,
    loss_function="MultiClass",
    eval_metric="TotalF1",
    random_seed=RANDOM_STATE,
    verbose=200,
    early_stopping_rounds=150,
    allow_writing_files=False,
)

cat_model.fit(
    X_tr_bal,
    y_tr_bal,
    eval_set=(X_val_s, y_val),
    sample_weight=sample_weights_bal,
    use_best_model=True,
)

y_pred_cat = np.asarray(cat_model.predict(X_test_s)).flatten()
results["CatBoost"] = evaluate_predictions(y_test, y_pred_cat, "CatBoost")
predictions["CatBoost"] = y_pred_cat
models["CatBoost"] = cat_model

probas["CatBoost"] = cat_model.predict_proba(X_test_s)
auc_summary["CatBoost"] = save_roc_figure(probas["CatBoost"], y_test, "CatBoost", "roc_catboost")
save_confusion_matrix(y_test, y_pred_cat, "CatBoost", "confusion_matrix_catboost", "Purples")
analyse_errors(y_test, y_pred_cat, "CatBoost")
joblib.dump(cat_model, out_path("catboost_model.pkl"))



# ==============================================================
# 10B. LEARNING CURVES — F1, ACCURACY, LOG LOSS
# ==============================================================

print_header("LEARNING CURVES — F1 / ACCURACY / LOG LOSS")

learning_curve_models = {
    "RandomForest": rf_model,
    "ExtraTrees": et_model,
    "XGBoost": xgb_model,
    "CatBoost": cat_model,
}

for lc_name, lc_model in learning_curve_models.items():
    print(f"\nLearning curves : {lc_name}")
    save_learning_curves(
        lc_model,
        X_tr_s,
        y_tr_res,
        lc_name,
        f"learning_curve_{lc_name.lower()}",
    )


# ==============================================================
# 11. CALIBRATION DES MODELES POUR ENSEMBLE
# ==============================================================

print_header("CALIBRATION DES MODELES")

calibrated_models = {}
calibrated_probas_val = {}
calibrated_probas_test = {}

for name, model in models.items():
    try:
        print(f"Calibration : {name}")
        cal = CalibratedClassifierCV(model, cv="prefit", method="isotonic")
        cal.fit(X_val_s, y_val)

        calibrated_models[name] = cal
        calibrated_probas_val[name] = cal.predict_proba(X_val_s)
        calibrated_probas_test[name] = cal.predict_proba(X_test_s)

        joblib.dump(cal, out_path(f"{name.lower()}_calibrated.pkl"))

    except Exception as e:
        print(f"Calibration ignorée pour {name}. Erreur : {e}")
        calibrated_models[name] = model
        calibrated_probas_val[name] = model.predict_proba(X_val_s)
        calibrated_probas_test[name] = model.predict_proba(X_test_s)


# ==============================================================
# 12. ENSEMBLE — RECHERCHE DES MEILLEURS POIDS
# ==============================================================

print_header("ENSEMBLE — RECHERCHE DES MEILLEURS POIDS")

ensemble_names = list(calibrated_probas_val.keys())
N = len(ensemble_names)

rng = np.random.default_rng(RANDOM_STATE)

# Favoriser XGBoost et CatBoost, sans éliminer les arbres
alpha = []
for name in ensemble_names:
    if name in ["XGBoost", "CatBoost"]:
        alpha.append(4.0)
    elif name in ["ExtraTrees"]:
        alpha.append(2.0)
    else:
        alpha.append(1.0)

alpha = np.array(alpha, dtype=float)

best_weights = np.ones(N) / N
best_val_f1 = -1.0
best_val_acc = -1.0

n_candidates = 50000
weight_candidates = rng.dirichlet(alpha, size=n_candidates)

for w in weight_candidates:
    combined_val = np.zeros_like(next(iter(calibrated_probas_val.values())))

    for i, name in enumerate(ensemble_names):
        combined_val += w[i] * calibrated_probas_val[name]

    pred_val = np.argmax(combined_val, axis=1)

    val_f1 = f1_score(y_val, pred_val, average="macro", zero_division=0)
    val_acc = accuracy_score(y_val, pred_val)

    if (val_f1 > best_val_f1) or (val_f1 == best_val_f1 and val_acc > best_val_acc):
        best_val_f1 = val_f1
        best_val_acc = val_acc
        best_weights = w.copy()

print("Meilleurs poids validation :")
for name, weight in zip(ensemble_names, best_weights):
    print(f"  {name:15s}: {weight:.4f}")

print(f"Validation F1 macro : {best_val_f1:.4f}")
print(f"Validation Accuracy : {best_val_acc:.4f}")

combined_test = np.zeros_like(next(iter(calibrated_probas_test.values())))

for i, name in enumerate(ensemble_names):
    combined_test += best_weights[i] * calibrated_probas_test[name]

y_pred_ensemble = np.argmax(combined_test, axis=1)

results["Ensemble_Weighted"] = evaluate_predictions(
    y_test,
    y_pred_ensemble,
    "Ensemble Weighted",
)

predictions["Ensemble_Weighted"] = y_pred_ensemble
probas["Ensemble_Weighted"] = combined_test

auc_summary["Ensemble_Weighted"] = save_roc_figure(
    combined_test,
    y_test,
    "Ensemble Weighted",
    "roc_ensemble_weighted",
)

save_confusion_matrix(
    y_test,
    y_pred_ensemble,
    "Ensemble Weighted",
    "confusion_matrix_ensemble_weighted",
    "Purples",
)

analyse_errors(y_test, y_pred_ensemble, "Ensemble Weighted")


# ==============================================================
# 13. OPTIMISATION DES SEUILS PAR CLASSE
# ==============================================================

print_header("OPTIMISATION DES SEUILS PAR CLASSE")

combined_val_best = np.zeros_like(next(iter(calibrated_probas_val.values())))

for i, name in enumerate(ensemble_names):
    combined_val_best += best_weights[i] * calibrated_probas_val[name]

thresholds = np.ones(n_classes) * (1.0 / n_classes)

for cls in range(n_classes):
    best_t = thresholds[cls]
    best_score = -1.0

    for t in np.arange(0.05, 0.95, 0.01):
        tmp_thresholds = thresholds.copy()
        tmp_thresholds[cls] = t

        adjusted_val = combined_val_best.copy()

        for c_idx in range(n_classes):
            adjusted_val[:, c_idx] /= (tmp_thresholds[c_idx] + 1e-8)

        pred_val = np.argmax(adjusted_val, axis=1)
        score = f1_score(y_val, pred_val, average="macro", zero_division=0)

        if score > best_score:
            best_score = score
            best_t = t

    thresholds[cls] = best_t

print("Seuils optimisés :")
for cls_idx, t in enumerate(thresholds):
    print(f"  {CLASS_NAMES[cls_idx]:20s}: {t:.3f}")

adjusted_test = combined_test.copy()

for c_idx in range(n_classes):
    adjusted_test[:, c_idx] /= (thresholds[c_idx] + 1e-8)

adjusted_test = adjusted_test / (adjusted_test.sum(axis=1, keepdims=True) + 1e-8)
y_pred_threshold = np.argmax(adjusted_test, axis=1)

results["Ensemble_Threshold"] = evaluate_predictions(
    y_test,
    y_pred_threshold,
    "Ensemble Weighted + Thresholds",
)

predictions["Ensemble_Threshold"] = y_pred_threshold
probas["Ensemble_Threshold"] = adjusted_test

auc_summary["Ensemble_Threshold"] = save_roc_figure(
    adjusted_test,
    y_test,
    "Ensemble Threshold",
    "roc_ensemble_threshold",
)

save_confusion_matrix(
    y_test,
    y_pred_threshold,
    "Ensemble Threshold",
    "confusion_matrix_ensemble_threshold",
    "Purples",
)

analyse_errors(y_test, y_pred_threshold, "Ensemble Threshold")


# ==============================================================
# 14. COMPARAISON FINALE
# ==============================================================

print_header("COMPARAISON FINALE")

best_model_name = max(
    results.keys(),
    key=lambda k: (results[k]["f1_macro"], results[k]["accuracy"]),
)

for name, metrics in results.items():
    marker = "  <-- MEILLEUR" if name == best_model_name else ""
    print(
        f"{name:24s} "
        f"Acc={metrics['accuracy']:.4f}  "
        f"F1_macro={metrics['f1_macro']:.4f}  "
        f"Recall_macro={metrics['recall_macro']:.4f}"
        f"{marker}"
    )

print("\nAUC macro :")
for name, value in auc_summary.items():
    if value is not None:
        print(f"{name:24s} AUC={value:.4f}")


# ==============================================================
# 15. FIGURE COMPARATIVE
# ==============================================================

names = list(results.keys())
accs = [results[k]["accuracy"] for k in names]
f1s = [results[k]["f1_macro"] for k in names]
recalls = [results[k]["recall_macro"] for k in names]

x = np.arange(len(names))
w = 0.25

fig, ax = plt.subplots(figsize=(12, 6))
b1 = ax.bar(x - w, accs, w, label="Accuracy")
b2 = ax.bar(x, f1s, w, label="F1 macro")
b3 = ax.bar(x + w, recalls, w, label="Recall macro")

for bars in [b1, b2, b3]:
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{bar.get_height():.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

ax.axhline(TARGET_SCORE, linestyle="--", linewidth=1.5, label="Objectif 0.95")
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=25, ha="right")
ax.set_ylim(0.8, 1.0)
ax.set_ylabel("Score")
ax.set_title("Comparaison des modèles — objectif 95 % — axe Y 0.8 à 1.0")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(out_path("comparison_models_v7.png"), dpi=160, bbox_inches="tight")
plt.savefig(out_path("comparison_models_y08_1.png"), dpi=160, bbox_inches="tight")
plt.savefig(out_path("comparison_models_v7.eps"), format="eps", bbox_inches="tight")
plt.savefig(out_path("comparison_models_y08_1.eps"), format="eps", bbox_inches="tight")
plt.close()


# ==============================================================
# 16. SAUVEGARDE DES PREPROCESSEURS ET PARAMETRES
# ==============================================================

print_header("SAUVEGARDE")

joblib.dump(scaler, out_path("scaler.pkl"))
joblib.dump(label_encoders, out_path("label_encoders.pkl"))
joblib.dump(target_encoder, out_path("target_encoder.pkl"))
joblib.dump(train_stats, out_path("train_stats.pkl"))

np.save(out_path("ensemble_weights.npy"), best_weights)
np.save(out_path("class_thresholds.npy"), thresholds)

with open(out_path("class_names.json"), "w", encoding="utf-8") as f:
    json.dump(CLASS_NAMES, f, indent=2, ensure_ascii=False)

with open(out_path("results_metrics.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

with open(out_path("auc_summary.json"), "w", encoding="utf-8") as f:
    json.dump(auc_summary, f, indent=2, ensure_ascii=False)

with open(out_path("ensemble_config.json"), "w", encoding="utf-8") as f:
    json.dump(
        {
            "ensemble_names": ensemble_names,
            "best_weights": best_weights.tolist(),
            "thresholds": thresholds.tolist(),
            "best_model_name": best_model_name,
            "balancing_method": balancing_method,
            "feature_mode": "raw_features_no_engineering",
            "target_score": TARGET_SCORE,
            "use_optuna": USE_OPTUNA,
            "optuna_trials_rf": OPTUNA_TRIALS_RF,
            "optuna_trials_et": OPTUNA_TRIALS_ET,
            "optuna_trials_xgb": OPTUNA_TRIALS_XGB,
            "optuna_trials_cat": OPTUNA_TRIALS_CAT,
            "best_rf_params": best_rf_params,
            "best_et_params": best_et_params,
            "best_xgb_params": best_xgb_params,
            "best_cat_params": best_cat_params,
        },
        f,
        indent=2,
        ensure_ascii=False,
    )

print(f"Tous les fichiers sont sauvegardés dans : {OUTPUT_DIR}")
print("Version utilisée : sans feature engineering, 4 variables seulement.")
print("Graphe comparaison : axe Y fixé entre 0.8 et 1.0.")
print("  comparison_models_y08_1.png/eps")
print("Learning curves générées :")
print("  learning_curve_<model>_f1_macro.png/eps")
print("  learning_curve_<model>_accuracy.png/eps")
print("  learning_curve_<model>_log_loss.png/eps")


# ==============================================================
# 17. VERIFICATION OBJECTIF 95 %
# ==============================================================

print_header("VERIFICATION OBJECTIF 95 %")

best_metrics = results[best_model_name]
best_acc = best_metrics["accuracy"]
best_f1 = best_metrics["f1_macro"]
best_recall = best_metrics["recall_macro"]

print(f"Meilleur modèle : {best_model_name}")
print(f"Accuracy     : {best_acc:.4f}")
print(f"F1 macro     : {best_f1:.4f}")
print(f"Recall macro : {best_recall:.4f}")

if best_acc >= TARGET_SCORE and best_f1 >= TARGET_SCORE:
    print("\nObjectif atteint : Accuracy et F1 macro >= 95 %.")
elif best_acc >= TARGET_SCORE:
    print("\nAccuracy >= 95 %, mais F1 macro < 95 %.")
    print("Cela indique que certaines classes restent mal reconnues.")
elif best_f1 >= TARGET_SCORE:
    print("\nF1 macro >= 95 %, mais Accuracy < 95 %.")
    print("Les classes sont mieux équilibrées, mais le score global reste inférieur.")
else:
    print("\nObjectif 95 % non atteint.")
    print("Actions recommandées :")
    print("  1. Vérifier les labels bruités ou incohérents.")
    print("  2. Examiner les confusions dans les matrices sauvegardées.")
    print("  3. Comparer avec la version feature engineering si les scores baissent fortement.")
    print("  4. Augmenter les OPTUNA_TRIALS_* à 80, 100 ou 150.")
    print("  5. Améliorer les variables explicatives si les classes restent fortement confondues.")

print("\nTERMINE")
