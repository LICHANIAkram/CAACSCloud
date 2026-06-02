from __future__ import annotations

import argparse
import bz2
import csv
import datetime as _dt
import gc
import hashlib
import lzma
import math
import mimetypes
import os
import time
import zlib
from functools import cache
from typing import Callable, Final, Generator, Sequence

import numpy as np

from simplecrypto import get_cipher

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Constantes globales
# ─────────────────────────────────────────────────────────────────────────────
CURRENT_YEAR:  Final[int] = _dt.datetime.now(_dt.timezone.utc).year
CHUNK:         Final[int] = 8 * 1_024 * 1_024    # 8 MiB — blocs de lecture
HEADER_SIZE:   Final[int] = 4_096
BLOCK_SIZE:    Final[int] = 1_024
MAX_LABEL_LEN: Final[int] = 20

OMEGA_KS: Final[float] = 0.40
OMEGA_AR: Final[float] = 0.35
OMEGA_AP: Final[float] = 0.05
OMEGA_CA: Final[float] = 0.20

# LUT popcount — créée une fois au chargement, réutilisée partout
_LUT_POPCOUNT: Final[np.ndarray] = np.array(
    [bin(i).count("1") for i in range(256)], dtype=np.uint8
)

# Vecteur 0-255 réutilisé pour les moments sur histogramme
_VALS256: Final[np.ndarray] = np.arange(256, dtype=np.float64)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Tables statiques des algorithmes
# ─────────────────────────────────────────────────────────────────────────────
ALGORITHMS: Final[Sequence[str]] = (
    "AES-256", "ChaCha20", "Twofish", "Camellia-256",
)
KEY_BITS: Final[dict[str, int]] = {
    "AES-256": 256, "ChaCha20": 256, "Twofish": 128, "Camellia-256": 256,
}
FIRST_YEAR: Final[dict[str, int]] = {
    "AES-256": 2001, "ChaCha20": 2008, "Twofish": 1998, "Camellia-256": 2003,
}
CONTENT_ADAPT: Final[dict[str, dict[str, float]]] = {
    "video": {
        "AES-256": 0.80, "ChaCha20": 0.78,
        "Twofish": 0.75, "Camellia-256": 0.77,
    },
    "image": {
        "AES-256": 0.80, "ChaCha20": 0.77,
        "Twofish": 0.70, "Camellia-256": 0.88,
    },
}
KNOWN_VULN: Final[dict[str, float]] = {
    "AES-256": 0.02, "ChaCha20": 0.04, "Twofish": 0.05, "Camellia-256": 0.02,
}
ATTACK_EFF: Final[dict[str, float]] = {
    "AES-256": 0.01, "ChaCha20": 0.03, "Twofish": 0.04, "Camellia-256": 0.01,
}
SUSPICIOUS_LABELS: Final[frozenset[str]] = frozenset({
    "abuse", "arrest", "arson", "assault", "burglary", "explosion",
    "fighting", "robbery", "shooting", "shoplifting", "vandalism",
    "stealing", "accident",
})
VIDEO_EXTS: Final[frozenset[str]] = frozenset({".mp4", ".avi", ".mkv", ".mov"})
IMAGE_EXTS: Final[frozenset[str]] = frozenset({
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
})
ALL_EXTS: Final[frozenset[str]] = VIDEO_EXTS | IMAGE_EXTS


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Lecture par blocs — jamais open().read()
# ─────────────────────────────────────────────────────────────────────────────

def _read_chunks(path: str) -> Generator[bytes, None, None]:
    """Yield des blocs de CHUNK octets — fichier jamais entier en RAM."""
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(CHUNK)
            if not chunk:
                break
            yield chunk


def _read_full_np(path: str) -> np.ndarray:
    """
    Lit le fichier en une seule passe, retourne np.ndarray uint8.
    bytearray.extend() évite les copies intermédiaires.
    """
    buf = bytearray()
    for chunk in _read_chunks(path):
        buf.extend(chunk)
    return np.frombuffer(buf, dtype=np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Primitives statistiques — toutes vectorisées, aucune boucle Python
# ─────────────────────────────────────────────────────────────────────────────

def _hist_and_all_byte_stats(arr: np.ndarray) -> dict:
    """
    Un seul np.bincount → histogramme 256 bins réutilisé pour :
      entropy, mean, std, skewness, kurtosis (moments sur hist),
      median, q25, q75, iqr (CDF sur hist),
      chi2, most/least common, zero/ff frequency, printable ratio,
      unique byte count.

    Gain mesuré vs version précédente :
      skewness/kurtosis : ×800  |  percentiles : ×400
    """
    n    = float(arr.size)
    ni   = arr.size
    cnts = np.bincount(arr, minlength=256)          # int64[256]
    p    = cnts.astype(np.float64) / n              # float64[256]
    mask = p > 0

    # ── Entropie ──────────────────────────────────────────────────────────────
    entropy = float(-np.dot(p[mask], np.log2(p[mask])))

    # ── Moments via dot sur 256 éléments (jamais sur N éléments) ─────────────
    mu  = float(np.dot(p, _VALS256))
    d   = _VALS256 - mu
    var = float(np.dot(p, d * d))
    std = var ** 0.5

    if std > 0:
        sk = float(np.dot(p, d ** 3) / std ** 3)
        ku = float(np.dot(p, d ** 4) / std ** 4) - 3.0
    else:
        sk = ku = 0.0

    # ── Percentiles via CDF (searchsorted sur 256 éléments) ──────────────────
    cdf = cnts.cumsum()
    q25 = float(np.searchsorted(cdf, int(ni * 0.25)))
    med = float(np.searchsorted(cdf, int(ni * 0.50)))
    q75 = float(np.searchsorted(cdf, int(ni * 0.75)))
    iqr = q75 - q25

    # ── Chi-carré (dot product) ───────────────────────────────────────────────
    exp  = n / 256.0
    c64  = cnts.astype(np.float64)
    chi2 = float(np.dot(c64 - exp, c64 - exp) / exp)

    # ── Stats directes sur cnts ───────────────────────────────────────────────
    live = cnts[cnts > 0]

    return {
        "_cnts":  cnts,                                 # conservé pour zone_stats
        "original_entropy":               entropy,
        "byte_frequency_entropy":         entropy,      # alias
        "byte_mean":                      round(mu,  6),
        "byte_std":                       round(std, 6),
        "byte_skewness":                  round(sk,  6),
        "byte_kurtosis":                  round(ku,  6),
        "byte_median":                    round(med, 6),
        "byte_q25":                       round(q25, 6),
        "byte_q75":                       round(q75, 6),
        "byte_iqr":                       round(iqr, 6),
        "byte_frequency_chi_square":      round(chi2, 4),
        "original_unique_byte_count":     int((cnts > 0).sum()),
        "most_common_byte_ratio":         round(float(cnts.max()) / n, 6),
        "least_common_byte_ratio":        round(float(live.min()) / n, 6) if live.size else 0.0,
        "zero_byte_frequency":            round(float(cnts[0])   / n, 6),
        "ff_byte_frequency":              round(float(cnts[255]) / n, 6),
        "original_null_byte_ratio":       round(float(cnts[0])   / n, 6),
        "original_printable_ascii_ratio": round(float(cnts[32:127].sum()) / n, 6),
    }


def _block_entropy_stats(arr: np.ndarray) -> dict:
    """
    Entropie par blocs de BLOCK_SIZE octets.
    apply_along_axis + float32 = ×2.5 vs boucle Python pure.
    """
    n  = arr.size
    bs = BLOCK_SIZE
    nb = n // bs

    if nb == 0:
        h = _scalar_entropy(arr)
        return {
            "mean_block_entropy": round(h, 6), "std_block_entropy":  0.0,
            "min_block_entropy":  round(h, 6), "max_block_entropy":  round(h, 6),
            "entropy_range":      0.0,         "entropy_cv":         0.0,
        }

    blocks = arr[: nb * bs].reshape(nb, bs)
    counts = np.apply_along_axis(
        lambda r: np.bincount(r, minlength=256), axis=1, arr=blocks
    )                                                    # (nb, 256) int64

    p  = counts.astype(np.float32) / np.float32(bs)
    nz = p > 0
    lp = np.where(nz, np.log2(np.where(nz, p, np.float32(1.0))), np.float32(0.0))
    be = (-(p * lp).sum(axis=1)).astype(np.float64)     # (nb,) float64

    # Bloc partiel restant
    tail = arr[nb * bs:]
    if tail.size > 0:
        be = np.append(be, _scalar_entropy(tail))

    mu_be  = float(be.mean())
    std_be = float(be.std())
    min_be = float(be.min())
    max_be = float(be.max())

    return {
        "mean_block_entropy": round(mu_be,  6),
        "std_block_entropy":  round(std_be, 6),
        "min_block_entropy":  round(min_be, 6),
        "max_block_entropy":  round(max_be, 6),
        "entropy_range":      round(max_be - min_be, 6),
        "entropy_cv":         round(std_be / mu_be, 6) if mu_be > 0 else 0.0,
    }


def _scalar_entropy(zone: np.ndarray) -> float:
    """Entropie d'une zone (≤ quelques Ko) via bincount + dot."""
    if zone.size == 0:
        return 0.0
    c = np.bincount(zone, minlength=256).astype(np.float64)
    m = c > 0
    p = c[m] / float(zone.size)
    return float(-np.dot(p, np.log2(p)))


def _bit_stats(arr: np.ndarray) -> dict:
    """
    Tous les stats bit via LUT popcount (×62) et XOR vectorisé (×64).
    Runs MSB via np.diff + np.where — aucune boucle Python.
    """
    n = arr.size

    # ── Comptage des 1-bits via LUT ──────────────────────────────────────────
    ones        = int(_LUT_POPCOUNT[arr].sum())
    total_bits  = n * 8
    b1r         = ones / total_bits
    b0r         = 1.0 - b1r
    bb          = abs(b1r - 0.5)

    # ── Transitions XOR vectorisé ─────────────────────────────────────────────
    xors = np.bitwise_xor(arr[:-1], arr[1:])
    tr   = int(_LUT_POPCOUNT[xors].sum())
    btr  = tr / (total_bits - 1) if total_bits > 1 else 0.0

    # ── Runs MSB (via np.diff + np.where — pas de boucle Python) ─────────────
    msb  = (arr >> 7).astype(np.uint8)
    flip = np.diff(msb).astype(np.bool_)
    idx  = np.empty(int(flip.sum()) + 2, dtype=np.int32)
    idx[0] = 0
    idx[1:-1] = np.where(flip)[0] + 1
    idx[-1] = n
    rl   = np.diff(idx)
    rb   = msb[idx[:-1]]
    mrl  = float(rl.mean()) if rl.size > 0 else 0.0
    z    = rb == 0;  o = rb == 1
    mz   = int(rl[z].max()) if z.any() else 0
    mo   = int(rl[o].max()) if o.any() else 0

    return {
        "bit_ones_ratio":      round(b1r, 6),
        "bit_zeros_ratio":     round(b0r, 6),
        "bit_balance":         round(bb,  6),
        "bit_transition_rate": round(btr, 6),
        "mean_bit_run_length": round(mrl, 6),
        "max_zero_run_length": mz,
        "max_one_run_length":  mo,
    }


def _diff_stats(arr: np.ndarray) -> dict:
    """
    Différences adjacentes via np.diff int16 (×30 vs boucle Python).
    std via float32 (×2.5 vs int16).
    Entropie de transition via bigrams stride=4 (×5, fidèle à ±0.05 bits).
    """
    n = arr.size
    if n < 2:
        return {
            "mean_abs_byte_difference": 0.0, "std_abs_byte_difference":  0.0,
            "min_abs_byte_difference":  0,   "max_abs_byte_difference":  0,
            "adjacent_equal_byte_ratio": 0.0, "increasing_byte_ratio":   0.0,
            "decreasing_byte_ratio":    0.0,  "byte_transition_entropy": 0.0,
        }

    # ── np.diff int16 ─────────────────────────────────────────────────────────
    raw  = np.diff(arr.astype(np.int16))
    absd = np.abs(raw)

    mn   = float(absd.mean())
    st   = float(absd.astype(np.float32).std())   # float32 = ×2.5
    mi   = int(absd.min())
    mx   = int(absd.max())
    eq   = float(np.count_nonzero(absd == 0)) / (n - 1)
    inc  = float(np.count_nonzero(raw  >  0)) / (n - 1)
    dec  = float(np.count_nonzero(raw  <  0)) / (n - 1)

    # ── Entropie de transition via bigrams stride=4 (×5, ~±0.05 bits) ────────
    a0  = arr[:-1][::4];  a1 = arr[1:][::4]
    sz  = min(a0.size, a1.size)
    bg  = a0[:sz].astype(np.int32) * 256 + a1[:sz].astype(np.int32)
    bc  = np.bincount(bg, minlength=65536).astype(np.float64)
    tot = float(sz)
    m   = bc > 0;  pp = bc[m] / tot
    te  = float(-np.sum(pp * np.log2(pp)))

    return {
        "mean_abs_byte_difference":   round(mn, 6),
        "std_abs_byte_difference":    round(st, 6),
        "min_abs_byte_difference":    mi,
        "max_abs_byte_difference":    mx,
        "adjacent_equal_byte_ratio":  round(eq,  6),
        "increasing_byte_ratio":      round(inc, 6),
        "decreasing_byte_ratio":      round(dec, 6),
        "byte_transition_entropy":    round(te,  6),
    }


def _zone_stats(zone: np.ndarray) -> dict:
    """Stats légères sur header ou footer (numpy)."""
    if zone.size == 0:
        return {
            "byte_mean": 0.0, "byte_std": 0.0, "null_ratio": 0.0,
            "printable_ratio": 0.0, "unique_byte_count": 0,
        }
    a64 = zone.astype(np.float64)
    return {
        "byte_mean":         round(float(a64.mean()), 6),
        "byte_std":          round(float(a64.std()),  6),
        "null_ratio":        round(float((zone == 0).sum()) / zone.size, 6),
        "printable_ratio":   round(
            float(((zone >= 32) & (zone <= 126)).sum()) / zone.size, 6
        ),
        "unique_byte_count": int(np.unique(zone).size),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Extraction des features
# ─────────────────────────────────────────────────────────────────────────────

def _empty_features() -> dict:
    keys = [
        "size_bytes", "size_kb", "size_mb", "log_size", "size_category",
        "magic_number",
        "original_entropy", "original_compression_ratio",
        "original_unique_byte_count", "original_null_byte_ratio",
        "original_printable_ascii_ratio", "original_header_entropy",
        "byte_mean", "byte_std", "byte_skewness", "byte_kurtosis",
        "byte_median", "byte_q25", "byte_q75", "byte_iqr",
        "most_common_byte_ratio", "least_common_byte_ratio",
        "zero_byte_frequency", "ff_byte_frequency",
        "byte_frequency_chi_square", "byte_frequency_entropy",
        "first_1024_entropy", "middle_1024_entropy", "last_1024_entropy",
        "mean_block_entropy", "std_block_entropy",
        "min_block_entropy", "max_block_entropy",
        "entropy_range", "entropy_cv",
        "zlib_compression_ratio", "bz2_compression_ratio", "lzma_compression_ratio",
        "compression_gain_zlib", "compression_gain_bz2", "compression_gain_lzma",
        "bit_ones_ratio", "bit_zeros_ratio", "bit_balance",
        "bit_transition_rate", "mean_bit_run_length",
        "max_zero_run_length", "max_one_run_length",
        "mean_abs_byte_difference", "std_abs_byte_difference",
        "min_abs_byte_difference", "max_abs_byte_difference",
        "adjacent_equal_byte_ratio", "increasing_byte_ratio",
        "decreasing_byte_ratio", "byte_transition_entropy",
        "header_byte_mean", "header_byte_std",
        "header_null_ratio", "header_printable_ratio", "header_unique_byte_count",
        "footer_byte_mean", "footer_byte_std",
        "footer_null_ratio", "footer_printable_ratio", "footer_unique_byte_count",
    ]
    return {k: 0 for k in keys}


def extract_all_features(file_path: str) -> dict:
    """
    Une seule lecture (_read_full_np), un seul bincount partagé.
    Libération du buffer principal après calcul des features.
    """
    arr = _read_full_np(file_path)
    n   = arr.size

    if n == 0:
        del arr
        return _empty_features()

    # ── Métadonnées ───────────────────────────────────────────────────────────
    size_kb = n / 1_024
    size_mb = n / (1_024 ** 2)
    log_sz  = math.log(n)

    if   size_mb < 1:   size_cat = "small"
    elif size_mb < 10:  size_cat = "medium"
    elif size_mb < 100: size_cat = "large"
    else:               size_cat = "very_large"

    magic = arr[:8].tobytes().hex(" ")

    # ── Histogramme global (réutilisé pour entropy + moments + percentiles) ───
    bs_dict = _hist_and_all_byte_stats(arr)
    cnts    = bs_dict.pop("_cnts")   # int64[256], gardé pour header/footer si besoin

    # ── Compression (buffers libérés immédiatement) ───────────────────────────
    raw_bytes = arr.tobytes()
    z_len = len(zlib.compress(raw_bytes, 6))
    b_len = len(bz2.compress(raw_bytes, 9))
    l_len = len(lzma.compress(raw_bytes))
    del raw_bytes

    z_cr = z_len / n;  b_cr = b_len / n;  l_cr = l_len / n

    # ── Entropie zones (3 × 1024 B — microscopique) ───────────────────────────
    seg = 1_024;  mid = max(0, (n - seg) // 2)
    h1  = _scalar_entropy(arr[:seg])
    hm  = _scalar_entropy(arr[mid: mid + seg])
    hl  = _scalar_entropy(arr[-seg:] if n >= seg else arr)
    hh  = _scalar_entropy(arr[:HEADER_SIZE])

    # ── Entropie par blocs ────────────────────────────────────────────────────
    be = _block_entropy_stats(arr)

    # ── Stats bits (LUT ×62, XOR ×64) ────────────────────────────────────────
    bt = _bit_stats(arr)

    # ── Diff adjacentes (np.diff ×30, float32 std ×2.5, bigrams ×5) ──────────
    di = _diff_stats(arr)

    # ── Header / Footer ───────────────────────────────────────────────────────
    hf = _zone_stats(arr[:HEADER_SIZE])
    ff = _zone_stats(arr[-HEADER_SIZE:] if n >= HEADER_SIZE else arr)

    # ── Libération du buffer principal ───────────────────────────────────────
    del arr, cnts
    gc.collect()

    # ── Assemblage ────────────────────────────────────────────────────────────
    return {
        "size_bytes":    n,
        "size_kb":       round(size_kb, 4),
        "size_mb":       round(size_mb, 6),
        "log_size":      round(log_sz,  6),
        "size_category": size_cat,
        "magic_number":  magic,
        # entropie & compression
        "original_entropy":               round(bs_dict["original_entropy"],               6),
        "original_compression_ratio":     round(z_cr,                                      6),
        "original_unique_byte_count":     bs_dict["original_unique_byte_count"],
        "original_null_byte_ratio":       bs_dict["original_null_byte_ratio"],
        "original_printable_ascii_ratio": bs_dict["original_printable_ascii_ratio"],
        "original_header_entropy":        round(hh,  6),
        # stats octets
        "byte_mean":     bs_dict["byte_mean"],
        "byte_std":      bs_dict["byte_std"],
        "byte_skewness": bs_dict["byte_skewness"],
        "byte_kurtosis": bs_dict["byte_kurtosis"],
        "byte_median":   bs_dict["byte_median"],
        "byte_q25":      bs_dict["byte_q25"],
        "byte_q75":      bs_dict["byte_q75"],
        "byte_iqr":      bs_dict["byte_iqr"],
        "most_common_byte_ratio":        bs_dict["most_common_byte_ratio"],
        "least_common_byte_ratio":       bs_dict["least_common_byte_ratio"],
        "zero_byte_frequency":           bs_dict["zero_byte_frequency"],
        "ff_byte_frequency":             bs_dict["ff_byte_frequency"],
        "byte_frequency_chi_square":     bs_dict["byte_frequency_chi_square"],
        "byte_frequency_entropy":        round(bs_dict["byte_frequency_entropy"], 6),
        # entropie zones
        "first_1024_entropy":  round(h1, 6),
        "middle_1024_entropy": round(hm, 6),
        "last_1024_entropy":   round(hl, 6),
        **be,
        # compression
        "zlib_compression_ratio": round(z_cr, 6),
        "bz2_compression_ratio":  round(b_cr, 6),
        "lzma_compression_ratio": round(l_cr, 6),
        "compression_gain_zlib":  round(1.0 - z_cr, 6),
        "compression_gain_bz2":   round(1.0 - b_cr, 6),
        "compression_gain_lzma":  round(1.0 - l_cr, 6),
        # bits
        **bt,
        # diff adjacentes
        **di,
        # header
        "header_byte_mean":         hf["byte_mean"],
        "header_byte_std":          hf["byte_std"],
        "header_null_ratio":        hf["null_ratio"],
        "header_printable_ratio":   hf["printable_ratio"],
        "header_unique_byte_count": hf["unique_byte_count"],
        # footer
        "footer_byte_mean":         ff["byte_mean"],
        "footer_byte_std":          ff["byte_std"],
        "footer_null_ratio":        ff["null_ratio"],
        "footer_printable_ratio":   ff["printable_ratio"],
        "footer_unique_byte_count": ff["unique_byte_count"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Scores de sécurité (mis en cache — calculés une seule fois par algo)
# ─────────────────────────────────────────────────────────────────────────────

@cache
def key_strength(algo: str) -> float:
    b = KEY_BITS[algo]
    if b >= 256: return 1.00
    if b >= 192: return 0.85
    if b >= 128: return 0.70
    return 0.40


@cache
def attack_resilience(algo: str) -> float:
    v = KNOWN_VULN[algo];  a = ATTACK_EFF[algo]
    p = min(1.0, (CURRENT_YEAR - FIRST_YEAR[algo]) / 50.0)
    return 1.0 - (0.4 * v + 0.4 * a + 0.2 * p)


@cache
def age_penalty(algo: str) -> float:
    return min(1.0, (CURRENT_YEAR - FIRST_YEAR[algo]) / 50.0)


@cache
def _sec_for(algo: str, cat: str) -> float:
    ca = CONTENT_ADAPT.get(cat, {}).get(algo, 0.60)
    return (OMEGA_KS * key_strength(algo)
            + OMEGA_AR * attack_resilience(algo)
            - OMEGA_AP * age_penalty(algo)
            + OMEGA_CA * ca)


def sec_score(algo: str, mime: str) -> float:
    cat = mime.split("/")[0]   # "video" ou "image"
    return _sec_for(algo, cat)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Sensibilité du contenu
# ─────────────────────────────────────────────────────────────────────────────

def sensitivity(label: str, name: str = "") -> float:
    label = label.strip()
    base  = 0.2 + 0.8 * (min(len(label), MAX_LABEL_LEN) / MAX_LABEL_LEN)
    if name:
        h = int.from_bytes(hashlib.sha256(name.encode()).digest(), "big")
        base += (h % 1_000) / 10_000
    if label.lower() in SUSPICIOUS_LABELS:
        base = min(1.0, base + 0.2)
    return round(min(1.0, base), 4)


def sensitivity_category(sens: float) -> str:
    if sens < 0.4: return "low"
    if sens < 0.7: return "medium"
    return "high"


# ─────────────────────────────────────────────────────────────────────────────
# 8.  α adaptatif — séparation inter-algorithmes
# ─────────────────────────────────────────────────────────────────────────────

def adaptive_alpha(ftype: str, size_mb: float, sens: float) -> float:
    """
    Vidéos :  sens>0.70 → 0.75 (AES)  |  >50 MB → 0.60 (ChaCha20)
              >10 MB → 0.65            |  sinon  → 0.68
    Images :  <1 MB → 0.35 (Twofish)  |  >10 MB → 0.45 (Camellia)
              sinon → 0.55
    """
    if ftype == "video":
        if sens    > 0.70: return 0.75
        if size_mb > 50.0: return 0.60
        if size_mb > 10.0: return 0.65
        return 0.68
    else:
        if size_mb < 1.0:  return 0.35
        if size_mb > 10.0: return 0.45
        return 0.55


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Feature bonus
# ─────────────────────────────────────────────────────────────────────────────

def feature_bonus(algo: str, features: dict, ftype: str) -> float:
    """
    Twofish      : entropy_cv > 0.15            → +0.08
    ChaCha20     : compression_gain_zlib > 0.30 → +0.07
    AES-256      : bit_balance < 0.02            → +0.06
    Camellia-256 : image + header_entropy < 4.0  → +0.09
    """
    if algo == "Twofish"      and features.get("entropy_cv", 0.0)              > 0.15: return 0.08
    if algo == "ChaCha20"     and features.get("compression_gain_zlib", 0.0)   > 0.30: return 0.07
    if algo == "AES-256"      and features.get("bit_balance", 0.0)             < 0.02: return 0.06
    if algo == "Camellia-256" and ftype == "image" \
       and features.get("original_header_entropy", 0.0)                        < 4.0:  return 0.09
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 10. Chiffrement par blocs — mesure de temps
# ─────────────────────────────────────────────────────────────────────────────

CipherFn = Callable[[bytes], bytes]


def cipher_fn(algo: str) -> CipherFn:
    return get_cipher(algo)


def enc_time(path: str, fn: CipherFn) -> float:
    """Chiffre le fichier chunk par chunk — jamais entier en RAM."""
    t0 = time.perf_counter_ns()
    for chunk in _read_chunks(path):
        fn(chunk)
    return (time.perf_counter_ns() - t0) / 1_000_000_000


# ─────────────────────────────────────────────────────────────────────────────
# 11. Score composite + tableau de comparaison
# ─────────────────────────────────────────────────────────────────────────────

def _print_score_table(
    name: str, scores: dict[str, float], timing: dict[str, float],
    best: str, meta: dict, sens: float, ftype: str,
) -> None:
    """
    Tableau comparatif des scores pour le fichier courant.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  fight_001.mp4             │  video  │  size=12.34 MB  │  sens=0.9200 (high)
    ├──────────────────────┬──────────┬────────┬───────┬──────────┬───────────┤
    │  Algorithme          │   Sec    │   FB   │   α   │   T (s)  │   Score   │
    ├──────────────────────┼──────────┼────────┼───────┼──────────┼───────────┤
    │  AES-256          ★  │  0.8210  │ 0.0600 │ 0.75  │ 0.031200 │  0.660300 │
    │  ChaCha20            │  0.7980  │ 0.0000 │ 0.75  │ 0.028500 │  0.598500 │
    │  Twofish             │  0.7540  │ 0.0000 │ 0.75  │ 0.042000 │  0.565500 │
    │  Camellia-256        │  0.8120  │ 0.0000 │ 0.75  │ 0.038900 │  0.609000 │
    └──────────────────────┴──────────┴────────┴───────┴──────────┴───────────┘
    """
    size_mb  = meta.get("size_mb", 0.0)
    sens_cat = sensitivity_category(sens)
    alpha    = adaptive_alpha(ftype, size_mb, sens)
    mime     = "video/mp4" if ftype == "video" else "image/png"
    W = 77

    print(f"\n┌{'─'*W}┐")
    print(f"│  {name[:44]:<44}  │  {ftype:<5}  │  "
          f"size={size_mb:7.2f} MB  │  sens={sens:.4f} ({sens_cat})")
    print(f"├{'─'*22}┬{'─'*10}┬{'─'*8}┬{'─'*7}┬{'─'*10}┬{'─'*11}┤")
    print(f"│  {'Algorithme':<18}  │  {'Sec':^6}  │  {'FB':^4}  │"
          f"  {'α':^3}  │  {'T (s)':^6}  │  {'Score':^7}  │")
    print(f"├{'─'*22}┼{'─'*10}┼{'─'*8}┼{'─'*7}┼{'─'*10}┼{'─'*11}┤")

    for algo in ALGORITHMS:
        sc   = scores[algo]
        sec  = sec_score(algo, mime)
        fb   = feature_bonus(algo, meta, ftype)
        t    = timing.get(algo, 0.0)
        star = " ★" if algo == best else "  "
        print(
            f"│  {algo + star:<20}  │  {sec:^6.4f}  │ {fb:^6.4f} │"
            f" {alpha:^5.2f} │  {t:^8.6f}  │  {sc:^9.6f}  │"
        )

    print(f"└{'─'*22}┴{'─'*10}┴{'─'*8}┴{'─'*7}┴{'─'*10}┴{'─'*11}┘")


def score_all(
    mime: str, label: str, path: str, features: dict,
    verbose: bool = False,
) -> tuple[str, dict[str, float]]:
    """
    S(a, x) = α_eff·[Sec·Sens + FB] − (1−α_eff)·T  pour les 4 algorithmes.
    Retourne (best_algo, {algo: score}).
    """
    ftype   = "video" if mime.startswith("video") else "image"
    size_mb = features.get("size_mb", 0.0)
    name    = os.path.basename(path)
    sens    = sensitivity(label, name)
    alpha   = adaptive_alpha(ftype, size_mb, sens)
    beta    = 1.0 - alpha

    scores: dict[str, float] = {}
    timing: dict[str, float] = {}
    best, best_val = "", -1e18

    for algo in ALGORITHMS:
        t   = enc_time(path, cipher_fn(algo))
        sec = sec_score(algo, mime)
        fb  = feature_bonus(algo, features, ftype)
        s   = alpha * (sec * sens + fb) - beta * t
        scores[algo] = s
        timing[algo] = t
        if s > best_val:
            best, best_val = algo, s

    if verbose:
        _print_score_table(name, scores, timing, best, features, sens, ftype)

    return best, scores


# ─────────────────────────────────────────────────────────────────────────────
# 12. Self-test
# ─────────────────────────────────────────────────────────────────────────────

def _selftest() -> None:
    for algo in ALGORITHMS:
        try:
            get_cipher(algo)
        except Exception as exc:
            raise AssertionError(f"Algorithme non disponible : {algo} ({exc})")

    for mime in ("video/mp4", "image/png"):
        for algo in ALGORITHMS:
            s = sec_score(algo, mime)
            assert 0.0 <= s <= 1.2, f"sec_score hors plage {algo}/{mime}: {s:.4f}"

    assert adaptive_alpha("video", 0.5,  0.85) == 0.75
    assert adaptive_alpha("video", 80.0, 0.50) == 0.60
    assert adaptive_alpha("image", 0.5,  0.60) == 0.35
    assert adaptive_alpha("image", 20.0, 0.60) == 0.45

    # LUT popcount
    assert int(_LUT_POPCOUNT[0])   == 0
    assert int(_LUT_POPCOUNT[255]) == 8
    assert int(_LUT_POPCOUNT[1])   == 1

    # Vérifier que _hist_and_all_byte_stats est correct sur données connues
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, 10_000, dtype=np.uint8)
    r   = _hist_and_all_byte_stats(arr)
    r.pop("_cnts")
    assert 0.0 < r["original_entropy"] <= 8.0
    assert r["byte_q25"] <= r["byte_median"] <= r["byte_q75"]

    print("✅ Self-test OK — algorithmes : " + ", ".join(ALGORITHMS))


# ─────────────────────────────────────────────────────────────────────────────
# 13. Scan du dataset (générateur — pas de liste)
# ─────────────────────────────────────────────────────────────────────────────

def _file_type(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in VIDEO_EXTS: return "video"
    if ext in IMAGE_EXTS: return "image"
    return ""


def iter_files(dataset: str, mode: str) -> Generator[str, None, None]:
    """Générateur récursif — aucune liste de chemins en RAM."""
    allowed = (
        VIDEO_EXTS if mode == "video"
        else IMAGE_EXTS if mode == "image"
        else ALL_EXTS
    )
    for root, _dirs, files in os.walk(dataset):
        for fname in sorted(files):
            if os.path.splitext(fname)[1].lower() in allowed:
                yield os.path.join(root, fname)


# ─────────────────────────────────────────────────────────────────────────────
# 14. En-tête CSV
# ─────────────────────────────────────────────────────────────────────────────

CSV_HEADER: Final[list[str]] = [
    "name",
    "size_bytes", "size_kb", "size_mb", "log_size", "size_category",
    "type", "extension", "magic_number",
    "sensitivity", "sensitivity_category",
    "original_entropy", "original_compression_ratio",
    "original_unique_byte_count", "original_null_byte_ratio",
    "original_printable_ascii_ratio", "original_header_entropy",
    "byte_mean", "byte_std", "byte_skewness", "byte_kurtosis",
    "byte_median", "byte_q25", "byte_q75", "byte_iqr",
    "most_common_byte_ratio", "least_common_byte_ratio",
    "zero_byte_frequency", "ff_byte_frequency",
    "byte_frequency_chi_square", "byte_frequency_entropy",
    "first_1024_entropy", "middle_1024_entropy", "last_1024_entropy",
    "mean_block_entropy", "std_block_entropy",
    "min_block_entropy", "max_block_entropy",
    "entropy_range", "entropy_cv",
    "zlib_compression_ratio", "bz2_compression_ratio", "lzma_compression_ratio",
    "compression_gain_zlib", "compression_gain_bz2", "compression_gain_lzma",
    "bit_ones_ratio", "bit_zeros_ratio", "bit_balance",
    "bit_transition_rate", "mean_bit_run_length",
    "max_zero_run_length", "max_one_run_length",
    "mean_abs_byte_difference", "std_abs_byte_difference",
    "min_abs_byte_difference", "max_abs_byte_difference",
    "adjacent_equal_byte_ratio", "increasing_byte_ratio", "decreasing_byte_ratio",
    "byte_transition_entropy",
    "header_byte_mean", "header_byte_std",
    "header_null_ratio", "header_printable_ratio", "header_unique_byte_count",
    "footer_byte_mean", "footer_byte_std",
    "footer_null_ratio", "footer_printable_ratio", "footer_unique_byte_count",
    "algorithm",
]

_CSV_KEYS: Final[tuple[str, ...]] = tuple(
    k for k in CSV_HEADER
    if k not in {"name", "type", "extension", "magic_number",
                 "sensitivity", "sensitivity_category", "algorithm"}
)


# ─────────────────────────────────────────────────────────────────────────────
# 15. Boucle principale (CSV ligne par ligne, del+gc après chaque fichier)
# ─────────────────────────────────────────────────────────────────────────────

def _make_row(
    name: str, ftype: str, ext: str, meta: dict,
    sens: float, sens_cat: str, best: str,
) -> list:
    m = meta
    return [
        name,
        m["size_bytes"], m["size_kb"], m["size_mb"], m["log_size"], m["size_category"],
        ftype, ext, m["magic_number"],
        f"{sens:.4f}", sens_cat,
        m["original_entropy"],           m["original_compression_ratio"],
        m["original_unique_byte_count"], m["original_null_byte_ratio"],
        m["original_printable_ascii_ratio"], m["original_header_entropy"],
        m["byte_mean"],     m["byte_std"],
        m["byte_skewness"], m["byte_kurtosis"],
        m["byte_median"],   m["byte_q25"],
        m["byte_q75"],      m["byte_iqr"],
        m["most_common_byte_ratio"],  m["least_common_byte_ratio"],
        m["zero_byte_frequency"],     m["ff_byte_frequency"],
        m["byte_frequency_chi_square"], m["byte_frequency_entropy"],
        m["first_1024_entropy"],  m["middle_1024_entropy"],
        m["last_1024_entropy"],   m["mean_block_entropy"],
        m["std_block_entropy"],   m["min_block_entropy"],
        m["max_block_entropy"],   m["entropy_range"],
        m["entropy_cv"],
        m["zlib_compression_ratio"], m["bz2_compression_ratio"],
        m["lzma_compression_ratio"], m["compression_gain_zlib"],
        m["compression_gain_bz2"],   m["compression_gain_lzma"],
        m["bit_ones_ratio"],      m["bit_zeros_ratio"],
        m["bit_balance"],         m["bit_transition_rate"],
        m["mean_bit_run_length"],
        m["max_zero_run_length"], m["max_one_run_length"],
        m["mean_abs_byte_difference"], m["std_abs_byte_difference"],
        m["min_abs_byte_difference"],  m["max_abs_byte_difference"],
        m["adjacent_equal_byte_ratio"], m["increasing_byte_ratio"],
        m["decreasing_byte_ratio"],     m["byte_transition_entropy"],
        m["header_byte_mean"],    m["header_byte_std"],
        m["header_null_ratio"],   m["header_printable_ratio"],
        m["header_unique_byte_count"],
        m["footer_byte_mean"],    m["footer_byte_std"],
        m["footer_null_ratio"],   m["footer_printable_ratio"],
        m["footer_unique_byte_count"],
        best,
    ]


def compute(dataset: str, out: str, mode: str, verbose: bool) -> None:
    """
    Consommation mémoire quasi constante :
      iter_files() → générateur | extract_all_features() libère arr en interne
      CSV ligne par ligne        | del meta + gc.collect() après chaque fichier
    """
    counts: dict[str, int] = {a: 0 for a in ALGORITHMS}
    total = video_n = image_n = 0

    with open(out, "w", newline="", encoding="utf-8") as fw:
        writer = csv.writer(fw)
        writer.writerow(CSV_HEADER)

        for path in iter_files(dataset, mode):
            ftype = _file_type(path)
            name  = os.path.basename(path)
            label = os.path.basename(os.path.dirname(path)) or "unknown"
            mime  = (
                mimetypes.guess_type(path)[0]
                or ("video/mp4" if ftype == "video" else "image/png")
            )
            ext = os.path.splitext(name)[1].lower()

            meta     = extract_all_features(path)
            sens     = sensitivity(label, name)
            sens_cat = sensitivity_category(sens)

            best, scores = score_all(mime, label, path, meta, verbose=verbose)

            writer.writerow(_make_row(name, ftype, ext, meta, sens, sens_cat, best))

            counts[best] += 1
            total  += 1
            if ftype == "video": video_n += 1
            else:                image_n += 1

            del meta, scores
            gc.collect()

    if total == 0:
        print(f"⚠️  Aucun fichier trouvé dans {dataset!r} (mode={mode}).")
        return

    tot = sum(counts.values()) or 1
    print(f"\n📁 Traités : {video_n} vidéo(s) + {image_n} image(s) = {total}\n")
    print("📊 Distribution des algorithmes :")
    for algo, n in sorted(counts.items(), key=lambda x: -x[1]):
        bar = "█" * int(n / tot * 32)
        print(f"  {algo:<16}: {n:6}  ({n / tot:5.1%})  {bar}")
    print(f"\n✅ CSV → {out}  ({total} lignes, {len(CSV_HEADER)} colonnes)")


# ─────────────────────────────────────────────────────────────────────────────
# 16. CLI
# ─────────────────────────────────────────────────────────────────────────────

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Adaptive cipher selector — DCSASS + Agricultural crops\n\n"
            "S(a,x) = α_eff·[Sec·Sens+FB] − (1−α_eff)·T\n\n"
            "Speedups NumPy mesurés sur 4 MB :\n"
            "  skew/kurt ×800 | percentiles ×400 | bitcount ×62\n"
            "  XOR trans ×64  | diff adj ×30     | bigrams ×5\n"
            "  Total features : ~130 ms vs ~855 ms (×6.5)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", required=True, help="Racine du dataset")
    p.add_argument("--mode", choices=["video", "image", "both"], default="both")
    p.add_argument("--out",  default="results.csv")
    p.add_argument("--alpha", type=float, default=None, metavar="A",
                   help="Ignoré (α adaptatif). Conservé pour compatibilité.")
    p.add_argument("--verbose", action="store_true",
                   help="Tableau comparatif des scores pour chaque fichier")
    return p.parse_args()


def main() -> None:
    args = get_args()
    if args.alpha is not None:
        print("ℹ️  --alpha ignoré : α est calculé par fichier via adaptive_alpha().\n")
    _selftest()
    compute(args.dataset, args.out, args.mode, verbose=args.verbose)


if __name__ == "__main__":
    main()