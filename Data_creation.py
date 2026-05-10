#!/usr/bin/env python3
"""
agri_impl.py — Adaptive cipher selector pour la classification d'images et de
vidéos de cultures agricoles
Théorie : a* = argmax_a S(a, x)
Usage  : python agri_impl.py --dataset PATH [--alpha A] [--beta B]

Dépendance : pip install pycryptodome
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import mimetypes
import os
import time
import hashlib
import struct
from functools import cache
from typing import Callable, Final, Iterable, Sequence

# ───────────────────────── 1. Crypto back-ends (pycryptodome) ───────────────
try:
    from Crypto.Cipher import AES, ChaCha20
    from Crypto.Util.Padding import pad
    _HAVE_PYCRYPTODOME = True
except ImportError:
    _HAVE_PYCRYPTODOME = False
    print("⚠️  pycryptodome non trouvé — pip install pycryptodome")

MISSING: list[str] = []

# ───────────────────────── 2. Paramètres globaux ────────────────────────────
CURRENT_YEAR: Final[int] = _dt.datetime.now(_dt.timezone.utc).year
CHUNK = 8 * 1024 * 1024          # 8 MiB
MAX_LABEL_LEN: Final[int] = 20

OMEGA_KS, OMEGA_AR, OMEGA_AP, OMEGA_CA = 0.40, 0.35, 0.05, 0.20
ALPHA_DEF, BETA_DEF = 0.8, 0.2

# ───────────────────────── 3. Tables statiques ──────────────────────────────
ALGORITHMS: Final[Sequence[str]] = (
    "AES-256",
    "ChaCha20",
    "Twofish",
    "Camellia-256",
)

KEY_BITS = {
    "AES-256": 256,
    "ChaCha20": 256,
    "Twofish": 256,
    "Camellia-256": 256,
}

FIRST_YEAR = {
    "AES-256": 2001,
    "ChaCha20": 2008,
    "Twofish": 1998,
    "Camellia-256": 2003,
}

CONTENT_ADAPT = {
    "image": {
        "AES-256": 0.80,
        "ChaCha20": 0.80,
        "Twofish": 0.70,
        "Camellia-256": 0.70,
    },
    "video": {
        "AES-256": 0.80,
        "ChaCha20": 0.85,
        "Twofish": 0.70,
        "Camellia-256": 0.70,
    },
}

KNOWN_VULN = {
    "AES-256": 0.02,
    "ChaCha20": 0.04,
    "Twofish": 0.05,
    "Camellia-256": 0.02,
}

ATTACK_EFF = {
    "AES-256": 0.01,
    "ChaCha20": 0.03,
    "Twofish": 0.04,
    "Camellia-256": 0.01,
}

SUSPICIOUS = {
    "abuse", "arrest", "arson", "assault", "burglary", "explosion",
    "fighting", "robbery", "shooting", "shoplifting", "vandalism",
    "stealing", "accident",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".mpeg", ".mpg"}
MEDIA_EXTS = IMAGE_EXTS | VIDEO_EXTS


def media_type_of(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    return "unknown"


# ───────────────────────── 4. Implémentation des chiffrements ───────────────
def _xor_fallback(key_seed: int) -> Callable[[bytes], bytes]:
    """Fallback déterministe si pycryptodome absent (benchmarking uniquement)."""
    key = hashlib.sha256(struct.pack(">Q", key_seed)).digest()

    def _fn(data: bytes) -> bytes:
        mask = (key * (len(data) // 32 + 1))[:len(data)]
        return bytes(a ^ b for a, b in zip(data, mask))

    return _fn


def _make_aes(key_bytes: int) -> Callable[[bytes], bytes]:
    if _HAVE_PYCRYPTODOME:
        from Crypto.Cipher import AES as _AES
        from Crypto.Util.Padding import pad as _pad

        key = os.urandom(key_bytes)
        iv = os.urandom(16)

        def _fn(data: bytes) -> bytes:
            return _AES.new(key, _AES.MODE_CBC, iv).encrypt(_pad(data, 16))

        return _fn
    return _xor_fallback(key_bytes * 8)


def _make_chacha20() -> Callable[[bytes], bytes]:
    if _HAVE_PYCRYPTODOME:
        from Crypto.Cipher import ChaCha20 as _C

        key = os.urandom(32)
        nonce = os.urandom(12)

        def _fn(data: bytes) -> bytes:
            return _C.new(key=key, nonce=nonce).encrypt(data)

        return _fn
    return _xor_fallback(256)


# Twofish et Camellia absents de pycryptodome → proxy AES (même taille de clé)
def _make_twofish() -> Callable[[bytes], bytes]:
    return _make_aes(16)


def _make_camellia() -> Callable[[bytes], bytes]:
    return _make_aes(32)


_CIPHER_BUILDERS: dict[str, Callable[[], Callable[[bytes], bytes]]] = {
    "AES-256": lambda: _make_aes(32),
    "ChaCha20": _make_chacha20,
    "Twofish": _make_twofish,
    "Camellia-256": _make_camellia,
}


def get_cipher(algo: str) -> Callable[[bytes], bytes]:
    builder = _CIPHER_BUILDERS.get(algo)
    if builder is None:
        raise ValueError(f"Algorithme inconnu : {algo}")
    return builder()


# ───────────────────────── 5. Sécurité & sensibilité ────────────────────────
@cache
def key_strength(algo: str) -> float:
    bits = KEY_BITS[algo]
    return 1.0 if bits >= 256 else 0.85 if bits >= 192 else 0.70 if bits >= 128 else 0.40


def attack_resilience(algo: str) -> float:
    v, a = KNOWN_VULN[algo], ATTACK_EFF[algo]
    p = min(1.0, (CURRENT_YEAR - FIRST_YEAR[algo]) / 50.0)
    return 1.0 - (0.4 * v + 0.4 * a + 0.2 * p)


def age_penalty(algo: str) -> float:
    return min(1.0, (CURRENT_YEAR - FIRST_YEAR[algo]) / 50.0)


def content_adaptability(algo: str, mime: str) -> float:
    top_level = mime.split("/")[0]
    return CONTENT_ADAPT.get(top_level, {}).get(algo, 0.6)


def sec_score(algo: str, mime: str) -> float:
    return (
        OMEGA_KS * key_strength(algo)
        + OMEGA_AR * attack_resilience(algo)
        - OMEGA_AP * age_penalty(algo)
        + OMEGA_CA * content_adaptability(algo, mime)
    )


def sensitivity(label: str, name: str = "") -> float:
    label = label.strip()
    length_ratio = min(len(label), MAX_LABEL_LEN) / MAX_LABEL_LEN
    base = length_ratio

    if name:
        h = int.from_bytes(hashlib.sha256(name.encode()).digest(), "big")
        base += (h % 1000) / 10000

    if label.lower() in SUSPICIOUS:
        base = min(1.0, base + 0.2)

    return round(min(1.0, base), 3)


# ───────────────────────── 6. Chiffrement & timing ──────────────────────────
CipherFn = Callable[[bytes], bytes]


def enc_time(path: str, fn: CipherFn) -> float:
    """Retourne le temps de chiffrement en millisecondes."""
    t0 = time.perf_counter()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            fn(chunk)
    return (time.perf_counter() - t0) * 1000.0  # secondes → millisecondes


# ───────────────────────── 7. Score composite ───────────────────────────────
def score(algo: str, mime: str, label: str, path: str,
          α: float, β: float, dbg: bool = False) -> float:
    name = os.path.basename(path)
    t = enc_time(path, get_cipher(algo))
    s = α * (β * sec_score(algo, mime) +( 1-β) sensitivity(label, name)) - (1-α) * t

    if dbg:
        print(
            f"{algo:<22} Sec={sec_score(algo, mime):.3f} "
            f"Sens={sensitivity(label, name):.3f} Time={t:.4f}ms → Score={s:.3f}"
        )

    return s


# ───────────────────────── 8. Self-test ─────────────────────────────────────
def _selftest() -> None:
    present = [a for a in ALGORITHMS if a not in MISSING]
    assert {"AES-256", "ChaCha20"}.issubset(present), "AES-256/ChaCha20 absents"

    for a in present:
        for mime in ("image/png", "video/mp4"):
            sec = sec_score(a, mime)
            assert 0.0 <= sec <= 1.2, (a, mime, sec)

        fn = get_cipher(a)
        fn(b"test" * 64)

    print("✅ Self-test : algorithmes =", ", ".join(present))
    print("✅ Self-test : MIME testés = image/png, video/mp4")

    if _HAVE_PYCRYPTODOME:
        print("✅ pycryptodome actif — chiffrements natifs")
    else:
        print("⚠️  Fallback XOR actif (pip install pycryptodome)")


# ───────────────────────── 9. Dataset scan ──────────────────────────────────
def iter_media(dataset: str) -> Iterable[str]:
    for root, _dirs, files in os.walk(dataset):
        for f in files:
            if os.path.splitext(f)[1].lower() in MEDIA_EXTS:
                yield os.path.join(root, f)


def compute(dataset: str, out: str, α: float, β: float) -> None:
    rows: list[list] = []
    counts = {a: 0 for a in ALGORITHMS if a not in MISSING}
    type_counts: dict[str, dict[str, int]] = {
        "image": {a: 0 for a in counts},
        "video": {a: 0 for a in counts},
    }

    first = True

    for path in iter_media(dataset):
        name = os.path.basename(path)
        size = os.path.getsize(path)
        ext = os.path.splitext(name)[1].lower()
        label = os.path.basename(os.path.dirname(path)) or "unknown"
        mime = mimetypes.guess_type(path)[0] or f"{media_type_of(path)}/unknown"
        mtype = media_type_of(path)
        sense = sensitivity(label, name)

        if first:
            print(f"\n🔎 Debug pour {name} — label={label}, MIME={mime}, type={mtype}")

        best, best_val = None, -1e9
        for a in counts:
            sc = score(a, mime, label, path, α, β, dbg=first)
            if sc > best_val:
                best, best_val = a, sc

        first = False
        counts[best] += 1

        if mtype in type_counts:
            type_counts[mtype][best] += 1

        rows.append([name, size, mtype, ext, f"{sense:.3f}", best])

    with open(out, "w", newline="", encoding="utf-8") as fw:
        csv.writer(fw).writerows(
            [["name", "size", "type", "extension", "sensitivity", "algorithm"], *rows]
        )

    tot = sum(counts.values()) or 1
    print("\n📊 Distribution globale :")
    for a, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {a:<22}: {n:6} fichiers ({n/tot:5.1%})")

    for mtype, tcounts in type_counts.items():
        subtot = sum(tcounts.values()) or 1
        if subtot == 0:
            continue

        print(f"\n📊 Distribution — {mtype}s :")
        for a, n in sorted(tcounts.items(), key=lambda x: -x[1]):
            print(f"  {a:<22}: {n:6} ({n/subtot:5.1%})")

    print(f"\n✅ Résultats écrits → {out}")


# ───────────────────────── 10. CLI ──────────────────────────────────────────
def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Adaptive cipher selector — images & vidéos")
    default_ds = os.path.dirname(__file__)
    p.add_argument("--dataset", default=default_ds)
    p.add_argument("--out", default="data0208_best.csv")
    p.add_argument("--alpha", type=float, default=ALPHA_DEF)
    p.add_argument("--beta", type=float, default=BETA_DEF)
    return p.parse_args()


def main() -> None:
    args = get_args()
    _selftest()
    compute(args.dataset, args.out, args.alpha, args.beta)


if __name__ == "__main__":
    main()