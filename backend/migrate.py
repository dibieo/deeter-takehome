"""
migrate.py — Migrate pre-trained notebook models to the web app format.

EN→FR (original notebook weights, input names "english"/"french"):
  1. Rebuilds each architecture with the correct input names ("source"/"target").
  2. Transfers weights from the notebook checkpoint.
  3. Saves as rnn_en_fr.keras / transformer_en_fr.keras.
  4. Builds and saves EN→FR vocabularies (vocab_en_src.json, vocab_fr_tgt.json).

FR→EN (weights trained in Colab, already use "source"/"target"):
  No weight migration needed — models are already in the correct format.
  5. Builds and saves FR→EN vocabularies (vocab_fr_src.json, vocab_en_tgt.json)
     if the JSON files are not already present (e.g. downloaded from Drive).
  6. Reports which FR→EN model files are present / missing.

Expected files in models/ after running the Colab notebook and downloading:
    rnn_fr_en.keras
    transformer_fr_en.keras
    vocab_fr_src.json   (optional — rebuilt from fra.txt if absent)
    vocab_en_tgt.json   (optional — rebuilt from fra.txt if absent)

Usage:
    python -m backend.migrate
"""

import json
import re
import string
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

NOTEBOOK_RNN = MODELS_DIR / "notebook_rnn_en_fr.keras"
NOTEBOOK_TFM = MODELS_DIR / "notebook_transformer_en_fr.keras"

VOCAB_SIZE = 20_000
SEQ_LEN = 30
BATCH_SIZE = 64

DATA_URL = "http://storage.googleapis.com/download.tensorflow.org/data/fra-eng.zip"
ZIP_PATH = DATA_DIR / "fra-eng.zip"
TXT_PATH = DATA_DIR / "fra.txt"

_STRIP = string.punctuation.replace("[", "").replace("]", "")
_STRIP_RE = re.escape(_STRIP)


def _custom_standardize(x):
    import tensorflow as tf

    return tf.strings.regex_replace(tf.strings.lower(x), f"[{_STRIP_RE}]", "")


# ── Data / vocab helpers ──────────────────────────────────────────────────────

def ensure_data():
    DATA_DIR.mkdir(exist_ok=True)
    if TXT_PATH.exists():
        return
    print("[data] Downloading fra-eng.zip …")
    urllib.request.urlretrieve(DATA_URL, ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(DATA_DIR)


def build_en_fr_tokenizers():
    from keras import layers

    train_pairs = _load_train_pairs()

    src_tok = layers.TextVectorization(
        max_tokens=VOCAB_SIZE, output_mode="int",
        output_sequence_length=SEQ_LEN,
    )
    tgt_tok = layers.TextVectorization(
        max_tokens=VOCAB_SIZE, output_mode="int",
        output_sequence_length=SEQ_LEN + 1,
        standardize=_custom_standardize,
    )
    src_tok.adapt([p[0] for p in train_pairs])
    tgt_tok.adapt(["[start] " + p[1] + " [end]" for p in train_pairs])
    return src_tok, tgt_tok


def build_fr_en_tokenizers():
    from keras import layers

    train_pairs = _load_train_pairs()

    src_tok = layers.TextVectorization(
        max_tokens=VOCAB_SIZE, output_mode="int",
        output_sequence_length=SEQ_LEN,
        standardize=_custom_standardize,
    )
    tgt_tok = layers.TextVectorization(
        max_tokens=VOCAB_SIZE, output_mode="int",
        output_sequence_length=SEQ_LEN + 1,
        standardize=_custom_standardize,
    )
    src_tok.adapt([p[1] for p in train_pairs])
    tgt_tok.adapt(["[start] " + p[0] + " [end]" for p in train_pairs])
    return src_tok, tgt_tok


def _load_train_pairs():
    """Parse fra.txt and return the same deterministic train split used everywhere."""
    import random

    with open(TXT_PATH, encoding="utf-8") as f:
        lines = f.read().split("\n")[:-1]

    pairs = []
    for line in lines:
        parts = line.split("\t")
        if len(parts) >= 2:
            pairs.append((parts[0].strip(), parts[1].strip()))

    random.seed(42)
    random.shuffle(pairs)
    val_n = int(0.15 * len(pairs))
    train_n = len(pairs) - 2 * val_n
    return pairs[:train_n]


def save_vocab(tok, path: Path):
    vocab = tok.get_vocabulary()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    print(f"[vocab] Saved → {path.name}  ({len(vocab)} tokens)")


# ── Weight migration ──────────────────────────────────────────────────────────

def migrate_rnn():
    import keras
    from keras import layers

    out_path = MODELS_DIR / "rnn_en_fr.keras"
    if out_path.exists():
        print(f"[migrate] {out_path.name} already exists — skipping.")
        return

    if not NOTEBOOK_RNN.exists():
        print(f"[migrate] {NOTEBOOK_RNN.name} not found — skipping RNN migration.")
        return

    print(f"[migrate] Loading notebook RNN weights from {NOTEBOOK_RNN.name} …")
    old_model = keras.models.load_model(str(NOTEBOOK_RNN))

    # Rebuild identical architecture with "source"/"target" input names
    source = keras.Input(shape=(None,), dtype="int32", name="source")
    x = layers.Embedding(VOCAB_SIZE, 256, mask_zero=True)(source)
    enc_out = layers.Bidirectional(layers.GRU(1024), merge_mode="sum")(x)

    target = keras.Input(shape=(None,), dtype="int32", name="target")
    x = layers.Embedding(VOCAB_SIZE, 256, mask_zero=True)(target)
    x = layers.GRU(1024, return_sequences=True)(x, initial_state=enc_out)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(VOCAB_SIZE, activation="softmax")(x)
    new_model = keras.Model([source, target], out)

    # Build new model to initialise weights
    import numpy as np
    dummy_src = np.zeros((1, SEQ_LEN), dtype="int32")
    dummy_tgt = np.zeros((1, SEQ_LEN), dtype="int32")
    new_model({"source": dummy_src, "target": dummy_tgt})

    # Transfer weights (architectures are identical, just different input names)
    new_model.set_weights(old_model.get_weights())
    new_model.save(str(out_path))
    print(f"[migrate] Saved → {out_path.name}")


def migrate_transformer():
    import keras
    from backend.model_definitions import (
        CUSTOM_OBJECTS,
        PositionalEmbedding,
        TransformerDecoder,
        TransformerEncoder,
    )

    out_path = MODELS_DIR / "transformer_en_fr.keras"
    if out_path.exists():
        print(f"[migrate] {out_path.name} already exists — skipping.")
        return

    if not NOTEBOOK_TFM.exists():
        print(f"[migrate] {NOTEBOOK_TFM.name} not found — skipping Transformer migration.")
        return

    print(f"[migrate] Loading notebook Transformer weights from {NOTEBOOK_TFM.name} …")

    # The notebook's Transformer layers lack get_config(), so we define shims
    # that allow Keras to instantiate them from the saved config.
    class _EncShim(TransformerEncoder):
        @classmethod
        def from_config(cls, cfg):
            cfg.setdefault("hidden_dim", 256)
            cfg.setdefault("intermediate_dim", 2056)
            cfg.setdefault("num_heads", 8)
            return cls(**{k: v for k, v in cfg.items() if k in ("hidden_dim", "intermediate_dim", "num_heads", "name", "dtype", "trainable")})

    class _DecShim(TransformerDecoder):
        @classmethod
        def from_config(cls, cfg):
            cfg.setdefault("hidden_dim", 256)
            cfg.setdefault("intermediate_dim", 2056)
            cfg.setdefault("num_heads", 8)
            return cls(**{k: v for k, v in cfg.items() if k in ("hidden_dim", "intermediate_dim", "num_heads", "name", "dtype", "trainable")})

    class _PosShim(PositionalEmbedding):
        @classmethod
        def from_config(cls, cfg):
            cfg.setdefault("sequence_length", SEQ_LEN)
            cfg.setdefault("vocab_size", VOCAB_SIZE)
            cfg.setdefault("embed_dim", 256)
            return cls(**{k: v for k, v in cfg.items() if k in ("sequence_length", "vocab_size", "embed_dim", "name", "dtype", "trainable")})

    shim_objects = {
        "TransformerEncoder": _EncShim,
        "TransformerDecoder": _DecShim,
        "PositionalEmbedding": _PosShim,
    }

    try:
        old_model = keras.models.load_model(str(NOTEBOOK_TFM), custom_objects=shim_objects)
    except Exception as exc:
        print(f"[migrate] Could not load notebook Transformer ({exc}).")
        print("[migrate] Run `python -m backend.train --dir en_fr` to train from scratch.")
        return

    # Rebuild with "source"/"target" inputs and proper custom layers
    import numpy as np
    from keras import layers

    source = keras.Input(shape=(None,), dtype="int32", name="source")
    x = PositionalEmbedding(SEQ_LEN, VOCAB_SIZE, 256)(source)
    enc_out = TransformerEncoder(256, 2056, 8)(source=x, source_mask=source != 0)

    target = keras.Input(shape=(None,), dtype="int32", name="target")
    x = PositionalEmbedding(SEQ_LEN, VOCAB_SIZE, 256)(target)
    x = TransformerDecoder(256, 2056, 8)(target=x, source=enc_out, source_mask=source != 0)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(VOCAB_SIZE, activation="softmax")(x)
    new_model = keras.Model([source, target], out)

    dummy_src = np.zeros((1, SEQ_LEN), dtype="int32")
    dummy_tgt = np.zeros((1, SEQ_LEN - 1), dtype="int32")
    new_model({"source": dummy_src, "target": dummy_tgt})

    new_model.set_weights(old_model.get_weights())
    new_model.save(str(out_path))
    print(f"[migrate] Saved → {out_path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    MODELS_DIR.mkdir(exist_ok=True)

    # ── Data (shared by both directions) ──────────────────────────────────────
    ensure_data()

    # Save config once
    cfg = {"vocab_size": VOCAB_SIZE, "seq_len": SEQ_LEN, "batch_size": BATCH_SIZE}
    with open(MODELS_DIR / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # ── Step 1: EN→FR vocabularies ────────────────────────────────────────────
    print("── Step 1: EN→FR vocabularies ───────────────────────────────────")
    src_tok, tgt_tok = build_en_fr_tokenizers()
    save_vocab(src_tok, MODELS_DIR / "vocab_en_src.json")
    save_vocab(tgt_tok, MODELS_DIR / "vocab_fr_tgt.json")

    # ── Step 2: Migrate EN→FR notebook weights ────────────────────────────────
    print("\n── Step 2: EN→FR model migration ────────────────────────────────")
    migrate_rnn()
    migrate_transformer()

    # ── Step 3: FR→EN vocabularies ────────────────────────────────────────────
    print("\n── Step 3: FR→EN vocabularies ───────────────────────────────────")
    fr_src_path = MODELS_DIR / "vocab_fr_src.json"
    en_tgt_path = MODELS_DIR / "vocab_en_tgt.json"

    if fr_src_path.exists() and en_tgt_path.exists():
        print("[vocab] vocab_fr_src.json and vocab_en_tgt.json already present — skipping rebuild.")
    else:
        print("[vocab] Building FR→EN vocabularies from fra.txt …")
        fr_src_tok, en_tgt_tok = build_fr_en_tokenizers()
        if not fr_src_path.exists():
            save_vocab(fr_src_tok, fr_src_path)
        if not en_tgt_path.exists():
            save_vocab(en_tgt_tok, en_tgt_path)

    # ── Step 4: FR→EN Colab model status ─────────────────────────────────────
    print("\n── Step 4: FR→EN Colab model status ─────────────────────────────")
    fr_en_models = {
        "rnn_fr_en.keras":         MODELS_DIR / "rnn_fr_en.keras",
        "transformer_fr_en.keras": MODELS_DIR / "transformer_fr_en.keras",
    }
    all_present = True
    for name, path in fr_en_models.items():
        if path.exists():
            size_mb = path.stat().st_size / 1e6
            print(f"[model] ✓  {name}  ({size_mb:.0f} MB)")
        else:
            print(f"[model] ✗  {name}  — not found")
            all_present = False

    print()
    if all_present:
        print("All artefacts ready. Start the web app with:")
        print("  uvicorn backend.app:app --reload")
    else:
        print("FR→EN model files are missing.  Copy them from Google Drive:")
        print("  Drive/MyDrive/translation_models/rnn_fr_en.keras")
        print("  Drive/MyDrive/translation_models/transformer_fr_en.keras")
        print("into your local models/ directory, then re-run this script.")


if __name__ == "__main__":
    main()
