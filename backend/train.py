"""
train.py — Train English↔French machine translation models.

Produces six artefacts in models/:
  rnn_en_fr.keras          GRU seq2seq,    English → French
  rnn_fr_en.keras          GRU seq2seq,    French  → English
  transformer_en_fr.keras  Transformer,    English → French
  transformer_fr_en.keras  Transformer,    French  → English
  vocab_en_src.json        English source  vocabulary
  vocab_fr_tgt.json        French  target  vocabulary  (contains [start]/[end])
  vocab_fr_src.json        French  source  vocabulary
  vocab_en_tgt.json        English target  vocabulary  (contains [start]/[end])
  config.json              Shared hyper-parameters

Usage:
    python -m backend.train              # train everything
    python -m backend.train --dir en_fr  # train only EN→FR models
    python -m backend.train --dir fr_en  # train only FR→EN models
"""

import argparse
import json
import random
import re
import string
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

DATA_URL = "http://storage.googleapis.com/download.tensorflow.org/data/fra-eng.zip"
ZIP_PATH = DATA_DIR / "fra-eng.zip"
TXT_PATH = DATA_DIR / "fra.txt"

# ── Hyper-parameters ──────────────────────────────────────────────────────────
VOCAB_SIZE = 20_000
SEQ_LEN = 30
BATCH_SIZE = 64
RANDOM_SEED = 42

# RNN
RNN_EMBED_DIM = 256
RNN_HIDDEN_DIM = 1024
RNN_EPOCHS = 15

# Transformer
TFM_HIDDEN_DIM = 256
TFM_INTERMEDIATE_DIM = 2048
TFM_NUM_HEADS = 8
TFM_EPOCHS = 30

CONFIG = {
    "vocab_size": VOCAB_SIZE,
    "seq_len": SEQ_LEN,
    "batch_size": BATCH_SIZE,
}

# ── Text standardization ──────────────────────────────────────────────────────
# Keep [ and ] so that [start] / [end] sentinel tokens survive.
_STRIP = string.punctuation.replace("[", "").replace("]", "")
_STRIP_RE = re.escape(_STRIP)


def _custom_standardize(input_string):
    import tensorflow as tf

    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, f"[{_STRIP_RE}]", "")


# ── Data helpers ──────────────────────────────────────────────────────────────

def download_data() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    if TXT_PATH.exists():
        print("[data] fra.txt already present.")
        return
    print("[data] Downloading fra-eng.zip …")
    urllib.request.urlretrieve(DATA_URL, ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(DATA_DIR)
    if not TXT_PATH.exists():
        raise FileNotFoundError("fra.txt not found after extraction.")
    print("[data] Download complete.")


def load_raw_pairs() -> list[tuple[str, str]]:
    with open(TXT_PATH, encoding="utf-8") as f:
        lines = f.read().split("\n")[:-1]
    pairs = []
    for line in lines:
        parts = line.split("\t")
        if len(parts) >= 2:
            pairs.append((parts[0].strip(), parts[1].strip()))
    return pairs


def split_pairs(pairs, val_frac=0.15):
    random.seed(RANDOM_SEED)
    random.shuffle(pairs)
    val_n = int(val_frac * len(pairs))
    train_n = len(pairs) - 2 * val_n
    return pairs[:train_n], pairs[train_n : train_n + val_n], pairs[train_n + val_n :]


# ── Tokenizer helpers ─────────────────────────────────────────────────────────

def _make_tokenizer(max_tokens: int, seq_len: int):
    from keras import layers

    return layers.TextVectorization(
        max_tokens=max_tokens,
        output_mode="int",
        output_sequence_length=seq_len,
        standardize=_custom_standardize,
    )


def build_en_fr_tokenizers(train_pairs):
    """Build and adapt tokenizers for the EN→FR direction."""
    src_tok = _make_tokenizer(VOCAB_SIZE, SEQ_LEN)
    tgt_tok = _make_tokenizer(VOCAB_SIZE, SEQ_LEN + 1)

    src_tok.adapt([p[0] for p in train_pairs])
    tgt_tok.adapt(["[start] " + p[1] + " [end]" for p in train_pairs])
    return src_tok, tgt_tok


def build_fr_en_tokenizers(train_pairs):
    """Build and adapt tokenizers for the FR→EN direction."""
    src_tok = _make_tokenizer(VOCAB_SIZE, SEQ_LEN)
    tgt_tok = _make_tokenizer(VOCAB_SIZE, SEQ_LEN + 1)

    src_tok.adapt([p[1] for p in train_pairs])
    tgt_tok.adapt(["[start] " + p[0] + " [end]" for p in train_pairs])
    return src_tok, tgt_tok


def save_vocab(tok, path: Path) -> None:
    vocab = tok.get_vocabulary()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    print(f"[vocab] Saved {len(vocab)} tokens → {path.name}")


# ── Dataset helpers ───────────────────────────────────────────────────────────

def _make_tf_dataset(src_texts, tgt_texts, src_tok, tgt_tok):
    import tensorflow as tf

    def format_batch(src, tgt):
        src_enc = src_tok(src)
        tgt_enc = tgt_tok(tgt)
        features = {"source": src_enc, "target": tgt_enc[:, :-1]}
        labels = tgt_enc[:, 1:]
        weights = tf.cast(labels != 0, tf.float32)
        return features, labels, weights

    ds = tf.data.Dataset.from_tensor_slices((src_texts, tgt_texts))
    return (
        ds.batch(BATCH_SIZE)
        .map(format_batch, num_parallel_calls=4)
        .shuffle(2048)
        .cache()
    )


def make_en_fr_datasets(train_pairs, val_pairs, src_tok, tgt_tok):
    def _prep(pairs):
        return (
            [p[0] for p in pairs],
            ["[start] " + p[1] + " [end]" for p in pairs],
        )

    train_ds = _make_tf_dataset(*_prep(train_pairs), src_tok, tgt_tok)
    val_ds = _make_tf_dataset(*_prep(val_pairs), src_tok, tgt_tok)
    return train_ds, val_ds


def make_fr_en_datasets(train_pairs, val_pairs, src_tok, tgt_tok):
    def _prep(pairs):
        return (
            [p[1] for p in pairs],
            ["[start] " + p[0] + " [end]" for p in pairs],
        )

    train_ds = _make_tf_dataset(*_prep(train_pairs), src_tok, tgt_tok)
    val_ds = _make_tf_dataset(*_prep(val_pairs), src_tok, tgt_tok)
    return train_ds, val_ds


# ── Model builders ────────────────────────────────────────────────────────────

def build_rnn_model(src_vocab_size: int, tgt_vocab_size: int):
    import keras
    from keras import layers

    source = keras.Input(shape=(None,), dtype="int32", name="source")
    x = layers.Embedding(src_vocab_size, RNN_EMBED_DIM, mask_zero=True)(source)
    encoder_out = layers.Bidirectional(
        layers.GRU(RNN_HIDDEN_DIM), merge_mode="sum"
    )(x)

    target = keras.Input(shape=(None,), dtype="int32", name="target")
    x = layers.Embedding(tgt_vocab_size, RNN_EMBED_DIM, mask_zero=True)(target)
    x = layers.GRU(RNN_HIDDEN_DIM, return_sequences=True)(x, initial_state=encoder_out)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(tgt_vocab_size, activation="softmax")(x)
    return keras.Model([source, target], out)


def build_transformer_model(src_vocab_size: int, tgt_vocab_size: int):
    import keras
    from keras import layers
    from backend.model_definitions import (
        PositionalEmbedding,
        TransformerDecoder,
        TransformerEncoder,
    )

    source = keras.Input(shape=(None,), dtype="int32", name="source")
    x = PositionalEmbedding(SEQ_LEN, src_vocab_size, TFM_HIDDEN_DIM)(source)
    enc_out = TransformerEncoder(TFM_HIDDEN_DIM, TFM_INTERMEDIATE_DIM, TFM_NUM_HEADS)(
        source=x, source_mask=source != 0
    )

    target = keras.Input(shape=(None,), dtype="int32", name="target")
    x = PositionalEmbedding(SEQ_LEN, tgt_vocab_size, TFM_HIDDEN_DIM)(target)
    x = TransformerDecoder(TFM_HIDDEN_DIM, TFM_INTERMEDIATE_DIM, TFM_NUM_HEADS)(
        target=x, source=enc_out, source_mask=source != 0
    )
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(tgt_vocab_size, activation="softmax")(x)
    return keras.Model([source, target], out)


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(model, train_ds, val_ds, save_path: Path, epochs: int):
    import keras

    if save_path.exists():
        print(f"[train] {save_path.name} already exists — skipping.")
        return

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        weighted_metrics=["accuracy"],
    )
    callbacks = [
        keras.callbacks.ModelCheckpoint(str(save_path), save_best_only=True),
        keras.callbacks.EarlyStopping(patience=5, monitor="val_loss", restore_best_weights=True),
    ]
    print(f"\n[train] Training {save_path.name} …")
    model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=callbacks)
    print(f"[train] Saved → {save_path.name}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main(directions=("en_fr", "fr_en")):
    MODELS_DIR.mkdir(exist_ok=True)

    # Save config
    with open(MODELS_DIR / "config.json", "w") as f:
        json.dump(CONFIG, f, indent=2)

    # Data
    download_data()
    raw_pairs = load_raw_pairs()
    train_pairs, val_pairs, _ = split_pairs(raw_pairs)
    print(f"[data] {len(train_pairs)} train / {len(val_pairs)} val pairs")

    if "en_fr" in directions:
        print("\n── EN→FR ──────────────────────────────────────────────────────")
        src_tok, tgt_tok = build_en_fr_tokenizers(train_pairs)
        save_vocab(src_tok, MODELS_DIR / "vocab_en_src.json")
        save_vocab(tgt_tok, MODELS_DIR / "vocab_fr_tgt.json")

        train_ds, val_ds = make_en_fr_datasets(train_pairs, val_pairs, src_tok, tgt_tok)

        rnn = build_rnn_model(VOCAB_SIZE, VOCAB_SIZE)
        train_model(rnn, train_ds, val_ds, MODELS_DIR / "rnn_en_fr.keras", RNN_EPOCHS)

        tfm = build_transformer_model(VOCAB_SIZE, VOCAB_SIZE)
        train_model(tfm, train_ds, val_ds, MODELS_DIR / "transformer_en_fr.keras", TFM_EPOCHS)

    if "fr_en" in directions:
        print("\n── FR→EN ──────────────────────────────────────────────────────")
        src_tok, tgt_tok = build_fr_en_tokenizers(train_pairs)
        save_vocab(src_tok, MODELS_DIR / "vocab_fr_src.json")
        save_vocab(tgt_tok, MODELS_DIR / "vocab_en_tgt.json")

        train_ds, val_ds = make_fr_en_datasets(train_pairs, val_pairs, src_tok, tgt_tok)

        rnn = build_rnn_model(VOCAB_SIZE, VOCAB_SIZE)
        train_model(rnn, train_ds, val_ds, MODELS_DIR / "rnn_fr_en.keras", RNN_EPOCHS)

        tfm = build_transformer_model(VOCAB_SIZE, VOCAB_SIZE)
        train_model(tfm, train_ds, val_ds, MODELS_DIR / "transformer_fr_en.keras", TFM_EPOCHS)

    print("\n[train] All done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EN↔FR translation models.")
    parser.add_argument(
        "--dir",
        choices=["en_fr", "fr_en", "both"],
        default="both",
        help="Which direction(s) to train (default: both)",
    )
    args = parser.parse_args()
    directions = ("en_fr", "fr_en") if args.dir == "both" else (args.dir,)
    main(directions)
