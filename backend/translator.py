"""
translator.py — TranslationEngine

Loads trained models + vocabulary files and exposes a single translate() method.

Models are loaded lazily on first use so the server starts instantly even if
all four models are available.
"""

import json
import logging
import re
import string
from pathlib import Path
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

Direction = Literal["en_fr", "fr_en"]
ModelType = Literal["rnn", "transformer"]

# ── Text standardization (must match train.py) ────────────────────────────────
_STRIP = string.punctuation.replace("[", "").replace("]", "")
_STRIP_RE = re.compile(f"[{re.escape(_STRIP)}]")


def _py_standardize(text: str) -> str:
    """Python equivalent of the TF custom_standardize used during training."""
    return _STRIP_RE.sub("", text.lower())


# ── Vocabulary helpers ────────────────────────────────────────────────────────

def _load_vocab(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class _Tokenizer:
    """
    Pure-Python tokenizer backed by a saved vocabulary list.

    Avoids TextVectorization.set_vocabulary() whose special-token handling
    (mask="" at 0, OOV="[UNK]" at 1) differs across Keras versions and can
    silently shift every token index by 2, corrupting all translations.

    Contract: vocab[i] == the string the model learned as token i during
    training, so word_to_idx[word] == the integer the model received for
    that word.  The output of __call__ is a (len(texts), seq_len) int32
    numpy array, which Keras model.predict accepts directly.
    """

    def __init__(self, vocab: list[str], seq_len: int) -> None:
        self._seq_len = seq_len
        self._oov_idx = 1  # vocab[1] is always "[UNK]"
        self._word_to_idx: dict[str, int] = {w: i for i, w in enumerate(vocab)}

    def __call__(self, texts: list[str]) -> np.ndarray:
        result = []
        for text in texts:
            tokens = _py_standardize(text).split()
            ids = [self._word_to_idx.get(t, self._oov_idx) for t in tokens]
            ids = ids[: self._seq_len]
            ids += [0] * (self._seq_len - len(ids))
            result.append(ids)
        return np.array(result, dtype="int32")


# ── Decoder ───────────────────────────────────────────────────────────────────

def _decode_greedy(
    model,
    src_tokens,  # shape (1, SEQ_LEN)
    tgt_tok,
    tgt_lookup: dict[int, str],
    seq_len: int,
    is_transformer: bool,
) -> str:
    """Auto-regressive greedy decoding."""
    # Read actual input names from the model so models saved with
    # "english"/"french" (notebook) and "source"/"target" (our training
    # script) both work without re-migration.
    src_key, tgt_key = [inp.name for inp in model.inputs]

    decoded = "[start]"
    for i in range(seq_len):
        tgt_tokens = tgt_tok([decoded])  # (1, seq_len+1)
        if is_transformer:
            tgt_tokens = tgt_tokens[:, :-1]  # keep within positional embedding range
        preds = model.predict(
            {src_key: src_tokens, tgt_key: tgt_tokens}, verbose=0
        )
        next_idx = int(np.argmax(preds[0, i, :]))
        next_tok = tgt_lookup.get(next_idx, "")
        decoded += " " + next_tok
        if next_tok == "[end]":
            break

    # Strip sentinel tokens and return
    tokens = [t for t in decoded.split() if t not in ("[start]", "[end]", "")]
    return " ".join(tokens)


# ── Engine ────────────────────────────────────────────────────────────────────

class TranslationEngine:
    """
    Loads models and vocabularies from models_dir and translates text.

    Expected files in models_dir:
        config.json
        vocab_en_src.json, vocab_fr_tgt.json   (for EN→FR)
        vocab_fr_src.json, vocab_en_tgt.json   (for FR→EN)
        rnn_en_fr.keras, transformer_en_fr.keras
        rnn_fr_en.keras, transformer_fr_en.keras
    """

    def __init__(self, models_dir: str | Path):
        self.models_dir = Path(models_dir)
        self._config = self._load_config()
        self._seq_len: int = self._config["seq_len"]

        # Lazy-loaded models  {(direction, model_type): keras.Model}
        self._models: dict[tuple[str, str], object] = {}

        # Tokenizers and lookup tables built once per direction
        self._src_toks: dict[str, object] = {}
        self._tgt_toks: dict[str, object] = {}
        self._tgt_lookups: dict[str, dict[int, str]] = {}

        self._init_tokenizers()

    # ── Internal setup ────────────────────────────────────────────────────────

    def _load_config(self) -> dict:
        cfg_path = self.models_dir / "config.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                return json.load(f)
        return {"vocab_size": 20000, "seq_len": 30}

    def _init_tokenizers(self) -> None:
        seq = self._seq_len
        dir_specs = {
            "en_fr": ("vocab_en_src.json", "vocab_fr_tgt.json"),
            "fr_en": ("vocab_fr_src.json", "vocab_en_tgt.json"),
        }
        for direction, (src_file, tgt_file) in dir_specs.items():
            src_path = self.models_dir / src_file
            tgt_path = self.models_dir / tgt_file
            if not src_path.exists() or not tgt_path.exists():
                logger.warning(
                    "Vocab files for %s not found — direction unavailable.", direction
                )
                continue

            src_vocab = _load_vocab(src_path)
            tgt_vocab = _load_vocab(tgt_path)

            self._src_toks[direction] = _Tokenizer(src_vocab, seq)
            self._tgt_toks[direction] = _Tokenizer(tgt_vocab, seq + 1)
            self._tgt_lookups[direction] = dict(enumerate(tgt_vocab))
            logger.info("Tokenizers ready for %s.", direction)

    def _load_model(self, direction: Direction, model_type: ModelType):
        key = (direction, model_type)
        if key in self._models:
            return self._models[key]

        import keras
        from backend.model_definitions import CUSTOM_OBJECTS

        filename = f"{model_type}_{direction}.keras"
        model_path = self.models_dir / filename
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {filename}. "
                "Run `python -m backend.train` to train the models first."
            )

        logger.info("Loading %s …", filename)
        model = keras.models.load_model(str(model_path), custom_objects=CUSTOM_OBJECTS)
        self._models[key] = model
        logger.info("Loaded %s.", filename)
        return model

    # ── Public API ────────────────────────────────────────────────────────────

    def available_directions(self) -> list[str]:
        return list(self._src_toks.keys())

    def model_exists(self, direction: Direction, model_type: ModelType) -> bool:
        return (self.models_dir / f"{model_type}_{direction}.keras").exists()

    def translate(
        self,
        text: str,
        source_lang: Literal["en", "fr"],
        target_lang: Literal["en", "fr"],
        model_type: ModelType = "transformer",
    ) -> str:
        if source_lang == target_lang:
            return text

        direction: Direction = f"{source_lang}_{target_lang}"  # type: ignore[assignment]

        if direction not in self._src_toks:
            raise ValueError(
                f"Direction '{direction}' not available. "
                "Run `python -m backend.train` to train the required models."
            )

        model = self._load_model(direction, model_type)
        src_tok = self._src_toks[direction]
        tgt_tok = self._tgt_toks[direction]
        tgt_lookup = self._tgt_lookups[direction]

        src_tokens = src_tok([text])  # (1, SEQ_LEN)
        return _decode_greedy(
            model,
            src_tokens,
            tgt_tok,
            tgt_lookup,
            self._seq_len,
            is_transformer=(model_type == "transformer"),
        )

    def status(self) -> dict:
        """Return a dict describing which models / vocabs are present."""
        directions = ["en_fr", "fr_en"]
        model_types = ["rnn", "transformer"]
        return {
            "tokenizers": {d: d in self._src_toks for d in directions},
            "models": {
                f"{mt}_{d}": self.model_exists(d, mt)  # type: ignore
                for d in directions
                for mt in model_types
            },
        }
