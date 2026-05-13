"""
Custom Keras layers for the Transformer-based translation model.

Each layer implements get_config() so models can be saved and reloaded
without re-defining these classes in the same session.
"""

import keras
from keras import layers, ops


class TransformerEncoder(keras.Layer):
    """Single Transformer encoder block (self-attention + feed-forward)."""

    def __init__(self, hidden_dim: int, intermediate_dim: int, num_heads: int, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        key_dim = hidden_dim // num_heads
        self.self_attention = layers.MultiHeadAttention(num_heads, key_dim)
        self.norm1 = layers.LayerNormalization()
        self.ff1 = layers.Dense(intermediate_dim, activation="relu")
        self.ff2 = layers.Dense(hidden_dim)
        self.norm2 = layers.LayerNormalization()

    def call(self, source, source_mask):
        mask = source_mask[:, None, :]
        attn = self.self_attention(query=source, key=source, value=source, attention_mask=mask)
        x = self.norm1(source + attn)
        ff = self.ff2(self.ff1(x))
        return self.norm2(x + ff)

    def get_config(self):
        config = super().get_config()
        config.update(
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            num_heads=self.num_heads,
        )
        return config


class TransformerDecoder(keras.Layer):
    """Single Transformer decoder block (masked self-attention + cross-attention + feed-forward)."""

    def __init__(self, hidden_dim: int, intermediate_dim: int, num_heads: int, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        key_dim = hidden_dim // num_heads
        self.self_attention = layers.MultiHeadAttention(num_heads, key_dim)
        self.norm1 = layers.LayerNormalization()
        self.cross_attention = layers.MultiHeadAttention(num_heads, key_dim)
        self.norm2 = layers.LayerNormalization()
        self.ff1 = layers.Dense(intermediate_dim, activation="relu")
        self.ff2 = layers.Dense(hidden_dim)
        self.norm3 = layers.LayerNormalization()

    def call(self, target, source, source_mask):
        self_attn = self.self_attention(query=target, key=target, value=target, use_causal_mask=True)
        x = self.norm1(target + self_attn)
        mask = source_mask[:, None, :]
        cross = self.cross_attention(query=x, key=source, value=source, attention_mask=mask)
        x = self.norm2(x + cross)
        ff = self.ff2(self.ff1(x))
        return self.norm3(x + ff)

    def get_config(self):
        config = super().get_config()
        config.update(
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            num_heads=self.num_heads,
        )
        return config


class PositionalEmbedding(keras.Layer):
    """Token embedding + learned positional embedding summed together."""

    def __init__(self, sequence_length: int, vocab_size: int, embed_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_embeddings = layers.Embedding(vocab_size, embed_dim)
        self.position_embeddings = layers.Embedding(sequence_length, embed_dim)

    def call(self, inputs):
        positions = ops.cumsum(ops.ones_like(inputs), axis=-1) - 1
        return self.token_embeddings(inputs) + self.position_embeddings(positions)

    def get_config(self):
        config = super().get_config()
        config.update(
            sequence_length=self.sequence_length,
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
        )
        return config

    @classmethod
    def from_config(cls, config):
        # The Colab notebook saved models with input_dim/output_dim; our code
        # uses vocab_size/embed_dim.  Accept either so both load correctly.
        config = config.copy()
        if "input_dim" in config:
            config["vocab_size"] = config.pop("input_dim")
        if "output_dim" in config:
            config["embed_dim"] = config.pop("output_dim")
        return cls(**config)


CUSTOM_OBJECTS = {
    "TransformerEncoder": TransformerEncoder,
    "TransformerDecoder": TransformerDecoder,
    "PositionalEmbedding": PositionalEmbedding,
}
