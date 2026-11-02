"""Project 52 - AI Chatbot with Transformer Models.

* A from-scratch NumPy Transformer encoder (multi-head self-attention, positional encoding,
  residual connections, layer norm, feed-forward) used as a sentence encoder.
* ``Chatbot``: lexical TF-IDF retrieval over intents, re-ranked with the encoder.
* ``HFChatBackend``: optional Hugging Face text generation (needs ``pip install transformers torch``).

Note: encoder weights are randomly initialised (deterministic seed), so it demonstrates the
architecture and acts as a re-ranker; swap in ``HFChatBackend`` for real language understanding.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

from .common.text import TfidfVectorizer, cosine_matrix, tokenize
from .common.nn import softmax


def positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    if d_model % 2:
        raise ValueError("d_model must be even")
    pos = np.arange(seq_len)[:, None]
    i = np.arange(d_model // 2)[None, :]
    angle = pos / np.power(10000.0, 2 * i / d_model)
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2], pe[:, 1::2] = np.sin(angle), np.cos(angle)
    return pe


def scaled_dot_product_attention(q, k, v, mask=None):
    """q,k,v: (..., T, d). mask: broadcastable bool, True = may attend. Returns (out, weights)."""
    scores = q @ np.swapaxes(k, -1, -2) / math.sqrt(q.shape[-1])
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    w = softmax(scores, axis=-1)
    return w @ v, w


def layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mu, var = x.mean(-1, keepdims=True), x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)


class MultiHeadAttention:
    def __init__(self, d_model: int, n_heads: int, rng: np.random.Generator):
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.h, self.dk = n_heads, d_model // n_heads
        s = 1 / math.sqrt(d_model)
        self.Wq, self.Wk, self.Wv, self.Wo = (rng.normal(0, s, (d_model, d_model)) for _ in range(4))

    def __call__(self, x: np.ndarray, mask=None):
        t, d = x.shape
        split = lambda a: a.reshape(t, self.h, self.dk).transpose(1, 0, 2)  # (h, T, dk)
        out, w = scaled_dot_product_attention(split(x @ self.Wq), split(x @ self.Wk), split(x @ self.Wv), mask)
        return out.transpose(1, 0, 2).reshape(t, d) @ self.Wo, w


class EncoderLayer:
    def __init__(self, d_model: int, n_heads: int, d_ff: int, rng: np.random.Generator):
        self.attn = MultiHeadAttention(d_model, n_heads, rng)
        self.W1 = rng.normal(0, 1 / math.sqrt(d_model), (d_model, d_ff))
        self.W2 = rng.normal(0, 1 / math.sqrt(d_ff), (d_ff, d_model))

    def __call__(self, x, mask=None):
        a, w = self.attn(x, mask)
        x = layer_norm(x + a)
        return layer_norm(x + np.maximum(x @ self.W1, 0) @ self.W2), w


class TransformerEncoder:
    def __init__(self, d_model=64, n_heads=4, d_ff=128, n_layers=2, max_len=64, seed=0):
        self.d_model, self.max_len = d_model, max_len
        rng = np.random.default_rng(seed)
        self.layers = [EncoderLayer(d_model, n_heads, d_ff, rng) for _ in range(n_layers)]
        self.pe = positional_encoding(max_len, d_model)

    def token_vector(self, token: str) -> np.ndarray:
        seed = int(hashlib.md5(token.encode()).hexdigest()[:8], 16)
        return np.random.default_rng(seed).normal(0, 1, self.d_model)

    def encode_tokens(self, tokens: list[str]) -> np.ndarray:
        tokens = tokens[: self.max_len]
        if not tokens:
            return np.zeros(self.d_model)
        x = np.stack([self.token_vector(t) for t in tokens]) + 0.1 * self.pe[: len(tokens)]
        for layer in self.layers:
            x, _ = layer(x)
        return x.mean(axis=0)

    def encode(self, text: str) -> np.ndarray:
        return self.encode_tokens(tokenize(text))


@dataclass
class Intent:
    name: str
    patterns: list[str]
    responses: list[str]


DEFAULT_INTENTS = [
    Intent("greeting", ["hello", "hi there", "hey", "good morning", "good evening"],
           ["Hello! How can I help you today?"]),
    Intent("goodbye", ["bye", "goodbye", "see you later", "talk to you soon"],
           ["Goodbye! Have a great day."]),
    Intent("thanks", ["thanks", "thank you", "thanks a lot", "much appreciated"],
           ["You're welcome!"]),
    Intent("name", ["what is your name", "who are you", "what should I call you"],
           ["I'm a small Transformer-powered chatbot."]),
    Intent("capabilities", ["what can you do", "help me", "how can you help", "your features"],
           ["I can answer FAQs about this project and chat about simple topics."]),
    Intent("transformer", ["what is a transformer", "explain attention mechanism",
                           "how do transformers work", "self attention"],
           ["A Transformer uses self-attention to let every token weigh every other token, "
            "followed by feed-forward layers, residual connections and layer norm."]),
]


class Chatbot:
    FALLBACK = "Sorry, I didn't understand that. Could you rephrase?"

    def __init__(self, intents: list[Intent] | None = None, threshold: float = 0.3,
                 encoder: TransformerEncoder | None = None):
        self.intents = intents or DEFAULT_INTENTS
        self.threshold = threshold
        self.encoder = encoder or TransformerEncoder()
        self.history: list[tuple[str, str]] = []
        self._examples = [(i, p) for i, intent in enumerate(self.intents) for p in intent.patterns]
        texts = [p for _, p in self._examples]
        self._tfidf = TfidfVectorizer(remove_stopwords=False)
        self._lex = self._tfidf.fit_transform(texts)
        self._enc = np.stack([self.encoder.encode(t) for t in texts])
        self._turn = 0

    def classify(self, text: str) -> tuple[Intent | None, float]:
        lex = cosine_matrix(self._tfidf.transform([text]), self._lex)[0]
        if lex.max() <= 0:
            return None, 0.0
        enc = np.clip(cosine_matrix(self.encoder.encode(text)[None], self._enc)[0], 0, 1)
        score = 0.7 * lex + 0.3 * enc * (lex > 0)
        best = int(np.argmax(score))
        if score[best] < self.threshold:
            return None, float(score[best])
        return self.intents[self._examples[best][0]], float(score[best])

    def reply(self, text: str) -> str:
        if not text.strip():
            return "Please type something."
        intent, _ = self.classify(text)
        if intent is None:
            answer = self.FALLBACK
        else:
            answer = intent.responses[self._turn % len(intent.responses)]
            self._turn += 1
        self.history.append((text, answer))
        return answer


class HFChatBackend:  # pragma: no cover - needs optional dependencies / model download
    """Generative backend using Hugging Face ``transformers`` (e.g. ``google/flan-t5-small``)."""

    def __init__(self, model: str = "google/flan-t5-small"):
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise ImportError("pip install transformers torch") from exc
        self._pipe = pipeline("text2text-generation", model=model)

    def reply(self, text: str) -> str:
        return self._pipe(text, max_new_tokens=80)[0]["generated_text"]


def demo() -> None:
    bot = Chatbot()
    for q in ["hello", "how do transformers work", "thank you", "blorp zxq"]:
        print(f"> {q}\n{bot.reply(q)}")


if __name__ == "__main__":  # pragma: no cover
    demo()
