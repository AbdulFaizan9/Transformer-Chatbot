import os
import tempfile
import unittest
from datetime import datetime, timedelta

import numpy as np

from aiprojects import p52_transformer_chatbot as p52


class P52(unittest.TestCase):
    def test_attention_properties(self):
        rng = np.random.default_rng(0)
        q = rng.normal(size=(5, 8))
        causal = np.tril(np.ones((5, 5), bool))
        out, w = p52.scaled_dot_product_attention(q, q, q, causal)
        self.assertTrue(np.allclose(w.sum(-1), 1))
        self.assertTrue(np.allclose(np.triu(w, 1), 0))
        self.assertEqual(out.shape, (5, 8))

    def test_positional_encoding_and_encoder(self):
        pe = p52.positional_encoding(10, 16)
        self.assertEqual(pe.shape, (10, 16))
        self.assertAlmostEqual(pe[0, 0], 0.0)
        enc = p52.TransformerEncoder()
        v1, v2 = enc.encode("hello world"), enc.encode("hello world")
        self.assertTrue(np.allclose(v1, v2))
        self.assertEqual(v1.shape, (64,))
        self.assertTrue(np.all(enc.encode("") == 0))
        with self.assertRaises(ValueError):
            p52.MultiHeadAttention(10, 3, np.random.default_rng(0))

    def test_chatbot(self):
        bot = p52.Chatbot()
        self.assertIn("Hello", bot.reply("hello"))
        self.assertIn("Goodbye", bot.reply("ok goodbye"))
        self.assertIn("self-attention", bot.reply("how do transformers work?"))
        self.assertEqual(bot.reply("qwzx vbnm"), bot.FALLBACK)
        self.assertEqual(bot.reply("   "), "Please type something.")
        self.assertEqual(len(bot.history), 4)


if __name__ == "__main__":
    unittest.main()
