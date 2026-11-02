# Project 52: AI Chatbot with Transformer Models

From-scratch NumPy Transformer encoder (multi-head attention, positional encoding, residuals, layer norm) used with TF-IDF retrieval as an intent chatbot.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run the demo
```bash
python -m aiprojects.p52_transformer_chatbot
```

## Run the tests
```bash
python -m unittest discover -s tests -t .
```

## Structure
- `aiprojects/p52_transformer_chatbot.py` - the project (read the module docstring first)
- `aiprojects/common/` - small shared helpers (text utils, NumPy MLP, toy game environments)
- `tests/test_p52.py` - unit tests

Requires Python 3.10+ and NumPy only.

## Notes
Encoder weights are random (fixed seed): it demonstrates the architecture and re-ranks lexical matches. For real language understanding use `HFChatBackend` (`pip install transformers torch`; untested here, needs a model download).
