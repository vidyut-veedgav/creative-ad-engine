# Giant Creative Ad Engine

## Overview

The Giant Creative Ad Engine analyzes ad-performance data for *Giant* (an AI storytelling
app for children) and surfaces, through a conversational UI, which creative themes are
rising or declining and what to test next.

It runs as a 5-layer pipeline. The first three layers are **deterministic Python (no LLM)**,
so every answer is traceable back to computed metrics:

1. **Signal** (`src/signal.py`) — turns per-ad 7-day deltas into per-theme trend, strength, and uniformity.
2. **Diagnosis** (`src/diagnosis.py`) — identifies leading/lagging/outlier ads and dominant format/placement.
3. **Hypothesis** (`src/hypothesis.py`) — produces an ICE-scored, ranked backlog of recommendations.
4. **Agent** (`src/agent.py`) — a Gemini (`gemini-2.5-flash`) assistant that *interprets* the layers above through typed tools; it never recomputes metrics itself.
5. **App** (`app/app.py`) — a Streamlit dashboard + chatbot that loads the data, runs layers 1–3, and serves the agent.

See `docs/system_architecture.md` for the full design spec and each layer's input/output contract.

## How to run

Requires Python 3.11+.

```bash
# 1. Set up the environment
python -m venv .venv
source .venv/bin/activate          # Windows (PowerShell): .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. (Optional) Set the API key for the conversational agent
export GEMINI_API_KEY=your-key-here   # or GOOGLE_API_KEY

# 3. Launch the app
streamlit run app/app.py
```

### Regenerate the dataset (optional)

The dataset ships in `data/`. To regenerate it deterministically (seed = 42):

```bash
cd data && python ../util/generate_dataset.py
```

### Validate the deterministic layers (optional)

Each layer has a narrative-assertion harness; run from the repo root:

```bash
python -m util.verify_signals
python -m util.verify_diagnosis
python -m util.verify_hypothesis
```
