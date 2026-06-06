# Giant Creative Ad Engine — Project Context

## What This Project Is

A data pipeline and AI-powered analysis tool for Giant, a personalized AI storytelling app for young children. Giant uses paid ads as a primary growth driver. This system ingests ad performance data, computes trend signals, diagnoses creative performance, generates hypotheses for what to test next, and surfaces everything through a conversational interface.

The system is built for a technical interview submission due June 8. The evaluator wants to see technical reasoning and architecture decisions — not polished slides. The demo moment is a growth team member asking "which themes are declining and what should we test next?" and getting a reasoned, evidence-grounded answer in plain language.

---

## Repo Structure

```
giant-ad-engine/
├── data/
│   ├── ads.csv
│   └── engagement.csv
├── signal_layer.py
├── diagnosis_layer.py
├── hypothesis_layer.py
├── agent.py
├── app.py
├── context.md
└── requirements.txt
```

---

## Data

### Two tables. Static dataset. No streaming.

**`data/ads.csv`** — one row per ad, static metadata

| Column | Type | Values / Notes |
|---|---|---|
| ad_id | str | unique identifier |
| platform | str | `"meta"` \| `"google"` |
| campaign_id | str | |
| adset_id | str | |
| theme | str | `"wonder"` \| `"safety"` \| `"confidence"` \| `"connection"` |
| format | str | `"video"` \| `"static"` \| `"carousel"` |
| hook_text | str | opening line of the ad |
| headline | str | |
| primary_text | str | |
| call_to_action | str | |
| video_length_sec | float | null for static and carousel |
| aspect_ratio | str | `"9:16"` \| `"1:1"` \| `"4:5"` |
| placement | str | `"feed"` \| `"stories"` \| `"reels"` \| `"search"` \| `"youtube"` |
| country | str | `"US"` |
| device_type | str | `"mobile"` \| `"desktop"` |
| audience_type | str | `"prospecting"` \| `"retargeting"` \| `"lookalike"` |

**`data/engagement.csv`** — one row per ad per day, composite key: `ad_id + date`

| Column | Type | Notes |
|---|---|---|
| ad_id | str | foreign key to ads.csv |
| date | date | |
| spend | float | |
| impressions | int | |
| clicks | int | |
| installs | int | |
| CTR | float | derived: clicks / impressions |
| CPI | float | derived: spend / installs |
| CPM | float | derived: (spend / impressions) * 1000 |
| CPC | float | derived: spend / clicks |
| CVR | float | derived: installs / clicks |
| thumbstop_rate | float | video only, null otherwise |
| hold_rate | float | video only, null otherwise |
| completion_rate | float | video only, null otherwise |
| video_plays_25pct | float | video only, null otherwise |
| video_plays_75pct | float | video only, null otherwise |
| avg_play_time_sec | float | video only, null otherwise |
| card_swipe_rate | float | carousel only, null otherwise |
| last_card_reach_rate | float | carousel only, null otherwise |
| likes | int | Meta only, static and carousel, null for video and all Google |
| comments | int | Meta only, static and carousel, null for video and all Google |
| shares | int | Meta only, static and carousel, null for video and all Google |
| saves | int | Meta only, static and carousel, null for video and all Google |

### Dataset parameters

- 4 themes × 6 ads each = 24 total ads
- 90 days of daily rows = 2,160 rows in engagement table
- Dataset is small by design — entire theme-level slices fit in an LLM context window without RAG

### Baked-in narratives — these must be detectable by the signal layer

| Theme | State | What the data shows |
|---|---|---|
| Wonder | Rising, strong | CPI declining from ~$4.20 to ~$2.80 over 90 days. CTR rising. Completion rate >40%. CPI improvement rate slows after day 60 — efficiency ceiling. |
| Safety | Declining from day 35 | CPI rising, CTR falling after day 35. `video_plays_75pct` and `avg_play_time_sec` drop sharply from day 35 — viewers dropping off mid-video, not at the hook. |
| Confidence | Stable, mid-tier | Flat metrics with natural variance. No strong trend direction. |
| Connection | Volatile / mixed signal | High week-over-week variance. Two specific video ads on reels outperform cluster by ~40% CPI. Overall slightly positive but noisy. |

### Placement distribution (deliberate, not random)

| Theme | Placements |
|---|---|
| Wonder | feed, reels, youtube |
| Safety | feed, stories |
| Confidence | feed, stories, search |
| Connection | feed, reels, stories |

### Format-level CVR differentials (global)

| Format | CTR vs baseline | CVR vs baseline |
|---|---|---|
| Video | 1.4x | 1.0x |
| Static | 0.85x | 0.75x |
| Carousel | 0.80x | 1.25x |

---

## System Architecture

Five layers. Each layer has a clean input/output contract. No layer skips a layer.

```
data/ads.csv + data/engagement.csv
            ↓
      signal_layer.py          →  theme_signals: dict
            ↓
     diagnosis_layer.py        →  theme_diagnosis: dict
            ↓
     hypothesis_layer.py       →  ranked_backlog: list[dict]
            ↓
         agent.py              →  natural language answers
            ↓
          app.py               →  Streamlit UI (chatbot + dashboard)
```

### Layer 1 — signal_layer.py

**What it does:** converts raw CSV data into trend evidence at the ad level and theme level. All logic is deterministic Python. No LLM calls.

**Computes at the ad level (per rolling 7-day window):**
- `cpi_7d_delta_pct` — percentage change in CPI over last 7 days. Negative = improving.
- `ctr_7d_delta_pct` — percentage change in CTR
- `cvr_7d_delta_pct` — percentage change in CVR
- `hold_rate_7d_delta_pct` — video only, null otherwise
- `thumbstop_rate_7d_delta_pct` — video only, null otherwise
- `trend_direction` — `"rising"` \| `"declining"` \| `"stable"` — classified from `cpi_7d_delta_pct`
- `trend_strength` — `"strong"` \| `"moderate"` \| `"weak"` — thresholds: >15% change = strong, 5–15% = moderate, <5% = stable
- `efficiency_ceiling` — bool — True if CPI delta is flattening over the last 30 days despite stable or increasing spend (30-day rolling slope of CPI delta approaching zero)

**Aggregates to theme level:**
- Weighted means of all delta metrics across ads in theme
- `ads_rising`, `ads_declining`, `ads_stable` — counts
- `uniformity` — `"uniform"` \| `"isolated"` \| `"mixed"` — >60% same direction = uniform, <40% = isolated, 40–60% = mixed
- `trend_direction`, `trend_strength` — dominant direction/strength across theme's ads
- `diagnosis_level` — `"strategy"` (uniform) \| `"execution"` (isolated)
- `hypothesis_type` — derived from direction + uniformity + efficiency_ceiling (see mapping below)
- `efficiency_ceiling` — True if majority of ads in theme show ceiling signal

**Hypothesis type mapping:**
```
uniform + rising + no ceiling  →  amplification
uniform + rising + ceiling      →  optimization
uniform + declining             →  retirement
isolated + rising outliers      →  replication
isolated + declining outliers   →  intervention
stable                          →  optimization (monitor)
```

**Output contract — `theme_signals` dict:**
```python
{
  "<theme_name>": {
    "trend_direction":              "rising" | "declining" | "stable",
    "trend_strength":               "strong" | "moderate" | "weak",
    "uniformity":                   "uniform" | "isolated" | "mixed",
    "cpi_7d_delta_pct":             float,        # negative = improving
    "ctr_7d_delta_pct":             float,
    "cvr_7d_delta_pct":             float,
    "hold_rate_7d_delta_pct":       float | None, # video themes only
    "thumbstop_rate_7d_delta_pct":  float | None, # video themes only
    "ads_rising":                   int,
    "ads_declining":                int,
    "ads_stable":                   int,
    "diagnosis_level":              "strategy" | "execution",
    "hypothesis_type":              "amplification" | "optimization" | "intervention" | "retirement" | "replication",
    "efficiency_ceiling":           bool
  }
}
```

**Public interface:**
```python
def build_theme_signals(ads_path: str, engagement_path: str) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Orchestrator. Returns:
    - theme_signals dict
    - ads_df (raw)
    - eng_df (raw)
    - ad_signals_df (ad-level computed signals, one row per ad)
    All four are needed downstream; return them together to avoid recomputation.
    """
```

---

### Layer 2 — diagnosis_layer.py

**What it does:** uses signals to produce structural and relative evidence. Explains *why* each theme is in its current state by drilling into ad-level patterns. Called for every theme, not just declining ones.

**For every theme it identifies:**
- Leading ads (top performers within theme by CPI delta)
- Lagging ads (bottom performers)
- Outlier ads (ads moving against the theme's dominant trend — only meaningful when uniformity is isolated/mixed)
- Dominant format and placement (what's driving the theme's performance)
- Weak format and placement (where the theme underperforms its own average, if anywhere)
- Distinguishing features of outlier ads (cross-tabulated against non-outliers in same theme)
- A deterministic `pattern_summary` string — one sentence, factual, metric-grounded, injected verbatim into agent context

**Material difference threshold:** >15% gap vs theme mean CPI delta to classify a format or placement as dominant or weak.

**Output contract — `theme_diagnosis` dict:**
```python
{
  "<theme_name>": {
    "leading_ads": [
      {
        "ad_id":            str,
        "format":           str,
        "placement":        str,
        "cpi_7d_delta_pct": float,
        "ctr_7d_delta_pct": float
      }
    ],
    "lagging_ads":        [...],  # same shape as leading_ads
    "outlier_ads": [
      {
        "ad_id":                    str,
        "format":                   str,
        "placement":                str,
        "cpi_7d_delta_pct":         float,
        "ctr_7d_delta_pct":         float,
        "distinguishing_features":  list[str]  # e.g. ["format:video", "placement:reels"]
      }
    ],
    "dominant_format":    str | None,
    "dominant_placement": str | None,
    "weak_format":        str | None,
    "weak_placement":     str | None,
    "pattern_summary":    str   # e.g. "5 of 6 ads rising; video on reels driving the trend"
  }
}
```

**Public interface:**
```python
def build_theme_diagnosis(
    theme_signals: dict,
    ad_signals_df: pd.DataFrame,
    ads_df: pd.DataFrame
) -> dict:
    """
    Orchestrator. Returns theme_diagnosis dict for all themes.
    """
```

---

### Layer 3 — hypothesis_layer.py

**What it does:** uses theme_signals and theme_diagnosis to generate ICE-scored hypotheses and a ranked action backlog. All logic is deterministic Python. No LLM calls.

**Hypothesis types and their triggers:**

| Type | Level | Trigger |
|---|---|---|
| Amplification | Strategy | Strong rising theme, uniform signal, no efficiency ceiling — increase spend, expand placements |
| Optimization | Strategy | Rising theme hitting efficiency ceiling, or stable theme — refresh creative, test new angles |
| Replication | Execution | Isolated rising outliers within a theme — identify the winning pattern, replicate across cluster |
| Intervention | Execution | Isolated declining ads within a theme — fix execution (format/placement), not theme |
| Retirement | Strategy | Uniform declining signal across theme — kill theme, reallocate budget |

**ICE scoring:**
- `ice_impact` — how much could this move the needle? Grounded in CPI delta magnitude and spend level.
- `ice_confidence` — how strong is the signal? Grounded in trend_strength and uniformity.
- `ice_ease` — how quickly can this be executed? Amplification/retirement = easy. New creative = harder.
- `ice_score` — 1–10 composite

**Output contract — `ranked_backlog` list:**
```python
[
  {
    "rank":             int,
    "hypothesis_type":  "amplification" | "optimization" | "intervention" | "retirement" | "replication",
    "level":            "strategy" | "execution",
    "theme":            str,
    "recommendation":   str,   # specific and data-grounded — what to do
    "evidence":         str,   # specific metrics + time windows + comparisons
    "ice_impact":       "high" | "medium" | "low",
    "ice_confidence":   "high" | "medium" | "low",
    "ice_ease":         "high" | "medium" | "low",
    "ice_score":        int,   # 1–10
    "open_questions":   list[str]  # what additional data would increase confidence
  }
]
```

**Public interface:**
```python
def build_ranked_backlog(
    theme_signals: dict,
    theme_diagnosis: dict
) -> list[dict]:
    """
    Orchestrator. Returns ranked_backlog list sorted by ice_score descending.
    """
```

---

### Layer 4 — agent.py

**What it does:** an LLM-powered layer that intelligently pulls precomputed analytics to answer questions and explain outputs. The agent knows the schema and the shape of all precomputed artifacts. It does not compute or derive metrics — it interprets, explains, and synthesizes.

**Key principle:** the agent is downstream of all deterministic analysis. Stages 1–3 run on startup and produce verified, traceable outputs. The agent reasons over those outputs. This makes answers reliable and every claim traceable to a specific metric, time window, and theme cluster.

**Tools the agent has access to:**
```python
get_theme_signals(theme: str | None) -> dict
# Returns theme_signals for one theme or all themes if theme=None

get_theme_diagnosis(theme: str) -> dict
# Returns theme_diagnosis for a specific theme

get_ranked_backlog(theme: str | None, hypothesis_type: str | None) -> list[dict]
# Returns backlog items, optionally filtered by theme or hypothesis_type

get_ad_detail(ad_id: str) -> dict
# Returns ad metadata + signal data for a specific ad
```

The agent calls these tools based on what the query requires. It does not load everything upfront — it pulls the specific slices relevant to each question.

**Two primary functions:**

```python
def generate_backlog_narrative(ranked_backlog: list[dict], theme_signals: dict) -> str:
    """
    Called once at startup. Produces a natural language narrative summarizing
    the full ranked backlog. Cached and used as context in answer_query().
    """

def answer_query(query: str, tools: dict) -> str:
    """
    Handles a single conversational query. Agent decides which tools to call
    based on the query, pulls relevant precomputed data, and returns a
    plain-language answer with inline metric citations.
    Every factual claim must reference a specific metric and time window.
    """
```

**Questions the agent must be able to answer:**

Marketer — snapshot:
1. Which themes are performing best right now?
2. Which themes are declining and how fast?
3. What's driving the decline in [theme] — which specific ads, and within those ads, is it the hook, the format, or the placement?
4. Is [theme]'s signal uniform across all its ads or isolated to specific ones?
5. Are there standout ads within an otherwise mixed or volatile theme?

Growth lead — hypothesis:
6. What new creative concepts should we test next, and what's the narrative/emotional direction for each?
7. What format should we test them in?
8. What placements should we test them in?
9. Should we scale spend on [theme] or have we hit an efficiency ceiling?
10. Are there specific ads within a declining theme worth saving, or should we kill the whole theme?

Writer — creative direction:
11. What emotional beats and narrative angles are resonating with parents right now?

CEO — summary:
12. Give me a 2-minute summary of where our paid creative strategy stands.

Engineer — traceability:
13. Why are you recommending [X] — walk me through the evidence.

---

### Layer 5 — app.py

**What it does:** Streamlit application. Wires all layers together. Runs the pipeline on startup, renders the conversational interface and a supporting dashboard.

**Startup sequence (runs once, cached in `st.session_state`):**
```python
theme_signals, ads_df, eng_df, ad_signals_df = signal_layer.build_theme_signals(ADS_PATH, ENG_PATH)
theme_diagnosis = diagnosis_layer.build_theme_diagnosis(theme_signals, ad_signals_df, ads_df)
ranked_backlog = hypothesis_layer.build_ranked_backlog(theme_signals, theme_diagnosis)
backlog_narrative = agent.generate_backlog_narrative(ranked_backlog, theme_signals)
```

**UI layout:**
- Left panel: theme trend summary table (theme, direction, strength, hypothesis type) + ranked backlog table
- Right panel: conversational chat interface — input box, message history, agent responses with inline metric citations
- Single Streamlit file, no routing, no authentication

---

## Analytical Mental Model

### Two analytical levels
- **Theme level (Claim layer)** — where should we focus?
- **Creative level (Evidence layer)** — what specifically should we change?

### Two analytical modes
- **Snapshot** — what is the state right now? Descriptive and diagnostic. Questions 1–5.
- **Hypothesis** — given the snapshot, what should we change? Generative and prescriptive. Questions 6–10. Always downstream of snapshot.

### Uniformity check logic
```
> 60% of ads in theme trend same direction  →  uniform  →  strategy-level diagnosis
40–60%                                      →  mixed    →  both levels
< 40%                                       →  isolated →  execution-level diagnosis
```

### Distinguishing intervention from retirement
- Weakness isolated to specific ads / formats / placements → fixable → **intervention**
- Weakness uniform across all ads in theme across all formats and platforms → theme-level problem → **retirement**

---

## Key Constraints and Design Decisions

**No backend server.** Streamlit calls pipeline functions directly. Function signatures are designed to be trivially wrappable in FastAPI later if needed.

**Agent is downstream of all analysis.** The LLM never sees raw data. It calls typed accessors into precomputed artifacts. This makes outputs reliable and every claim traceable.

**Three deterministic stages before the agent.** Each stage is independently testable. Signal layer validated against known baked-in narratives before diagnosis layer is written.

**Diagnosis runs for every theme.** Not just declining ones. Rising themes like Wonder still need ad-level diagnosis to answer "which formats and placements are driving the rise" — that's the evidence base for amplification hypotheses.

**`pattern_summary` is deterministic, not generated.** The one-sentence per-theme summary injected into agent context is computed in the diagnosis layer, not produced by the LLM. Agent answers are grounded in verifiable facts.

**Small dataset by design.** 2,160 rows fits in an LLM context window. No vector database or RAG layer needed. Explicit architectural decision, not a limitation — worth calling out in the submission.

**Derived metrics computed from base metrics, not stored independently.** CTR, CPI, CPM, CPC, CVR are all in engagement.csv. The signal layer computes deltas from these; it does not recompute them from impressions/clicks/spend.

**No demographic columns in schema.** Demographic data lives at the audience/ad set level in real platforms, not the ad level. `audience_type` (prospecting / retargeting / lookalike) is the meaningful distinction available at ad level.

**Meta targeting note.** Meta restricts targeting users under 18. All campaigns target parents, not children. Parents are the buyer; the child's experience is the emotional proof. This affects how engagement metrics are interpreted — CTR and CVR reflect parent behavior, not child behavior.

---

## Stack

- Python 3.11+
- pandas — data loading, signal computation, aggregations
- anthropic — LLM calls in agent.py (use `claude-sonnet-4-20250514`)
- streamlit — application layer
- No backend server, no database, no vector store

---

## What Success Looks Like

The system is complete when a user can ask any of the 13 questions listed above and receive a plain-language answer that:
- Cites specific metrics and time windows
- Is traceable back to a precomputed signal or diagnosis value
- Distinguishes between theme-level claims and ad-level claims
- Gives a specific recommended action, not a general observation