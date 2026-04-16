# Reply Agentic AI Challenge 2026

This repository contains my submission code for the Reply Agentic AI Challenge 2026.

The challenge focused on building an agent-based fraud detection pipeline that could analyze multi-source financial datasets and identify suspicious transactions with interpretable reasoning. The solution combines transaction behavior, user behavior, location anomalies, communication phishing signals, and audio-adjacent risk into a modular scoring pipeline.

## Challenge Overview

The task was to:

- read one dataset folder at a time
- inspect schema dynamically instead of hard-coding assumptions
- combine multiple evidence sources
- output only suspected fraudulent transaction IDs
- trace pipeline runs with Langfuse

Each dataset included different combinations of:

- `transactions.csv`
- `users.json`
- `sms.json`
- `mails.json`
- `locations.json`
- `audio/` for larger datasets

## Solution Approach

The project is built around an extensible agent-style architecture:

- `TransactionPatternAgent`
- `UserBehaviorAgent`
- `LocationRiskAgent`
- `CommunicationRiskAgent`
- `AudioRiskAgent`
- `DecisionAgent`

Each agent produces interpretable risk signals, and the `DecisionAgent` combines them into a final fraud score with dataset-aware selection logic.

## Repository Structure

```text
.
├── main.py
├── requirements.txt
├── the-truman-show/
│   └── main.py
├── deus-ex/
│   └── main.py
├── brave-new-world/
│   └── main.py
├── blade-runner/
│   └── main.py
└── 1984/
    └── main.py
```

Dataset folders are intentionally kept out of Git and are expected under:

```text
dataset/
├── the-truman-show/
├── deus-ex/
├── brave-new-world/
├── blade-runner/
└── 1984/
```

## Achievements

- Finished **55th out of 1971 teams**
- Built a reusable fraud detection pipeline across all 5 challenge datasets
- Added dataset-specific tuning while keeping a shared modular architecture
- Integrated Langfuse tracing for observability and session tracking
- Improved leaderboard performance through iterative score-driven tuning

## Key Learnings

- Interpretable agent outputs are much easier to tune than opaque end-to-end scoring.
- Communication signals are powerful, but can easily overfire without safeguards.
- Small, evidence-driven iterations often outperform large rewrites in hackathon settings.
- Dynamic schema inference and defensive parsing are essential when datasets vary in shape and quality.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

Run any dataset wrapper from the repository root:

```bash
python the-truman-show/main.py
python deus-ex/main.py
python brave-new-world/main.py
python blade-runner/main.py
python 1984/main.py
```

Outputs are written to the `outputs/` directory.

## Notes

- `.env` values are expected for Langfuse configuration.
- `dataset/` and generated outputs are excluded from Git through `.gitignore`.
- The shared logic lives in `main.py`; dataset folders provide challenge-specific entrypoints.
