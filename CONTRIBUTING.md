# Contributing to Reticus

Thanks for considering a contribution. Reticus is a small, focused tool - keep that in
mind when proposing changes: prefer fixing a real bug or filling a real gap over adding
speculative features.

## Reporting a bug

Open an issue with:
- What you ran (a minimal script reproducing the issue, if possible)
- What you expected vs. what happened
- Your OS, Python version, and which extras you installed (`pip install ".[?]"` or `pip install -e ".[?]"`)
- Any traceback, in full

## Requesting a feature

Open an issue describing the concrete problem you're hitting, not just the feature -
it's easier to evaluate "I can't tell which epoch overfitting started at" than
"add more charts."

## Development setup

```bash
git clone <this-repo>
cd reticus
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[all]"
pytest
```

Frontend (only needed if you're touching the dashboard UI):

```bash
cd frontend
npm install
npm run dev     # local dev server with hot reload
npm run build   # rebuilds reticus/dashboard/dist/ - must be committed
```

The dashboard is pre-built and shipped inside the Python package
(`reticus/dashboard/dist/`), so a normal `pip install` never needs Node.js - only
contributors editing the frontend do.

## Before submitting a PR

- Run the full test suite (`pytest`) and make sure it passes.
- If you touched the frontend, run `npm run build` and commit the rebuilt
  `reticus/dashboard/dist/` output alongside your source changes.
- Keep changes scoped - a bug fix shouldn't also refactor unrelated code.
- Add a test for any bug fix or new behavior where practical.

## Project philosophy

Reticus is deliberately local-first and privacy-first: no telemetry, no accounts, no
network calls to anywhere but `127.0.0.1`. PRs that would introduce external service
dependencies, analytics, or cloud sync are out of scope for this project.
