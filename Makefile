# ============================================================
# Satellite Image Classification — fMoW Pipeline
# Usage: make all
# ============================================================

.PHONY: all download preprocess features select train evaluate report clean

all: download preprocess features select train evaluate report
	@echo ""
	@echo "✅ Full pipeline complete!"

# ── Part 1: Download dataset ──────────────────────────────────
download:
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  Part 1 — Dataset Download"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	.venv/bin/python download_fmow.py

# ── Part 2: Preprocessing ─────────────────────────────────────
preprocess:
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  Part 2 — Preprocessing"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	.venv/bin/python preprocessing.py

# ── Part 3: Feature Engineering ───────────────────────────────
features:
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  Part 3 — Feature Engineering"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	.venv/bin/python feature_engineering.py

# ── Part 4: Feature Selection ─────────────────────────────────
select:
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  Part 4 — Feature Selection"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	.venv/bin/python feature_selection.py

# ── Part 5: Model Training ────────────────────────────────────
train:
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  Part 5 — Model Training"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	.venv/bin/python model_training.py

# ── Part 6: Model Evaluation ──────────────────────────────────
evaluate:
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  Part 6 — Model Evaluation"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	.venv/bin/python model_evaluate.py

# ── Part 7: Report ────────────────────────────────────────────
report:
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  Part 7 — Analysis & Report"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	.venv/bin/python report.py

# ── Run from specific step ────────────────────────────────────
from-preprocess: preprocess features select train evaluate report
from-features:   features select train evaluate report
from-select:     select train evaluate report
from-train:      train evaluate report
from-evaluate:   evaluate report

# ── Clean generated folders (keeps raw data) ──────────────────
clean:
	@echo "Cleaning generated folders..."
	rm -rf fmow_preprocessed fmow_features fmow_selected fmow_models fmow_evaluation
	@echo "✅ Cleaned"

# ── Clean everything including raw data ───────────────────────
clean-all:
	@echo "Cleaning everything..."
	rm -rf fmow_data fmow_preprocessed fmow_features fmow_selected fmow_models fmow_evaluation
	@echo "✅ Cleaned all"