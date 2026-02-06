#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python examples/train_phase_1.py

