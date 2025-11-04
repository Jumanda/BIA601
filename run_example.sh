#!/bin/bash
# Script to run the example with correct Python path

cd "$(dirname "$0")"
source venv/bin/activate
export PYTHONPATH=.
python examples/example_usage.py

