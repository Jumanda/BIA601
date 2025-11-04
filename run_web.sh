#!/bin/bash
# Script to run the web application

cd "$(dirname "$0")"
source venv/bin/activate
export PYTHONPATH=.
python examples/run_web_app.py

