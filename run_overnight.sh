#!/bin/bash
# Activate virtual environment if needed
source .venv/bin/activate
# Run FinRL overnight with logging and background execution
nohup python -m finrl.test > overnight_run.log 2>&1 &
echo "FinRL overnight run started. Check overnight_run.log for progress."
