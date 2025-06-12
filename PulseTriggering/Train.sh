#!/bin/bash
# filepath: /home/fedor-tairli/work/CDEs/PulseTriggering/Code/Train.sh


cd /cr/work/tairli/TracesTriggers/Code || exit 1
source /cr/work/tairli/pythonVenv/bin/activate
python3.10 Training.py "$@"
