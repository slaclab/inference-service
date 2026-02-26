#!/bin/bash

NUM_CLIENTS=${1:-4}
RATE=${2:-1.0}
DURATION=${3:-60}

echo "Starting $NUM_CLIENTS clients at $RATE Hz for $DURATION seconds..."

for i in $(seq 1 $NUM_CLIENTS); do
    python stress_test.py \
        --client-id $i \
        --rate $RATE \
        --duration $DURATION \
        > client_${i}.log 2>&1 &
    echo "Started client $i (PID: $!)"
done

echo "All clients started. Logs: client_*.log"
echo "To stop all: pkill -f stress_test.py"

# Wait for all background jobs
wait

echo "All clients finished"