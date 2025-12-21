#!/bin/bash

# Name of the Python script to search for
script_name="script.py"

# Find the process IDs (PIDs) of all running Python processes with the specific script
pids=$(pgrep -f "$script_name")

# Check if any processes were found
if [ -z "$pids" ]; then
    echo "No processes found running $script_name"
else
    # Kill the processes
    echo "Killing the following processes: $pids"
    kill $pids
fi
