#!/bin/bash

script_name="script.py"

pids=$(pgrep -f "$script_name")

if [ -z "$pids" ]; then
    echo "No processes found running $script_name"
else
    echo "Killing the following processes: $pids"
    kill $pids
fi
