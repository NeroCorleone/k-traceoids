#!/bin/bash

SCRIPT="python script-parallel.py"

for i in {1..5}
do
  nohup $SCRIPT >> nohup.out 2>&1 &
  wait $!
done
