#!/bin/bash

SCRIPT="python script.py"

for i in {1..5}
do
  $SCRIPT &
  wait $!
done
