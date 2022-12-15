#!/bin/bash
echo "... Starts feature paramater linear search ..."
for i in {1..18..1}
do
  python3 hyperparameter_script.py $i
  echo "... Next one ..."
done