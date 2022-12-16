#!/bin/bash
echo "... Starts grid search..."
for i in {1..34992..1}
do
  echo $i
  python3 grid_search_features_script.py $i
done