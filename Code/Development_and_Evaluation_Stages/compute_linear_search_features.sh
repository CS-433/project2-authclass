#!/bin/bash
echo "... Starts feature paramater linear search ..."
for i in {1..21..1}
do
  python3 linear_search_script.py $i
  echo "... Next one ..."
done