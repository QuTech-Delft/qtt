#!/bin/bash
#Collect all notebooks (except the ones in the measurements and unfinished folder) to be executed
#One or more notebooks from the measurements folder or others can be added manually by copying, uncommenting
#and adjusting the filename of the following line and appending it to the bottom of this script.
#echo "jupyter nbconvert --to notebook --execute docs/notebooks/measurements/some_notebook_to_add.ipynb"

echo "Running notebooks"
set -ev
export QTT_UNITTEST=ON
find docs/notebooks/ -type d \( -path "docs/notebooks/measurements" -o -path "docs/notebooks/unfinished" \) -prune -o -type f -name "*.ipynb" -print0 | while IFS= read -r -d $'\0' file;
do
    jupyter nbconvert --to notebook --execute "$file"
done
