#!/bin/bash
#Collect all notebooks (except the ones in the measurements and unfinished folder) to be executed
#One or more notebooks from the measurements folder or others can be added manually by copying, uncommenting
#and adjusting the filename of the following line and appending it to the bottom of this script.
#echo "jupyter nbconvert --to notebook --execute docs/notebooks/measurements/some_notebook_to_add.ipynb" >>$OUTFILE

echo "Collecting notebooks to run"
OUTFILE="docs/notebooks/run_selected_notebooks.sh"
/bin/cat <<EOM >$OUTFILE
# Generated file, do not edit it or put it in github
# Run the notebooks
echo "Running notebooks"
set -ev
EOM
find docs/notebooks/ -type d \( -path "docs/notebooks/measurements" -o -path "docs/notebooks/unfinished" \) -prune -o -type f -name "*.ipynb" -print0 | while IFS= read -r -d $'\0' file;
do
    echo "$file"
    echo "jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=600 $file" >>$OUTFILE
done
