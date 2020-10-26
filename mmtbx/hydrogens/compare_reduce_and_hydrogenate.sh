#!/bin/bash

# Create output directory
OD="$1"
echo "Making output directory $OD"
mkdir -p $OD

# Create log file telling about the run
OF="$OD/log.txt"
echo "Comparison log file" > "$OF"
echo >> "$OF"

echo "Building from updated source"
(cd ~/src/cctbx_project; git pull)
echo "Hydrogenate commit" >> "$OF"
(cd ~/src/cctbx_project; git log | head -3) >> "$OF"
echo >> "$OF"
(cd ~/rlab/cctbx; python3 ~/src/cctbx_project/libtbx/auto_build/bootstrap.py --use-conda --nproc 8 --python 36)

echo "Reduce version" >> "$OF"
~/build/reduce/reduce_src/reduce -version 2>> "$OF"
echo >> "$OF"

echo "Running comparisons on each file"
files=$(cd inputs; ls *.pdb)
for f in $files
do
  # Copy the file to the output directory
  echo "  $f"
  cp inputs/$f "$OD"

  # Run Reduce
  ~/build/reduce/reduce_src/reduce -DROP_HYDROGENS_ON_ATOM_RECORDS -DROP_HYDROGENS_ON_OTHER_RECORDS -NOOPT "inputs/$f" 2> "$OD"/"$f"_reduceDROPADDNOOPT.txt 1> "$OD"/"$f"_reduceDROPADDNOOPT.pdb
  ~/build/reduce/reduce_src/reduce -DROP_HYDROGENS_ON_ATOM_RECORDS -DROP_HYDROGENS_ON_OTHER_RECORDS "inputs/$f" 2> "$OD"/"$f"_reduceDROPADDOPT.txt 1> "$OD"/"$f"_reduceDROPADDOPT.pdb

  # Run Hydrogenate
  (cd "$OD"; ~/rlab/cctbx/build/bin/mmtbx.hydrogenate "$f" 2> "$f"_hydrogenate_errors.txt 1> "$f"_hydrogenate.txt)

done

