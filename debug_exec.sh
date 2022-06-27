#!/bin/bash
SCRIPT=$(realpath $0)
THISDIR=$(dirname $SCRIPT)
PARAMS="${@:1}"
#echo $THISDIR
#echo $PARAMS

source $THISDIR/venv/bin/activate
python3 $THISDIR/autocutmarks.py $PARAMS
