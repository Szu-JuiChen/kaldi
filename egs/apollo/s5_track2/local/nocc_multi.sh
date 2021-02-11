#!/usr/bin/env bash

min=1
max=8
data=
split=
cmd=run.pl

[ -f ./path.sh ] && . ./path.sh; # source the path.
. utils/parse_options.sh || exit 1;

mkdir -p trill/$data

$cmd JOB=$min:$max trill/log/extract_trill.JOB.log \
    python3 local/nocc.py data/${data}/split${split}/JOB/ \
    trill/${data}/trill_${data}_matrix.JOB.txt  || exit 1;

echo "Done generating trill feature."
