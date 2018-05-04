#!/bin/bash

# Copyright 2014  Guoguo Chen, 2015 GoVivace Inc. (Nagendra Goel)
#           2017  Vimal Manohar
# Apache 2.0

# This script is similar to steps/cleanup/decode_segmentation.sh, but 
# does decoding using nnet3 model.

set -e
set -o pipefail

# Begin configuration section.
stage=-1
transform_dir=    # dir to find fMLLR transforms.
nj=4 # number of decoding jobs.  If --transform-dir set, must match that number!
acwt=0.1  # Just a default value, used for adaptation and beam-pruning..
post_decode_acwt=1.0  # can be used in 'chain' systems to scale acoustics by 10 so the
                      # regular scoring script works.
cmd=run.pl
beam=15.0
frames_per_chunk=50
max_active=7000
min_active=200
ivector_scale=1.0
lattice_beam=8.0  # Beam we use in lattice generation. We can reduce this if 
                  # we only need the best path
iter=final
num_threads=1 # if >1, will use gmm-latgen-faster-parallel
scoring_opts=
skip_diagnostics=false
skip_scoring=false
allow_partial=true
extra_left_context=0
extra_right_context=0
extra_left_context_initial=-1
extra_right_context_final=-1
online_ivector_dir=
minimize=false
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
   echo "$0: This is a special decoding script for segmentation where we"
   echo "use one decoding graph per segment. We assume a file HCLG.fsts.scp exists"
   echo "which is the scp file of the graphs for each segment."
   echo "This will normally be obtained by steps/cleanup/make_biased_lm_graphs.sh."
   echo ""
   echo "Usage: $0 [options] <graph-dir> <data-dir> <decode-dir>"
   echo " e.g.: $0 --online-ivector-dir exp/nnet3/ivectors_train_si284_split "
   echo "             exp/nnet3/tdnn/graph_train_si284_split \\"
   echo "             data/train_si284_split exp/nnet3/tdnn/decode_train_si284_split"
   echo ""
   echo "where <decode-dir> is assumed to be a sub-directory of the directory"
   echo "where the model is."
   echo ""
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --iter <iter>                                    # Iteration of model to test."
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   echo "  --transform-dir <trans-dir>                      # dir to find fMLLR transforms "
   echo "  --acwt <float>                                   # acoustic scale used for lattice generation "
   echo "  --scoring-opts <string>                          # options to local/score.sh"
   echo "  --num-threads <n>                                # number of threads to use, default 1."
   exit 1;
fi

graphdir=$1
data=$2
dir=$3

mkdir -p $dir/log

if [ -e $dir/$iter.mdl ]; then
  srcdir=$dir
elif [ -e $dir/../$iter.mdl ]; then
  srcdir=$(dirname $dir)
else
  echo "$0: expected either $dir/$iter.mdl or $dir/../$iter.mdl to exist"
  exit 1
fi
model=$srcdir/$iter.mdl

extra_files=
if [ ! -z "$online_ivector_dir" ]; then
  steps/nnet2/check_ivectors_compatible.sh $srcdir $online_ivector_dir || exit 1
  extra_files="$online_ivector_dir/ivector_online.scp $online_ivector_dir/ivector_period"
fi

for f in $data/feats.scp $model $graphdir/HCLG.fsts.scp $extra_files; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

sdata=$data/split$nj;
cmvn_opts=`cat $srcdir/cmvn_opts` || exit 1;
thread_string=
[ $num_threads -gt 1 ] && thread_string="-parallel --num-threads=$num_threads"

mkdir -p $dir/log
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;
echo $nj > $dir/num_jobs

if [ -f $srcdir/phones.txt ]; then
  utils/lang/check_phones_compatible.sh $graph_dir/phones.txt $srcdir/phones.txt
fi

# Split HCLG.fsts.scp by input utterance
n1=$(cat $graphdir/HCLG.fsts.scp | wc -l)
n2=$(cat $data/feats.scp | wc -l)
if [ $n1 != $n2 ]; then
  echo "$0: expected $n2 graphs in $graphdir/HCLG.fsts.scp, got $n1"
fi


mkdir -p $dir/split_fsts
utils/filter_scps.pl --no-warn -f 1 JOB=1:$nj \
  $sdata/JOB/feats.scp $graphdir/HCLG.fsts.scp $dir/split_fsts/HCLG.fsts.JOB.scp
HCLG=scp:$dir/split_fsts/HCLG.fsts.JOB.scp

## Set up features.
echo "$0: feature type is raw"

feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- |"
if [ ! -z "$transform_dir" ]; then # add transforms to features...
  echo "$0: using transforms from $transform_dir"
  [ ! -s $transform_dir/num_jobs ] && \
    echo "$0: expected $transform_dir/num_jobs to contain the number of jobs." && exit 1;
  nj_orig=$(cat $transform_dir/num_jobs)

  if [ ! -f $transform_dir/raw_trans.1 ]; then
    echo "$0: expected $transform_dir/raw_trans.1 to exist (--transform-dir option)"
    exit 1;
  fi
  if [ $nj -ne $nj_orig ]; then
    # Copy the transforms into an archive with an index.
    for n in $(seq $nj_orig); do cat $transform_dir/raw_trans.$n; done | \
       copy-feats ark:- ark,scp:$dir/raw_trans.ark,$dir/raw_trans.scp || exit 1;
    feats="$feats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk scp:$dir/raw_trans.scp ark:- ark:- |"
  else
    # number of jobs matches with alignment dir.
    feats="$feats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark:$transform_dir/raw_trans.JOB ark:- ark:- |"
  fi
elif grep 'transform-feats --utt2spk' $srcdir/log/train.1.log >&/dev/null; then
  echo "$0: **WARNING**: you seem to be using a neural net system trained with transforms,"
  echo "  but you are not providing the --transform-dir option in test time."
fi
##

if [ ! -z "$online_ivector_dir" ]; then
  ivector_period=$(cat $online_ivector_dir/ivector_period) || exit 1;
  ivector_opts="--online-ivectors=scp:$online_ivector_dir/ivector_online.scp --online-ivector-period=$ivector_period"
fi

if [ "$post_decode_acwt" == 1.0 ]; then
  lat_wspecifier="ark:|gzip -c >$dir/lat.JOB.gz"
else
  lat_wspecifier="ark:|lattice-scale --acoustic-scale=$post_decode_acwt ark:- ark:- | gzip -c >$dir/lat.JOB.gz"
fi

frame_subsampling_opt=
if [ -f $srcdir/frame_subsampling_factor ]; then
  # e.g. for 'chain' systems
  frame_subsampling_opt="--frame-subsampling-factor=$(cat $srcdir/frame_subsampling_factor)"
fi

if [ -f "$graphdir/num_pdfs" ]; then
  [ "`cat $graphdir/num_pdfs`" -eq `am-info --print-args=false $model | grep pdfs | awk '{print $NF}'` ] || \
    { echo "Mismatch in number of pdfs with $model"; exit 1; }
fi

if [ $stage -le 1 ]; then
  $cmd --num-threads $num_threads JOB=1:$nj $dir/log/decode.JOB.log \
    nnet3-latgen-faster$thread_string $ivector_opts $frame_subsampling_opt \
     --frames-per-chunk=$frames_per_chunk \
     --extra-left-context=$extra_left_context \
     --extra-right-context=$extra_right_context \
     --extra-left-context-initial=$extra_left_context_initial \
     --extra-right-context-final=$extra_right_context_final \
     --minimize=$minimize --max-active=$max_active --min-active=$min_active --beam=$beam \
     --lattice-beam=$lattice_beam --acoustic-scale=$acwt --allow-partial=$allow_partial \
     --word-symbol-table=$graphdir/words.txt "$model" \
     "$HCLG" "$feats" "$lat_wspecifier" || exit 1;
fi


if [ $stage -le 2 ]; then
  if ! $skip_diagnostics ; then
    [ ! -z $iter ] && iter_opt="--iter $iter"
    steps/diagnostic/analyze_lats.sh --cmd "$cmd" $iter_opt $graphdir $dir
  fi
fi


# The output of this script is the files "lat.*.gz"-- we'll rescore this at
# different acoustic scales to get the final output.
if [ $stage -le 3 ]; then
  if ! $skip_scoring ; then
    [ ! -x local/score.sh ] && \
      echo "$0: Not scoring because local/score.sh does not exist or not executable." && exit 1;
    echo "score best paths"
    [ "$iter" != "final" ] && iter_opt="--iter $iter"
    local/score.sh $iter_opt $scoring_opts --cmd "$cmd" $data $graphdir $dir ||
      { echo "$0: Scoring failed. (ignore by '--skip-scoring true')"; exit 1; }
  fi
fi
echo "Decoding done."
exit 0;
