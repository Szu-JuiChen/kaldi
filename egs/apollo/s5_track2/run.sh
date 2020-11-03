#!/usr/bin/env bash

# Based mostly on the chime6 recipe
#
# Copyright 2020 University of Dallas (Szu-Jui Chen)
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)


# Begin configuration section
stage=0
nj=72
decode_nj=48
decode=true
augment=false
num_data_reps=4
background_snrs="20:10:15:5:0"
foreground_snrs="20:10:15:5:0"

# End configuration section

. ./cmd.sh 
. ./path.sh           
. utils/parse_options.sh  

set -e

FS2_corpus=/scratch2/sxc200004/A11_100hr/FS02_Challenge_Data
audio_dir=${FS2_corpus}/Audio/Segments/ASR_track2
json_dir=${FS2_corpus}/Transcripts/ASR_track2

train_set=train
test_set=dev

# We will need the phonetisaurus g2p, srilm
./local/check_tools.sh || exit 1

if [ $stage -le 0 ]; then
  # data preparation.
  for dataset in Train Dev; do
    local/prepare_data.sh ${audio_dir}/${dataset} ${json_dir}/${dataset} data/${dataset} || exit 1;
  done
  mv data/Train data/train
  mv data/Dev data/dev
fi

if [ $stage -le 1 ]; then
  # prepare LM
  echo "$0: train lm..."
  local/prepare_dict.sh
  
  utils/prepare_lang.sh data/local/dict "<unk>" data/local/lang data/lang

  local/train_lms_srilm.sh \
    --train-text data/train/text --dev-text data/dev/text \
    --oov-symbol "<unk>" --words-file data/lang/words.txt \
    data/ data/srilm
fi
 
LM=data/srilm/best_3gram.gz
if [ $stage -le 2 ]; then
  # Compiles G for FS02 trigram LM
  echo "$0: prepare lang..."
  utils/format_lm.sh data/lang $LM data/local/dict/lexicon.txt data/lang
fi

if [ $stage -le 3 ] && $augment; then
  echo "$0: doing data augmentation with rir"
  if [ ! -d RIRS_NOISES/ ]; then
    # Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
    wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
    unzip rirs_noises.zip
  fi

  # This is the config for the system using simulated RIRs
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")
  steps/data/reverberate_data_dir.py \
    "${rvb_opts[@]}" \
    --prefix "rev" \
    --foreground-snrs $foreground_snrs \
    --backbround-snrs $background_snrs \
    --speech-rvb-probability 1 \
    --pointsource-noise-addition-probability 0 \
    --isotropic-noise-addition-probability 0 \
    --num-replications $num_data_reps \
    --max-noises-per-minute 1 \
    --source-sampling-rate 16000 \
    data/train data/train_rvb

  train_set=train_all
  utils/combine_data.sh data/${train_set} data/train data/train_rvb
fi

if [ $stage -le 4 ]; then
  # Split speakers up into 3-minute chunks.  This doesn't hurt adaptation, and
  # lets us use more jobs for decoding etc.
  for dataset in ${train_set} ${test_set}; do
    utils/copy_data_dir.sh data/${dataset} data/${dataset}_nosplit
    utils/data/modify_speaker_info.sh --seconds-per-spk-max 180 data/${dataset}_nosplit data/${dataset}
  done
fi

##################################################################################
# Now make 13-dim MFCC features. We use 13-dim fetures for GMM-HMM systems.
##################################################################################

if [ $stage -le 5 ]; then
  # Now make MFCC features.
  # mfccdir should be some place with a largish disk where you want to store MFDD features.
  echo "$0: make features..."
  mfccdir=mfcc
  for x in ${train_set} ${test_set}; do
    steps/make_mfcc.sh --nj 20 --cmd "$train_cmd" \
        data/$x exp/make_mfcc/$x $mfccdir
    steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir
    utils/fix_data_dir.sh data/$x
  done
fi

###################################################################################
# Stages 6 to 10 train monophone and triphone models. They will be used for
# generating lattices for training the chain model
###################################################################################

if [ $stage -le 6 ]; then
  utils/subset_data_dir.sh --shortest data/${train_set} 10000 data/${train_set}_10kshort

  steps/train_mono.sh --nj $nj --cmd "$train_cmd" \
      data/${train_set}_10kshort data/lang exp/mono
fi

if [ $stage -le 7 ]; then
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
      data/${train_set} data/lang exp/mono exp/mono_ali
  steps/train_deltas.sh --cmd "$train_cmd" \
      2500 30000 data/${train_set} data/lang exp/mono_ali exp/tri1

fi

if [ $stage -le 8 ]; then
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
      data/${train_set} data/lang exp/tri1 exp/tri1_ali
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
      4000 50000 data/${train_set} data/lang exp/tri1_ali exp/tri2
  if $decode; then
    (
      utils/mkgraph.sh data/lang exp/tri2 exp/tri2/graph || exit 1;
      for data in ${test_set}; do
        steps/decode.sh --cmd "$decode_cmd" --nj $decode_nj \
            exp/tri2/graph data/$data exp/tri2/decode_${data}
      done
    )&
  fi
fi

if [ $stage -le 9 ]; then
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
      data/${train_set} data/lang exp/tri2 exp/tri2_ali
  steps/train_sat.sh --cmd "$train_cmd" \
      5000 100000 data/${train_set} data/lang exp/tri2_ali exp/tri3
  if $decode; then
    utils/mkgraph.sh data/lang exp/tri3 exp/tri3/graph || exit 1;
    for data in ${test_set}; do
      steps/decode_fmllr.sh --cmd "$decode_cmd" --nj $decode_nj \
          exp/tri3/graph data/$data exp/tri3/decode_${data}
    done
  fi
fi

exit 0;
#### preparing ####
if [ $stage -le 5 ]; then
  # Estimate pronunciation and silence probabilities.

  # Silprob for normal lexicon.
  steps/get_prons.sh --cmd "$train_cmd" \
    data/train_si284 data/lang_nosp exp/tri3b || exit 1;
  utils/dict_dir_add_pronprobs.sh --max-normalize true \
    data/local/dict_nosp \
    exp/tri3b/pron_counts_nowb.txt exp/tri3b/sil_counts_nowb.txt \
    exp/tri3b/pron_bigram_counts_nowb.txt data/local/dict || exit 1

  utils/prepare_lang.sh data/local/dict \
    "<SPOKEN_NOISE>" data/local/lang_tmp data/lang || exit 1;
fi

exit 0;
