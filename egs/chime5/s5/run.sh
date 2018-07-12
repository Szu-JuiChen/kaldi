#!/bin/bash
#
# Based mostly on the TED-LIUM and Switchboard recipe
#
# Copyright  2017  Johns Hopkins University (Author: Shinji Watanabe and Yenda Trmal)
# Apache 2.0
#

# Begin configuration section.
nj=96
decode_nj=20
stage=0
datasize=400
enhancement=beamformit # for a new enhancement method,
                       # change this variable and stage 4
# End configuration section
. ./utils/parse_options.sh

. ./cmd.sh
. ./path.sh


set -e # exit on error

# chime5 main directory path
# please change the path accordingly
chime5_corpus=/export/corpora4/CHiME5
json_dir=${chime5_corpus}/transcriptions
audio_dir=${chime5_corpus}/audio
non_overlap=/export/b02/leibny/chime5/chime5-neural-beamforming/chime5-data-preparation/data/train/speech
#/export/b18/asubraman/chime5_data_preparation/data/train/speech/

# training and test data
train_set=train_worn_u"$datasize"k
test_sets="dev_worn dev_${enhancement}_dereverb_ref dev_addition_dereverb_ref"
# use the below once you obtain the evaluation data. Also remove the comment #eval# in the lines below
#eval#test_sets="dev_worn dev_${enhancement}_ref eval_${enhancement}_ref"

# This script also needs the phonetisaurus g2p, srilm, beamformit
./local/check_tools.sh || exit 1

if [ $stage -le 1 ]; then
  # skip u03 as they are missing
  for mictype in worn u01 u02 u04 u05 u06; do
    local/prepare_data.sh --mictype ${mictype} \
			  ${audio_dir}/train ${json_dir}/train data/train_${mictype}
  done
  #eval#for dataset in dev eval; do
  for dataset in dev; do
    for mictype in worn; do
      local/prepare_data.sh --mictype ${mictype} \
			    ${audio_dir}/${dataset} ${json_dir}/${dataset} \
			    data/${dataset}_${mictype}
    done
  done
fi

if [ $stage -le 2 ]; then
  local/prepare_dict.sh

  utils/prepare_lang.sh \
    data/local/dict "<unk>" data/local/lang data/lang

  local/train_lms_srilm.sh \
    --train-text data/train_worn/text --dev-text data/dev_worn/text \
    --oov-symbol "<unk>" --words-file data/lang/words.txt \
    data/ data/srilm
fi

LM=data/srilm/best_3gram.gz
if [ $stage -le 3 ]; then
  # Compiles G for chime5 trigram LM
  utils/format_lm.sh \
		data/lang $LM data/local/dict/lexicon.txt data/lang

fi

if [ $stage -le 4 ]; then
  # Beamforming using reference arrays
  # enhanced WAV directory
  enhandir=enhan
  #eval#for dset in dev eval; do
  for dset in dev; do
    for mictype in u01 u02 u03 u04 u05 u06; do
      local/run_beamformit.sh --cmd "$train_cmd" \
			      ${audio_dir}/${dset} \
			      ${enhandir}/${dset}_${enhancement}_${mictype} \
			      ${mictype}
    done
  done

  #eval#for dset in dev eval; do
  for dset in dev; do
    local/prepare_data.sh --mictype ref "$PWD/${enhandir}/${dset}_${enhancement}_u0*" \
			  ${json_dir}/${dset} data/${dset}_${enhancement}_ref
  done
fi

if [ $stage -le 5 ]; then
  # remove possibly bad sessions (P11_S03, P52_S19, P53_S24, P54_S24)
  # see http://spandh.dcs.shef.ac.uk/chime_challenge/data.html for more details
  utils/copy_data_dir.sh data/train_worn data/train_worn_org # back up
  grep -v -e "^P11_S03" -e "^P52_S19" -e "^P53_S24" -e "^P54_S24" data/train_worn_org/text > data/train_worn/text
  utils/fix_data_dir.sh data/train_worn
  
  # only use left channel for worn mic recognition
  # you can use both left and right channels for training
  #eval#for dset in train dev eval; do
  for dset in train dev; do
    utils/copy_data_dir.sh data/${dset}_worn data/${dset}_worn_stereo
    grep "\.L-" data/${dset}_worn_stereo/text > data/${dset}_worn/text
    utils/fix_data_dir.sh data/${dset}_worn
  done
  
  # combine mix array and worn mics
  # randomly extract first 100k utterances from all mics
  # if you want to include more training data, you can increase the number of array mic utterances
  utils/combine_data.sh data/train_uall data/train_u01 data/train_u02 data/train_u04 data/train_u05 data/train_u06
  utils/subset_data_dir.sh data/train_uall $(($datasize * 1000)) data/train_u"$datasize"k
  utils/combine_data.sh data/${train_set} data/train_worn data/train_u"$datasize"k
  utils/combine_data.sh data/train_worn_uall data/train_worn data/train_uall
fi

if [ $stage -le 6 ]; then
  # prepare non-overlap data for extracting iVector in run_ivector_common.sh
  for mictype in worn; do
    local/prepare_nonoverlap_data.sh --mictype ${mictype} \
			  ${non_overlap} ${json_dir}/train data/train_non_overlap_${mictype}
  done

  # remove possibly bad sessions (P11_S03, P52_S19, P53_S24, P54_S24)
  utils/copy_data_dir.sh data/train_non_overlap_worn data/train_non_overlap_worn_org # back up
  grep -v -e "^P11_S03" -e "^P52_S19" -e "^P53_S24" -e "^P54_S24" data/train_non_overlap_worn_org/text > data/train_non_overlap_worn/text
  utils/fix_data_dir.sh data/train_non_overlap_worn

  # only use left channel for worn mic recognition
  # you can use both left and right channels for training
  utils/copy_data_dir.sh data/train_non_overlap_worn data/train_non_overlap_worn_stereo
  grep "\.L-" data/train_non_overlap_worn_stereo/text > data/train_non_overlap_worn/text
  utils/fix_data_dir.sh data/train_non_overlap_worn
fi

if [ $stage -le 7 ]; then
  # Split speakers up into 3-minute chunks.  This doesn't hurt adaptation, and
  # lets us use more jobs for decoding etc.
  for dset in train_non_overlap_worn; do #train_worn_uall ${train_set} ${test_sets}
    utils/copy_data_dir.sh data/${dset} data/${dset}_nosplit
    utils/data/modify_speaker_info.sh --seconds-per-spk-max 180 data/${dset}_nosplit data/${dset}
  done
fi

if [ $stage -le 8 ]; then
  # Now make MFCC features.
  # mfccdir should be some place with a largish disk where you
  # want to store MFCC features.
  mfccdir=mfcc
  for x in train_non_overlap_worn; do #train_worn_uall ${train_set} ${test_sets}
    steps/make_mfcc.sh --nj 20 --cmd "$train_cmd" \
		       data/$x exp/make_mfcc/$x $mfccdir
    steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir
    utils/fix_data_dir.sh data/$x
  done
fi
exit 1
if [ $stage -le 9 ]; then
  # make a subset for monophone training
  utils/subset_data_dir.sh --shortest data/${train_set} 100000 data/${train_set}_100kshort
  utils/subset_data_dir.sh data/${train_set}_100kshort 30000 data/${train_set}_30kshort
fi

if [ $stage -le 10 ]; then
  # Starting basic training on MFCC features
  steps/train_mono.sh --nj $nj --cmd "$train_cmd" \
		      data/${train_set}_30kshort data/lang exp/mono
fi

if [ $stage -le 11 ]; then
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
		    data/${train_set} data/lang exp/mono exp/mono_ali

  steps/train_deltas.sh --cmd "$train_cmd" \
			2500 30000 data/${train_set} data/lang exp/mono_ali exp/tri1
fi

if [ $stage -le 12 ]; then
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
		    data/${train_set} data/lang exp/tri1 exp/tri1_ali

  steps/train_lda_mllt.sh --cmd "$train_cmd" \
			  4000 50000 data/${train_set} data/lang exp/tri1_ali exp/tri2
fi

if [ $stage -le 13 ]; then
  utils/mkgraph.sh data/lang exp/tri2 exp/tri2/graph
  for dset in ${test_sets}; do
    steps/decode.sh --nj $decode_nj --cmd "$decode_cmd"  --num-threads 4 \
		    exp/tri2/graph data/${dset} exp/tri2/decode_${dset} &
  done
  wait
fi

if [ $stage -le 14 ]; then
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
		    data/${train_set} data/lang exp/tri2 exp/tri2_ali

  steps/train_sat.sh --cmd "$train_cmd" \
		     5000 100000 data/${train_set} data/lang exp/tri2_ali exp/tri3
fi

if [ $stage -le 15 ]; then
  utils/mkgraph.sh data/lang exp/tri3 exp/tri3/graph
  for dset in ${test_sets}; do
    steps/decode_fmllr.sh --nj $decode_nj --cmd "$decode_cmd"  --num-threads 4 \
			  exp/tri3/graph data/${dset} exp/tri3/decode_${dset} &
  done
  wait
fi

if [ $stage -le 16 ]; then
  # The following script cleans the data and produces cleaned data
  steps/cleanup/clean_and_segment_data.sh --nj ${nj} --cmd "$train_cmd" \
    --segmentation-opts "--min-segment-length 0.3 --min-new-segment-length 0.6" \
    data/${train_set} data/lang exp/tri3 exp/tri3_cleaned data/${train_set}_cleaned
fi

if [ $stage -le 17 ]; then
  # chain TDNN
  local/chain/run_tdnn.sh --nj ${nj} --train-set ${train_set}_cleaned --test-sets "$test_sets" \
    --gmm tri3_cleaned --nnet3-affix _${train_set}_cleaned
  exit 1
fi

if [ $stage -le 18 ]; then
  # Please specify the affix of the TDNN model you want to use below
  affix=1a
  
  # echo "$0: creating high-resolution MFCC features"
  # mfccdir=data/train_worn_uall_hires/data
  # if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
   # utils/create_split_dir.pl /export/b1{5,6,7,8}/$USER/kaldi-data/mfcc/chime5-$(date +'%m_%d_%H_%M')/s5/$mfccdir/storage $mfccdir/storage
  # fi
  # utils/copy_data_dir.sh data/train_worn_uall data/train_worn_uall_hires
  # steps/make_mfcc.sh --nj 10 --mfcc-config conf/mfcc_hires.conf \
      # --cmd "$train_cmd" data/train_worn_uall_hires || exit 1;
  # steps/compute_cmvn_stats.sh data/train_worn_uall_hires || exit 1;
  # utils/fix_data_dir.sh data/train_worn_uall_hires || exit 1;

  # ivectordir=exp/nnet3_${train_set}_cleaned/ivectors_train_worn_uall_hires
  # if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $ivectordir/storage ]; then
    # utils/create_split_dir.pl /export/b0{5,6,7,8}/$USER/kaldi-data/ivectors/chime5-$(date +'%m_%d_%H_%M')/s5/$ivectordir/storage $ivectordir/storage
  # fi
  # temp_data_root=${ivectordir}
  # utils/data/modify_speaker_info.sh --utts-per-spk-max 2 \
    # data/train_worn_uall_hires ${temp_data_root}/train_worn_uall_hires_max2

  # steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj ${nj} \
    # ${temp_data_root}/train_worn_uall_hires_max2 \
    # exp/nnet3_${train_set}_cleaned/extractor $ivectordir

  # utils/copy_data_dir.sh data/train_worn_uall_hires data/train_worn_uall_hires_nosplit
  # utils/data/modify_speaker_info.sh --seconds-per-spk-max 180 data/train_worn_uall_hires_nosplit data/train_worn_uall_hires
  
  #attemp 2
  # local/nnet3/run_ivector_common.sh --stage 3 \
                                    # --train-set train_worn_uall \
                                    # --test-sets "" \
                                    # --gmm tri3 \
                                    # --nnet3-affix _train_worn_uall || exit 1;
  
  # The following scripts cleans all the data using TDNN acoustic model and produces tdnn cleaned data
  local/clean_and_segment_data.sh --nj ${nj} --cmd "$train_cmd" \
    --segmentation-opts "--min-segment-length 0.3 --min-new-segment-length 0.6" \
    data/train_worn_uall_sp_hires data/lang_chain exp/chain_${train_set}_cleaned/tdnn${affix}_sp \
    exp/chain_${train_set}_cleaned/tdnn1a_sp_cleaned data/train_worn_uall_sp_hires_tdnn_cleaned
fi

if [ $stage -le 19 ]; then
  local/rnnlm/run_lstm.sh
  exit 1
fi

if [ $stage -le 20 ]; then
  local/rnnlm/tuning/run_lstm_1b.sh
fi
