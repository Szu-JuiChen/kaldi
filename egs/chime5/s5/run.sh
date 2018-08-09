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
conf_dir=/export/b05/cszu/conf
noise_dir=/export/b05/cszu/noise
non_overlap_data=/export/b02/leibny/chime5/chime5-neural-beamforming/chime5-data-preparation

# training and test data
train_set=train_worn_simu_u"$datasize"k
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
  for dset in dev eval; do
    for mictype in u01 u02 u03 u04 u05 u06; do
      local/run_beamformit.sh --cmd "$train_cmd" \
			      ${audio_dir}/${dset} \
			      ${enhandir}/${dset}_${enhancement}_${mictype} \
			      ${mictype}
    done
  done
  
  for dset in dev; do
    local/prepare_data.sh --mictype ref "$PWD/${enhandir}/${dset}_${enhancement}_u0*" \
			  ${json_dir}/${dset} data/${dset}_${enhancement}_ref
  done
fi

if [ $stage -le 5 ]; then
  # Prepare data for reverberant speech
  # augmented WAV directory
  for set in train; do
    for mictype in aug; do
      local/prepare_data.sh --mictype aug --rn 1 --cdir ${conf_dir} \
                            --ndir ${noise_dir}/${set} \
                            ${audio_dir}/${set} ${json_dir}/${set} \
                            data/${set}_${mictype}
    done
  done
  # prepare non-overlapped training data for extracting iVector in run_ivector_common.sh
  # first obtain the non-overlapped segment information
  # worn data
  for file in $(find ${non_overlap_data}_worn/data/train/ -maxdepth 1 -name "segments_S[0-9]*"); do
    grep "S[0-9]*_P[0-9]" $file | sort >> segments.tmp_train_worn
  done
  awk -F " " '{print $2}' segments.tmp_train_worn | awk -F "_" '{print $2"_"$1"_NOLOCATION.L-"$3"-"$4}{print $2"_"$1"_NOLOCATION.R-"$3"-"$4}' | awk -F "." '{print $1"."$2}' > segments.tmp_train_worn2
  # array data
  for file in $(find ${non_overlap_data}_array/data/train -maxdepth 1 -name "segments_S[0-9]*"); do
    grep "S[0-9]*_U[0-9].*P[0-9]" $file | sort >> segments.tmp_train_array
  done
  awk -F " " '{print $2}' segments.tmp_train_array | awk -F'[_.]' '{print $5"_"$1"_"$2"_NOLOCATION."$7"-"$3"-"$4}' | sort > segments.tmp_train_array2
  # prepare non-overlapped dev data
  # worn data
  for file in $(find ${non_overlap_data}_worn/data/dev/ -maxdepth 1 -name "segments_S[0-9]*"); do
    grep "S[0-9]*_P[0-9]" $file | sort >> segments.tmp_dev_worn
  done
  awk -F " " '{print $2}' segments.tmp_dev_worn | awk -F'[_.]' '{print $2"_"$1"_"toupper($6)".L-"$3"-"$4}{print $2"_"$1"_"toupper($6)".R-"$3"-"$4}' | sort > segments.tmp_dev_worn2
  # array data
  for file in $(find ${non_overlap_data}_array/data/dev -name "segments_S[0-9]*"); do
    grep "S[0-9]*_U[0-9].*P[0-9]*.[a-z]" $file | sort >> segments.tmp_dev_array
  done
  awk -F " " '{print $2}' segments.tmp_dev_array | awk -F'[_.]' '{print $5"_"$1"_"$2"_"toupper($6)".ENH-"$3"-"$4}' | sort > segments.tmp_dev_array2
fi

if [ $stage -le 6 ]; then
  # remove possibly bad sessions (P11_S03, P52_S19, P53_S24, P54_S24)
  # see http://spandh.dcs.shef.ac.uk/chime_challenge/data.html for more details
  for type in worn; do #aug
    utils/copy_data_dir.sh data/train_${type} data/train_${type}_org # back up
    grep -v -e "^P11_S03" -e "^P52_S19" -e "^P53_S24" -e "^P54_S24" data/train_${type}_org/text > data/train_${type}/text
    utils/fix_data_dir.sh data/train_${type}
  done
  
  # combine mix array and worn mics
  # randomly extract first 100k utterances from all mics
  # if you want to include more training data, you can increase the number of array mic utterances
  utils/combine_data.sh data/train_uall data/train_u01 data/train_u02 data/train_u04 data/train_u05 data/train_u06
  utils/subset_data_dir.sh data/train_uall $(($datasize * 1000)) data/train_u"$datasize"k
  utils/subset_data_dir.sh data/train_aug $(($datasize * 1000)) data/train_aug"$datasize"k
  utils/combine_data.sh data/${train_set} data/train_worn data/train_u"$datasize"k data/train_aug"$datasize"k
  utils/combine_data.sh data/train_worn_uall data/train_worn data/train_uall
  
  # non-overlapped data extraction
  utils/subset_data_dir.sh --utt-list segments.tmp_train_worn2 data/train_worn data/train_non_overlap_worn
  utils/subset_data_dir.sh --utt-list segments.tmp_train_array2 data/train_uall data/train_non_overlap_uall
  utils/subset_data_dir.sh --utt-list segments.tmp_dev_worn2 data/dev_worn data/dev_worn_non_overlap
  utils/subset_data_dir.sh --utt-list segments.tmp_dev_array2 data/dev_beamformit_dereverb_ref data/dev_beamformit_dereverb_ref_non_overlap
  utils/combine_data.sh data/train_non_overlap data/train_non_overlap_worn data/train_non_overlap_uall data/train_aug"$datasize"k
  rm -f segments.*
  # only use left channel for worn mic recognition
  # you can use both left and right channels for training
  #eval#for dset in train dev eval; do
  for dset in train dev; do
    utils/copy_data_dir.sh data/${dset}_worn data/${dset}_worn_stereo
    grep "\.L-" data/${dset}_worn_stereo/text > data/${dset}_worn/text
    utils/fix_data_dir.sh data/${dset}_worn
  done
fi

if [ $stage -le 7 ]; then
  # fix speaker ID issue (thanks to Dr. Naoyuki Kanda)
  # add array ID to the speaker ID to avoid the use of other array information to meet regulations
  # Before this fix
  # $ head -n 2 data/eval_beamformit_ref_nosplit/utt2spk
  # P01_S01_U02_KITCHEN.ENH-0000192-0001278 P01
  # P01_S01_U02_KITCHEN.ENH-0001421-0001481 P01
  # After this fix
  # $ head -n 2 data/eval_beamformit_ref_nosplit_fix/utt2spk
  # P01_S01_U02_KITCHEN.ENH-0000192-0001278 P01_U02
  # P01_S01_U02_KITCHEN.ENH-0001421-0001481 P01_U02
  for dset in dev_${enhancement}_dereverb_ref dev_addition_dereverb_ref dev_${enhancement}_dereverb_ref_non_overlap; do #eval_${enhancement}_ref
    utils/copy_data_dir.sh data/${dset} data/${dset}_nosplit
    mkdir -p data/${dset}_nosplit_fix
    cp data/${dset}_nosplit/{segments,text,wav.scp} data/${dset}_nosplit_fix/
    awk -F "_" '{print $0 "_" $3}' data/${dset}_nosplit/utt2spk > data/${dset}_nosplit_fix/utt2spk
    utils/utt2spk_to_spk2utt.pl data/${dset}_nosplit_fix/utt2spk > data/${dset}_nosplit_fix/spk2utt
  done
  # Split speakers up into 3-minute chunks.  This doesn't hurt adaptation, and
  # lets us use more jobs for decoding etc.
  for dset in dev_worn_non_overlap; do #train_non_overlap ${train_set} dev_worn
    utils/copy_data_dir.sh data/${dset} data/${dset}_nosplit
    utils/data/modify_speaker_info.sh --seconds-per-spk-max 180 data/${dset}_nosplit data/${dset}
  done
  for dset in dev_${enhancement}_dereverb_ref dev_${enhancement}_dereverb_ref_non_overlap dev_addition_dereverb_ref eval_${enhancement}_ref; do
    utils/data/modify_speaker_info.sh --seconds-per-spk-max 180 data/${dset}_nosplit_fix data/${dset}
  done
fi

if [ $stage -le 8 ]; then
  # Now make MFCC features.
  # mfccdir should be some place with a largish disk where you
  # want to store MFCC features.
  mfccdir=mfcc
  for x in dev_worn_non_overlap dev_${enhancement}_dereverb_ref_non_overlap; do #train_non_overlap train_worn_uall ${train_set} ${test_sets}
    steps/make_mfcc.sh --nj 20 --cmd "$train_cmd" \
		       data/$x exp/make_mfcc/$x $mfccdir
    steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir
    utils/fix_data_dir.sh data/$x
  done
fi

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
  local/chain/run_tdnn_lstm.sh --nj ${nj} --train-set ${train_set}_cleaned --test-sets "$test_sets" \
    --gmm tri3_cleaned --nnet3-affix _${train_set}_cleaned
fi

if [ $stage -le 18 ]; then
  # Please specify the affix of the TDNN model you want to use below
  affix=1a
  local/nnet3/run_ivector_common.sh --train-set train_worn_uall \
                                    --test-sets "" \
                                    --non-overlap "" \
                                    --non_overlap_test "" \
                                    --gmm tri3 \
                                    --nnet3-affix _train_worn_uall || exit 1;
  
  # The following scripts cleans all the data using TDNN acoustic model and produces tdnn cleaned data
  local/clean_and_segment_data.sh --nj ${nj} --cmd "$train_cmd" \
    --segmentation-opts "--min-segment-length 0.3 --min-new-segment-length 0.6" \
    data/train_worn_uall_sp_hires data/lang_chain exp/chain_${train_set}_cleaned/tdnn${affix}_sp \
    exp/chain_${train_set}_cleaned/tdnn1a_sp_cleaned data/train_worn_uall_sp_hires_tdnn_cleaned
fi

if [ $stage -le 19 ]; then
  local/rnnlm/run_lstm.sh
fi

if [ $stage -le 20 ]; then
  # final scoring to get the official challenge result
  # please specify both dev and eval set directories so that the search parameters
  # (insertion penalty and language model weight) will be tuned using the dev set
  local/score_for_submit.sh \
      --dev exp/chain_${train_set}_cleaned/tdnn1a_sp/decode_dev_${enhancement}_ref \
      --eval exp/chain_${train_set}_cleaned/tdnn1a_sp/decode_eval_${enhancement}_ref
fi
