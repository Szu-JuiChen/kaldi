# Import TF 2.X and make sure we're running eager.
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
assert tf.executing_eagerly()

import tensorflow_hub as hub
import soundfile as sf
import numpy as np
import math
import sys
import io
from subprocess import PIPE, run
import pdb
# Load the module and run inference.
module = hub.load('https://tfhub.dev/google/nonsemantic-speech-benchmark/trill/3')

# set tf logging leve
tf.get_logger().setLevel('ERROR')

# `wav_as_float_or_int16` can be a numpy array or tf.Tensor of float type or
# int16. The sample rate must be 16kHz. Resample to this sample rate, if
# necessary.
def extract_embedding(wav_path):
    wav_result = run(wav_path, shell=True, stdout=PIPE, stderr=PIPE).stdout
    data, samplerate = sf.read(io.BytesIO(wav_result), dtype='float32')

    # data should be wav_as_float_or_int16
    emb_dict = module(samples=data, sample_rate=samplerate)

    # For a description of the difference between the two endpoints, please see the
    # paper (https://arxiv.org/abs/2002.12764), section "Neural Network Layer".
    # We choose 'embedding' here for smaller dimension (512 vs 12288)
    emb = emb_dict['embedding']
    #emb_layer19 = emb_dict['layer19']

    # Embeddings are a [time, feature_dim] Tensors.
    emb.shape.assert_is_compatible_with([None, 512])
    #emb_layer19.shape.assert_is_compatible_with([None, 12288])
  
    #return np.mean(emb, axis=0)
    return emb.numpy()

def main():
    num_frame_dict = {}
    for line in open(sys.argv[1] + '/utt2num_frames', 'r', encoding='utf8'):
        line = line.split()
        # key is utt and val is num_frames
        num_frame_dict[line[0]] = int(line[1])

    emb_dict = {}
    cnt = 0
    for line in open(sys.argv[1] + '/wav.scp', 'r', encoding='utf8'):
        line = line.split()
        utt = line[0]
        cmd = ' '.join(line[1:])
        if 'sox' in cmd:
            cmd += ' sox -t wav -r 8000 - -t wav -r 16000 -'
        else:
            cmd = 'sox -t wav -r 8000 ' + cmd + ' -t wav -r 16000 -'
        emb_raw = extract_embedding(cmd)
        
        # duplicate row of emb to match with rows in ivector
        # we find the multiple of n_emb closest to n_ivector using the method in the below link:
        # (https://www.geeksforgeeks.org/multiple-of-x-closest-to-n/)
        n_frame = num_frame_dict[utt]
        n_ivector = math.ceil(n_frame/10)
        n_emb = len(emb_raw)
        closest = n_ivector + n_emb//2 # the second term is floor of n_emb
        closest = closest - (closest % n_emb)
        factor = closest / n_emb
        # we then add/remove rows if n_ivector != closest
        if n_ivector > closest:
            diff = n_ivector - closest
            rep = np.full(n_emb, factor, dtype=int)
            rep[-diff:] = np.full(diff, factor+1, dtype=int)
            emb_dict[utt] = np.repeat(emb_raw, repeats=rep, axis=0)
        elif n_ivector < closest:
            diff = closest - n_ivector
            rep = np.full(n_emb, factor, dtype=int)
            rep[-diff:] = np.full(diff, factor-1, dtype=int)
            emb_dict[utt] = np.repeat(emb_raw, repeats=rep, axis=0) 
        else:
            emb_dict[utt] = np.repeat(emb_raw, repeats=factor, axis=0)
        cnt += 1
        print('processing: ', cnt)

    with open(sys.argv[2], 'w', encoding='utf8') as f:
        for key,val in emb_dict.items():
            #str_val = np.array2string(val, formatter={'float_kind':lambda x: "%.8f" % x})
            #str_val = str_val.replace('\n', '')
            #str_val = str_val.replace('[', '[  ')
            #str_val = str_val.replace(']', ' ]\n')
            #out = key + '  ' + str_val
            f.write(key + '  [\n')
            mat = np.matrix(val)
            for line in mat:
                np.savetxt(f, line, fmt='%.8f')
            f.write(']\n')
    # fix format
    s = r"sed -i ':r;$!{N;br};s/\n]/ ]/g' " + sys.argv[2]
    run(s, shell=True, stdout=PIPE, stderr=PIPE)

if __name__ == '__main__':
    main()
