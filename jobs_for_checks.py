import pandas as pd
import numpy as np
import os
import glob
import time
import argparse

from utils import get_objects_list, copy_list_objects_from_oss, copy_object_to_oss
from utils import check_gpu, get_audio_duration, get_total_duration
from utils import check_sample_rate, check_wav, check_mono, check_pcm

import oci
from ads import set_auth

DEBUG = True

SAMPLE_RATE = 16000

#
# Functions
#
def read_command_line():
    parser = argparse.ArgumentParser()

    parser.add_argument("-n" , "--namespace", required=True)
    parser.add_argument("-wd" , "--wav_dir", required=True)
    parser.add_argument("-wb" , "--wav_bucket", required=True)

    args = parser.parse_args()
    
    return args

#
# Main
#
args = read_command_line()

NAMESPACE = args.namespace
WAV_DIR = args.wav_dir
WAV_BUCKET = args.wav_bucket


print()
print("Starting checks JOBS...")
print()
print("Running with arguments:")
print("NAMESPACE:", NAMESPACE)
print("WAV_DIR:", WAV_DIR)
print("WAV_BUCKET:", WAV_BUCKET)


# check that GPU is available
print("Checking GPU...")
check_gpu()

# when copying file, we need to set security. Now using RP
print()
print("Setting up RPS...")
set_auth(auth='resource_principal')
rps = oci.auth.signers.get_resource_principals_signer()

# copy all wav files to WAV_DIR
print()
print("Copy wav files to local dir...")

if not os.path.exists(WAV_DIR):
    os.mkdir(WAV_DIR)
else:
    # remove all wav files from dir before copy
    print("Cleaning WAV_DIR...")
    for f in glob.glob(WAV_DIR + "/*.wav"):
        os.remove(f)

# filtra solo i files wav
tmp_list_wav = get_objects_list(NAMESPACE, WAV_BUCKET, rps)
list_wav = []
for f_name in tmp_list_wav:
    if f_name.split(".")[-1] == "wav":
        list_wav.append(f_name)
        
copy_list_objects_from_oss(list_wav, WAV_DIR, NAMESPACE, WAV_BUCKET, rps)

# compute the total audio duration
n_rec = len(list_wav)
tot_audio_duration = get_total_duration(WAV_DIR)

# to register the elapsed time
tStart = time.time()

#
# All checks here

# build the list with path
list_wav = glob.glob(WAV_DIR + "/*.wav")

print()
print("Checking only files with wav extension...")
print("Checks results:")
check_wav(list_wav)
check_sample_rate(list_wav, SAMPLE_RATE)
check_mono(list_wav)
check_pcm(list_wav)

#
# register elapsed time
tEla = round(time.time() - tStart, 1)
#
# End, final summary
#
print()
print("All checks completed !!!")
print()
print(f"Processed {n_rec} wav files")
print(f"Total audio duration: {tot_audio_duration} secs.")
print(f"Total elapsed time is: {tEla} secs.")
print()
