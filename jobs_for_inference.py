import pandas as pd
import numpy as np
import os
import glob
import time
from datetime import datetime
import argparse

# HF
from datasets import load_dataset, load_metric, Audio
from datasets import Dataset

from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2ForCTC
from transformers import Wav2Vec2FeatureExtractor

# we're using PyTorch
import torch

import oci
from ads import set_auth

# utils.py
from utils import get_objects_list, copy_list_objects_from_oss, copy_object_to_oss
from utils import check_gpu, init_empty_list, get_audio_duration, get_total_duration

#
# Settings
#
DEBUG = True

# the local dir where model files are copied
MODEL_DIR = "model"


# WAV files

# the local dir where all wav files are copied


# get the datetime for the name of output
now = datetime.now().strftime('%Y_%m_%d_%H_%M')

OUT_CSV = f"out_{now}.csv"

SAMPLE_RATE = 16000

#
# Functions
#
def copy_model_oss_2_local(local_dir, namespace, model_bucket, rps):
    list_model_files = ["vocab.json", "config.json", "preprocessor_config.json", "pytorch_model.bin"]

    copy_list_objects_from_oss(list_model_files, local_dir, namespace, model_bucket, rps)

def create_hf_dataset_for_inference(dir_path_name):
    # takes all wav file in the dir and create an hf dataset
    # for inference: sentence columns is full of blank string
    list_wav = sorted(glob.glob(dir_path_name + "/*.wav"))
    list_txts = init_empty_list(len(list_wav))
    
    dict_res = {"path": list_wav, "audio" : list_wav, "sentence": list_txts}

    # create the dataset in HF format
    hf_ds = Dataset.from_dict(dict_res).cast_column("audio", Audio())
    
    return hf_ds

def prepare_dataset(batch):
    audio = batch["audio"]

    # batched output is "un-batched"
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["sentence"]).input_ids
    return batch

def do_inference(index):
    # input is the index in the dataset
    
    input_dict = processor(ds_hf_test[index]["input_values"], return_tensors="pt", padding=True, sampling_rate=SAMPLE_RATE)
    # it is expected to run on GPU (to(cuda))
    
    # added LS 08092022
    with torch.no_grad():
        logits = model(input_dict.input_values.to("cuda")).logits
    pred_ids = torch.argmax(logits, dim=-1)[0]
    pred_text = processor.decode(pred_ids)
    
    return pred_text

def read_command_line():
    parser = argparse.ArgumentParser()

    parser.add_argument("-mb" , "--model_bucket", required=True)
    parser.add_argument("-n" , "--namespace", required=True)
    parser.add_argument("-wd" , "--wav_dir", required=True)
    parser.add_argument("-wb" , "--wav_bucket", required=True)
    parser.add_argument("-ob" , "--out_bucket", required=True)

    args = parser.parse_args()
    
    return args


#
# Main
#

# reading command line arguments
args = read_command_line()

NAMESPACE = args.namespace
MODEL_BUCKET = args.model_bucket
WAV_DIR = args.wav_dir
WAV_BUCKET = args.wav_bucket
OUT_BUCKET = args.out_bucket

print()
print("Starting inference JOBS...")
print()
print("Running with arguments:")
print("NAMESPACE:", NAMESPACE)
print("MODEL_BUCKET:", MODEL_BUCKET)
print("WAV_DIR:", WAV_DIR)
print("WAV_BUCKET:", WAV_BUCKET)
print("OUT_BUCKET:", OUT_BUCKET) 


# check that GPU is available
print("Checking GPU...")
check_gpu()

# when copying file, we need to set security. Now using RP
print()
print("Setting up RPS...")
set_auth(auth='resource_principal')
rps = oci.auth.signers.get_resource_principals_signer()

# create directory MODEL_DIR where we copy model files
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

# copy all model files from bucket to local dir
print()
print("Copy model files to local dir...")
copy_model_oss_2_local(MODEL_DIR, NAMESPACE, MODEL_BUCKET, rps)

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
tot_audio_duration = get_total_duration(WAV_DIR)

# istantiate Model
print()
print("Loading Model in GPU memory...")

tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(MODEL_DIR, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=SAMPLE_RATE, padding_value=0.0, do_normalize=True, return_attention_mask=True)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
# it is expected to run on GPU (to(cuda))
model = Wav2Vec2ForCTC.from_pretrained(MODEL_DIR).to("cuda")

print()
print("Model loaded !!!")

# create the dataset for the inference
print()
print("Creating HF dataset...")
ds_hf_test = create_hf_dataset_for_inference(WAV_DIR)

n_rec = len(ds_hf_test)
print(f"We have {n_rec} records in the dataset.")

# to register the elapsed time
tStart = time.time()

print()
print("Preparing dataset for inference....")
ds_hf_test = ds_hf_test.map(prepare_dataset, remove_columns=ds_hf_test.column_names)

# ---> inference
print()
print("Doing inference...")

# only for formatting
if DEBUG:
    print()
    
out_list_txts = []
for index in range(len(ds_hf_test)):
    res_txt = do_inference(index)
    out_list_txts.append(res_txt)
    
    if DEBUG:
        print("--->" + res_txt)

if DEBUG:
    print()

# register elapsed time
tEla = round(time.time() - tStart, 1)

# preparing output
print("Preparing output...")
out_dict = {"path": list_wav, "sentence": out_list_txts}
out_df = pd.DataFrame(out_dict)
out_df.to_csv(OUT_CSV, index=None)

print("Copy output to Object Storage..")
print(f"Output is copied to {OUT_BUCKET} bucket")
copy_object_to_oss(OUT_CSV, ".", NAMESPACE, OUT_BUCKET, rps)
#
# End, final summary
#
print()
print("Inference completed !!!")
print()
print(f"Processed {n_rec} wav files")
print(f"Total audio duration: {tot_audio_duration} secs.")
print(f"Total elapsed time is: {tEla} secs.")
print()
