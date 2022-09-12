import oci
from ads import set_auth
import pandas as pd
import os
import soundfile as sf
import re
import torch
import glob

# to get some info about audio format
def get_audio_channels(path_name):
    info_obj = sf.info(path_name)
    
    return info_obj.channels

def get_audio_format(path_name):
    info_obj = sf.info(path_name)
    
    return info_obj.format

def is_wav(path_name):
    if get_audio_format(path_name) == "WAV":
        v_rit = True
    else:
        v_rit = False
    
    return v_rit

# return the audio duration in secs
def get_audio_duration(path_name):
    info_obj = sf.info(path_name)
    
    # lets trunc to 1 decimal digit
    return round(info_obj.duration, 1)

def get_total_duration(dir_path_name):
    list_wav = sorted(glob.glob(dir_path_name + "/*.wav"))
    
    tot_dur = 0.
    for f_name in list_wav:
        tot_dur += get_audio_duration(f_name)
        
    return round(tot_dur, 1)

def check_wav(list_wav):
    
    num_except = 0
    
    for file_wav in list_wav:
        if is_wav(file_wav) != True:
            num_except += 1
            print(f"{file_wav}: is not of WAV format")

    if num_except == 0:
        print(f"All files have the expected WAV format.")
        
def check_sample_rate(list_wav, ref_sample_rate):
    
    num_except = 0
    
    for file_wav in list_wav:
        data, sample_rate = sf.read(file_wav)
    
        if sample_rate != ref_sample_rate:
            num_except += 1
            print(f"{file_wav}: is not OK, sample_rate: {sample_rate}")

    if num_except == 0:
        print(f"All wav files have sample rate = {ref_sample_rate}.")

# to check that all files are MONO
def check_mono(list_wav):
    MONO = 1
    
    num_except = 0
    
    for file_wav in list_wav:
        num_channels = get_audio_channels(file_wav)
    
        if num_channels != MONO:
            num_except += 1
            print(f"{file_wav} is not MONO")

    if num_except == 0:
        print(f"All wav files are MONO.")

def check_pcm(list_wav):
    
    num_except = 0
    
    for file_wav in list_wav:
        if sf.info(file_wav).subtype != "PCM_16":
            num_except += 1
            print(f"{file_wav}: is not of PCM_16 format")

    if num_except == 0:
        print(f"All files have the expected PCM_16 format.")
        
def check_files_exists(list_files):
    not_existing = 0

    for f_name in list_files:
        if not os.path.exists(f_name):
            not_existing += 1
            print(f"File {f_name} is not available...")
        
    if not_existing > 0:
        print(f"There are {not_existing} missing files!")
    else:
        print()
        print("All required wav files are available!")

def init_empty_list(n_elem):
    v_list = []
    
    for i in range(n_elem):
        v_list.append("")
        
    return v_list

def check_gpu():
    assert torch.cuda.is_available()

    print("GPU is available, OK")

# get the list of objects in the bucket
# works for num_objects <= 1000
# need to add pagination
def get_objects_list(namespace, bucket_name, rps):
    object_storage = oci.object_storage.ObjectStorageClient(config={}, signer=rps)

    resp = object_storage.list_objects(namespace, bucket_name)

    # extract only the names
    list_files = []
    
    if len(resp.data.objects) > 0:
        for obj in resp.data.objects:
            list_files.append(obj.name)
    
    return list_files

    # copy from bucket to local dir
def copy_object_from_oss(f_name, dir_name, namespace, bucket_name, rps):
    CHUNK_SIZE = 1024 * 1024
    
    object_storage = oci.object_storage.ObjectStorageClient(config={}, signer=rps)
    
    get_obj = object_storage.get_object(namespace, bucket_name, f_name)
    
    path_name = dir_name + "/" + f_name
    
    with open(path_name, 'wb') as f:
        for chunk in get_obj.data.raw.stream(CHUNK_SIZE, decode_content=False):
            f.write(chunk)
            
    print(f"Copy {f_name} done!")
            
# copy a file from a local dir to a bucket of OSS
def copy_object_to_oss(f_name, dir_name, namespace, bucket_name, rps):
    object_storage = oci.object_storage.ObjectStorageClient(config={}, signer=rps)
    
    path_name = dir_name + "/" + f_name
    
    with open(path_name, 'rb') as f:
        obj = object_storage.put_object(namespace, bucket_name, f_name, f)
        
    print(f"Copy {f_name} done!")

#
# massive copy
#
def copy_list_objects_from_oss(list_files, dir_name, namespace, bucket_name, rps):
    # dir_name is the local dir where to copy to
    for f_name in list_files:
        copy_object_from_oss(f_name, dir_name, namespace, bucket_name, rps)

def copy_list_objects_to_oss(list_files, dir_name, namespace, bucket_name, rps):
    for f_name in list_files:
        copy_object_to_oss(f_name, dir_name, namespace, bucket_name, rps)