import torch.cuda

import tempfile
from PIL import Image
import scipy
import argparse

from llama.mumu_llama import MuMu_LLaMA
import llama
import numpy as np
import os
import torch
import torchaudio
import torchvision.transforms as transforms
import av
import subprocess
import librosa

from datasets import load_from_disk
from tqdm import tqdm

import pandas as pd

import json

import logging

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="./ckpts/checkpoint.pth", type=str,
        help="Name of or path to MuMu_LLaMA pretrained checkpoint",
    )
    parser.add_argument(
        "--llama_type", default="7B", type=str,
        help="Type of llama original weight",
    )
    parser.add_argument(
        "--llama_dir", default="./ckpts/LLaMA", type=str,
        help="Path to LLaMA pretrained checkpoint",
    )
    parser.add_argument(
        "--mert_path", default="./ckpts/m-a-p/MERT-v1-330M", type=str,
        help="Path to MERT pretrained checkpoint",
    )
    parser.add_argument(
        "--vit_path", default="./ckpts/google/vit-base-patch16-224-in21k", type=str,
        help="Path to ViT pretrained checkpoint",
    )
    parser.add_argument(
        "--vivit_path", default="./ckpts/google/vivit-b-16x2-kinetics400", type=str,
        help="Path to ViViT pretrained checkpoint",
    )
    parser.add_argument(
        "--knn_dir", default="./ckpts", type=str,
        help="Path to directory with KNN Index",
    )
    parser.add_argument(
        '--music_decoder', default="musicgen", type=str,
        help='Decoder to use musicgen/audioldm2')

    parser.add_argument(
        '--music_decoder_path', default="facebook/musicgen-small", type=str,
        help='Path to decoder to use musicgen/audioldm2')

    # Input Arguments
    parser.add_argument(
        "--prompt", default="Generate a music", type=str,
        help="Input Prompt to the MuMu_LLaMA model",
    )
    parser.add_argument(
        "--audio_file", default=None, type=str,
        help="Input Audio File to the MuMu_LLaMA model",
    )
    parser.add_argument(
        "--image_file", default=None, type=str,
        help="Input Image File to the MuMu_LLaMA model",
    )
    parser.add_argument(
        "--video_file", default=None, type=str,
        help="Input Video File to the MuMu_LLaMA model",
    )
    parser.add_argument(
        "--eval_set", default=None, type=str,
        help="musiccaps|muimage|muvideo|imemnet",
    )
    parser.add_argument(
        "--filename", default=None, type=str,
        help="Output filename"
    )
    parser.add_argument(
        "--output_folder", default=None, type=str,
        help="Output folder"
    )

    return parser.parse_args()

def parse_text(text, image_path, video_path, audio_path):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    outputs = text
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines) + "<br>"
    if image_path is not None:
        text += f'<img src="./file={image_path}" style="display: inline-block;"><br>'
        outputs = f'<Image>{image_path}</Image> ' + outputs
    if video_path is not None:
        text += f' <video controls playsinline height="320" width="240" style="display: inline-block;"  src="./file={video_path}"></video6><br>'
        outputs = f'<Video>{video_path}</Video> ' + outputs
    if audio_path is not None:
        text += f'<audio controls playsinline><source src="./file={audio_path}" type="audio/wav"></audio><br>'
        outputs = f'<Audio>{audio_path}</Audio> ' + outputs
    # text = text[::-1].replace(">rb<", "", 1)[::-1]
    text = text[:-len("<br>")].rstrip() if text.endswith("<br>") else text
    return text, outputs


def save_audio_to_local(audio, sec, output_folder=None, output_filename=None):
    global generated_audio_files
    generated_dir=output_folder
    if not os.path.exists(generated_dir):
        os.mkdir(generated_dir)
    # filename = os.path.join(generated_dir, next(tempfile._get_candidate_names()) + '.wav')
    filename = os.path.join(generated_dir, output_filename)
    if args.music_decoder == "audioldm2":
        scipy.io.wavfile.write(filename, rate=16000, data=audio[0])
    else:
        scipy.io.wavfile.write(filename, rate=model.generation_model.config.audio_encoder.sampling_rate, data=audio)
    generated_audio_files.append(filename)
    return filename


def parse_reponse(model_outputs, audio_length_in_s, output_folder=None, output_filename=None):
    response = ''
    text_outputs = []
    filename = None
    for output_i, p in enumerate(model_outputs):
        if isinstance(p, str):
            response += p
            response += '<br>'
            text_outputs.append(p)
        elif 'aud' in p.keys():
            _temp_output = ''
            for idx, m in enumerate(p['aud']):
                if isinstance(m, str):
                    response += m.replace(''.join([f'[AUD{i}]' for i in range(8)]), '')
                    response += '<br>'
                    _temp_output += m.replace(''.join([f'[AUD{i}]' for i in range(8)]), '')
                else:
                    filename = save_audio_to_local(m, audio_length_in_s, output_folder, output_filename)
                    _temp_output = f'<Audio>{filename}</Audio> ' + _temp_output
                    response += f'<audio controls playsinline><source src="./file={filename}" type="audio/wav"></audio>'
            text_outputs.append(_temp_output)
        else:
            pass
    response = response[:-len("<br>")].rstrip() if response.endswith("<br>") else response
    return response, text_outputs, filename


def read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    if converted_len > seg_len:
        converted_len = 0
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


def get_video_length(filename):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    return int(round(float(result.stdout)))


def get_audio_length(filename):
    return int(round(librosa.get_duration(path=filename)))


def predict(
        prompt_input,
        image_path,
        audio_path,
        video_path,
        top_p,
        temperature,
        audio_length_in_s, 
        output_folder=None,
        output_filename=None):
    prompts = [llama.format_prompt(prompt_input)]
    prompts = [model.tokenizer(x).input_ids for x in prompts]
    image, audio, video = None, None, None
    if image_path is not None:
        image = transform(Image.open(image_path))
    if audio_path is not None:
        sample_rate = 24000
        waveform, sr = torchaudio.load(audio_path)
        if sample_rate != sr:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=sample_rate)
        audio = torch.mean(waveform, 0)
    if video_path is not None:
        container = av.open(video_path)
        indices = sample_frame_indices(clip_len=32, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
        video = read_video_pyav(container=container, indices=indices)

    if video_path is not None:
        audio_length_in_s = get_video_length(video_path)
        print(f"Video Length: {audio_length_in_s}")
    if audio_path is not None:
        audio_length_in_s = get_audio_length(audio_path)
        generated_audio_files.append(audio_path)
        print(f"Audio Length: {audio_length_in_s}")

    response = model.generate(prompts, audio, image, video, 512, temperature, top_p,
                              audio_length_in_s=audio_length_in_s)
    response_chat, response_outputs, filename = parse_reponse(response, audio_length_in_s, output_folder, output_filename)
    print(f"Q. {prompt_input}")
    print(f"A. {response[0]}")
    if filename is not None:
        print(f"Generated Audio: {filename}")

if __name__ == "__main__":
    print("MuMu-LLaMA inference.")
    global args
    args = parse_args()
    print(f"Eval set is {args.eval_set}")

    generated_audio_files = []

    print(f"Setting up model architecture")
    llama_type = args.llama_type
    llama_ckpt_dir = os.path.join(args.llama_dir, llama_type)
    llama_tokenzier_path = args.llama_dir
    global model
    model = MuMu_LLaMA(llama_ckpt_dir, llama_tokenzier_path, args, knn=False, stage=3, load_llama=False)

    print("Loading Model Checkpoint")
    checkpoint = torch.load(args.model, map_location='cpu')

    new_ckpt = {}
    for key, value in checkpoint['model'].items():
        key = key.replace("module.", "")
        new_ckpt[key] = value

    load_result = model.load_state_dict(checkpoint['model'], strict=False)
    assert len(load_result.unexpected_keys) == 0, f"Unexpected keys: {load_result.unexpected_keys}"
    model.eval()
    model.to("cuda")

    print("Setting up transforrms")

    global transform
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x)])

    if (args.eval_set is None):
        print(f"Generating single sample for prompt: {args.prompt}")
        print(args.image_file)
        print(args.audio_file)
        print(args.video_file)
        output_folder = '/home2/faculty/ktiurina/composer/data/survey'
        output_filename = args.filename
        if (output_filename is None):
            output_filename = next(tempfile._get_candidate_names()) + '.wav'
        predict(args.prompt, args.image_file, args.audio_file, args.video_file, 0.8, 0.6, 30, output_folder, output_filename)
    elif args.eval_set == 'musiccaps':
        print("Start generating for MusicCaps dataset")
        musiccaps = load_from_disk('/home2/faculty/ktiurina/composer/data/MusicCaps/metadata')
        output_folder = '/home2/faculty/ktiurina/composer/data/generated/MusicCaps/MuMu-LLaMA'
        os.makedirs(output_folder, exist_ok=True)
        for sample in musiccaps:
            file_id = sample['ytid']
            output_filename = f'{file_id}.wav'
            outfile = os.path.join(output_folder, f'{file_id}.wav')
            if os.path.isfile(outfile):
                print(f"Sample {file_id} already generated. Skip")
            else:
                caption = f"Generate music for the following caption: {sample['caption']}"
                predict(caption, None, None, None, 0.8, 0.6, 30, output_folder, output_filename)
            print(f"Generated {output_filename}")
    elif args.eval_set == 'muimage':
        print("Start generating for MUImage dataset")
        muimage_instructions = '/home2/faculty/ktiurina/composer/data/MUImage/MUImageInstructions.json'
        images_folder = '/home2/faculty/ktiurina/composer/data/MUImage/muimage_images/hpctmp/e0589920/MUGen/data/MUImage/audioset_images'
        with open(muimage_instructions, 'r') as file:
            MuImage = json.load(file)

        output_folder = '/home2/faculty/ktiurina/composer/data/generated/MUImage/MuMu-LLaMA'
        os.makedirs(output_folder, exist_ok=True)
        for sample in MuImage:
            file_id = sample['input_file'].split('.')[0]
            output_filename = f'{file_id}.wav'
            outfile = os.path.join(output_folder, f'{file_id}.wav')
            if os.path.isfile(outfile):
                print(f"Sample {file_id} already generated. Skip")
            else:
                image_path = os.path.join(images_folder, sample['input_file'])
                human_msg = [msg for msg in sample['conversation'] if msg.get('from') == 'human']
                if len(human_msg) > 0:
                    prompt = human_msg[0]['value']
                else:
                    prompt = 'Generate music for the image'
                predict(prompt, image_path, None, None, 0.8, 0.6, 30, output_folder, output_filename)
                print(f"Generated {output_filename}")
    elif args.eval_set == 'custom':
        print("Start generating for custom dataset")
        output_folder = args.output_folder
        os.makedirs(output_folder, exist_ok=True)
        styles_df = pd.read_csv("/home2/faculty/ktiurina/composer/data/custom/MusicTheory.csv")

        for index, row in styles_df.iterrows():
            file_id = row['musicTheoryTerm'].replace("/", "_")
            filename = f'{file_id}.wav'
            outfile = os.path.join(output_folder, filename)
            if os.path.isfile(outfile):
                print(f"Sample {file_id} already generated.Skip")
            else:
                print(row['prompt'])
                predict(row['prompt'], None, None, None, 0.8, 0.6, 30, output_folder, filename)
                print(f"Generated {filename}")

    elif args.eval_set == 'imemnet':
        print("Start generating for IMEMNet dataset")
        images_folder = '/home2/faculty/ktiurina/composer/data/IMEMNet/images_processed'

        output_folder = '/home2/faculty/ktiurina/composer/data/generated/IMEMNet/MuMu-LLaMA'
        os.makedirs(output_folder, exist_ok=True)
        jpg_files = [f for f in os.listdir(images_folder) if f.lower().endswith('.jpg')]

        for input_image in jpg_files:
            file_id = input_image.split('.')[0]
            output_filename = f'{file_id}.wav'
            outfile = os.path.join(output_folder, f'{file_id}.wav')
            if os.path.isfile(outfile):
                print(f"Sample {file_id} already generated. Skip")
            else:
                image_path = os.path.join(images_folder, input_image)
                prompt = 'Generate music for the image'
                predict(prompt, image_path, None, None, 0.8, 0.6, 30, output_folder, output_filename)
                print(f"Generated {output_filename}")
    elif args.eval_set == 'muvideo':
        print("Start generating for MUVideo dataset")
        muvideo_instructions = '/home2/faculty/ktiurina/composer/data/MUVideo/MUVideoInstructions.json'
        videos_folder = '/home2/faculty/ktiurina/composer/data/MUVideo/muvideo_videos/hpctmp/e0589920/MUGen/data/MUVideo/audioset_video'
        with open(muvideo_instructions, 'r') as file:
            MuVideo = json.load(file)

        output_folder = '/home2/faculty/ktiurina/composer/data/generated/MUVideo/MuMU-LLaMA'
        os.makedirs(output_folder, exist_ok=True)
        for sample in MuVideo:
            file_id = sample['input_file'].split('.')[0]
            output_filename = f'{file_id}.wav'
            outfile = os.path.join(output_folder, f'{file_id}.wav')
            if os.path.isfile(outfile):
                print(f"Sample {file_id} already generated. Skip")
            else:
                print(f"Start generating for {file_id}")
                video_path = os.path.join(videos_folder, sample['input_file'])
                human_msg = [msg for msg in sample['conversation'] if msg.get('from') == 'human']
                if len(human_msg) > 0:
                    prompt = human_msg[0]['value']
                else:
                    prompt = 'Generate music for the video'
                try:
                    predict(prompt, None, None, video_path, 0.8, 0.6, 30, output_folder, output_filename)
                    print(f"Generated {output_filename}")
                except Exception as e:
                    print("ERROR: ", e)
                    logging.exception("Error while generating music for MUVideo")
