import os
import av
import fsspec
import shutil
import numpy as np
import json
from tqdm import tqdm

from transformers import Trainer, TrainingArguments, Seq2SeqTrainingArguments, DataCollatorForLanguageModeling
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaNextVideoForConditionalGeneration
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import BitsAndBytesConfig, LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from huggingface_hub import snapshot_download, hf_hub_download, HfFileSystem
from datasets import load_dataset, concatenate_datasets


MAX_LENGTH = 128
BATCH_SIZE = 4
NUM_FRAMES = 8 # more frames -> more VRAM needed
DATASET_PATH = ""
OUTPUT_DIR = 'YOUR_CHECKPOINT_DIR'
MODEL_ID = "llava-hf/LLaVa-NeXT-Video-7b-hf"
REPO_ID = "YOUR_REPO_ID"

USE_LORA = False
USE_QLORA = True

def collate_fn(example, caption):
    video_clip = example 

    answer_caption = caption['answer']

    conversation = [
            {
          "role": "user",
          "content": [
              {"type": "text", "text": "Above are 16 frames of a driving scenario captured by the ego vehicle camera based on the video taken, in which the pedestrian of interest is located with a red bounding box "},
              {"type": "text", "text": "Using these frames, provided context and pedestrian bounding box, and your reasoning, answer the following question only with ‘Yes’ or ‘No’. You may use your knowledge if needed. DO NOT EXPLAIN your reasoning, be confident.\nQuestion: Does the indicated pedestrian intend to cross the intersection in future frames of this video?"},
              {"type": "video"},
              ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer_caption},
                     ],
            },
        ]
    processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=False)
    processor.tokenizer.padding_side = "right" 
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=False)

    batch = processor(
        text=prompt,
        videos=video_clip,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    return batch

def gen_hf_format_dataset(ped_attr_json_file_dir, clip_npy_base_dir):
    # converting the costum dataset into hf dataset format
    file_path = ped_attr_json_file_dir
    with open(file_path, 'r') as json_file:
        ped_dict = json.load(json_file)

    base_dir = clip_npy_base_dir
    npy_files_list = os.listdir(base_dir)

    datasets_combined = []
    for nfl in tqdm(npy_files_list):
        file_dir = base_dir + nfl
        pie_clip = np.load(file_dir)

        ped_id = nfl[:-4]
        action_bin = int(ped_dict[ped_id][1])
        # convert 0/1 to 'yes' or 'no'
        answer_caption = 'Yes' if action_bin == 1 else 'No'

        collate = collate_fn(pie_clip, answer_caption)
        hf_dataset = Dataset.from_dict(collate)
        datasets_combined.append(hf_dataset)

    hf_dataset = concatenate_datasets(datasets_combined)
    return hf_dataset

def display_video(example_clip):
    # example: seeing one of the videos in training set
    from matplotlib import pyplot as plt
    from matplotlib import animation
    from IPython.display import HTML

    # convert to image from proceessed tensors
    clip = example_clip["pixel_values_videos"][0] * 255
    clip = clip.permute(0, 2, 3, 1).clamp(0, 255)

    # np array with shape (frames, height, width, channels)
    video = np.array(clip).astype(np.uint8)

    fig = plt.figure()
    im = plt.imshow(video[0,:,:,:])

    plt.close() # this is required to not display the generated image

    def init():
        im.set_data(video[0,:,:,:])

    def animate(i):
        im.set_data(video[i,:,:,:])
        return im

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=video.shape[0],
                                interval=100)
    HTML(anim.to_html5_video())



class LlavaNextVideoDataCollatorWithPadding:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        padded_inputs = self.processor.tokenizer.pad(
            {
                "input_ids": [feat['input_ids'][0] for feat in features], # each element is one batch only so we slice [0]
                "attention_mask": [feat['attention_mask'][0] for feat in features],
            },
            padding=True,
            return_tensors="pt",
        )

        labels = padded_inputs["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        padded_inputs["labels"] = labels
        padded_inputs["pixel_values_videos"] = torch.cat([feat['pixel_values_videos'] for feat in features], dim=0)

        return padded_inputs

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['multi_modal_projector', 'vision_model']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def unsqueeze_pixel_values(example):
    for k in example.keys():
        if example[k].shape[0] == 1:
            continue
        else:
            example[k] = example[k].unsqueeze(0)
    return example

def run():
    processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=False)
    processor.tokenizer.padding_side = "right" 

    ## Load model
    # Three options for training, from the lowest precision training to the highest precision training:
    # QLoRA: model uses 4-bit quantization, which helps in reducing memory usage while maintaining performance.
    # Standard LoRA:  model is loaded with standard LoRA adaptations.
    # Full Fine-Tuning: no memory optimization are done. In that case Flash Attention is used to speed up training, if hardware supports it.

    if USE_QLORA or USE_LORA:
        if USE_QLORA:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
        model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        # for full fine-tuning, we can speed up the model using Flash Attention
        # only available on certain devices, see https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features
        model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            _attn_implementation="flash_attention_2",
            device_map="auto",
    )
        
    lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules=find_all_linear_names(model),
        init_lora_weights="gaussian",
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    
    args = TrainingArguments(

        # args related to training
        output_dir = OUTPUT_DIR,
        eval_strategy = 'steps',
        eval_steps=200,
        num_train_epochs=2,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size = BATCH_SIZE,
        gradient_accumulation_steps = 8,
        learning_rate = 2e-05,
        max_steps = 2000, # adjust this depending on your dataset size
        lr_scheduler_type = 'cosine',
        warmup_ratio = 0.1,
        # args related to eval/save
        logging_steps = 200,
        save_strategy = 'steps',
        save_steps=200,
        save_total_limit = 10,
        fp16 = True, # we have the model train and eval with fp16 precision
        fp16_full_eval = True,
        optim = 'adamw_bnb_8bit', # adam in lower-bits to save memory, consider changing to 'adamw_torch' if model is not converging
        report_to = "wandb", # install wand to use this
        hub_model_id = REPO_ID,
        push_to_hub = True, # wel'll push the model to hub after each epoch

        # model that was wrapped for QLORA training with peft will not have arguments listed in its signature
        # so we need to pass lable names explicitly to calculate val loss
        label_names=["labels"],
        dataloader_num_workers=4, # let's get more workers since iterating on video datasets might be slower in general
    )

    # first load train and test dataset from disk
    # ...
    from datasets import load_from_disk

    train_val_dataset_dir = "YOUR_TRAIN_DIR"
    train_val_dataset = load_from_disk(train_val_dataset_dir)
   
    train_val_dataset = train_val_dataset.train_test_split(test_size=0.2)
    
    train_dataset = train_val_dataset['train']
    val_dataset = train_val_dataset['test']

    print('len train: ' + str(len(train_dataset)))
    print('len val: ' + str(len(val_dataset)))
    # print('len test: ' + str(len(test_dataset)))
    trainer = Trainer(
        model = model,
        tokenizer = processor,
        data_collator = LlavaNextVideoDataCollatorWithPadding(processor=processor),
        train_dataset = train_dataset,
        eval_dataset = val_dataset,
        args=args,
        )
    
    trainer.train()
    # trainer.model.push_to_hub(REPO_ID)


if __name__ == '__main__':
    from huggingface_hub import login

    token = "YOUR_HF_TOKEN"
    login(token=token)
    run()