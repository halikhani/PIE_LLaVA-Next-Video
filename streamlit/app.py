import streamlit as st
from transformers import AutoTokenizer, LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor
import torch
from transformers import LlavaNextVideoProcessor
from datasets import load_from_disk
from datasets import load_dataset, concatenate_datasets


@st.cache_resource()
def load_models():

    model_ft = LlavaNextVideoForConditionalGeneration.from_pretrained(
    'hamidra/pie-llava-augmented',
    token="",
    torch_dtype=torch.float16,
    device_map="auto",
    )

    # loading base model
    model_base = LlavaNextVideoForConditionalGeneration.from_pretrained(
    "llava-hf/LLaVa-NeXT-Video-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto",
    )
    return model_base, model_ft
def run_inference(video_clip, model_base, model_ft):
    # NOTE: this part is hard-coded for the sake of demo, for actual inference on videos, necessary preprocessing should be aplied as it is done in the inference scripts
    val_dataset_p = load_from_disk('../hamidreza_files/hf_datasets_16frames/val_p/original')
    val_dataset_n = load_from_disk('../hamidreza_files/hf_datasets_16frames/val_n/original')
    # val_dataset = concatenate_datasets([val_dataset_p, val_dataset_n]).with_format("torch")
    positive_sample_idx = 7
    negative_sample_idx = 4
    p_sample = val_dataset_p[positive_sample_idx]
    n_sample = val_dataset_n[negative_sample_idx]
    if video_clip == 'crossing_clipped':
        pixel_vals = p_sample["pixel_values_videos"]
    else:
        pixel_vals = n_sample["pixel_values_videos"]
    video_clip = pixel_vals
    # end of hard-code
    processor = LlavaNextVideoProcessor.from_pretrained("/mnt/esperanto/et/intern/hamidreza/PIE/hamidreza_files/checkpoints/run_aug_dataset")
    processor.tokenizer.padding_side = "right"
    conversation = [
      {
          "role": "user",
          "content": [
              {"type": "text", "text": "Above are 16 frames of a driving scenario captured by the ego vehicle camera based on the video taken, in which the pedestrian of interest is located with a red bounding box "},
              {"type": "text", "text": "Using these frames, provided context and pedestrian bounding box, and your reasoning, answer the following question only with ‘Yes’ or ‘No’. You may use your knowledge if needed. DO NOT EXPLAIN your reasoning, be confident.\nQuestion: Does the indicated pedestrian intend to cross the intersection in future frames of this video?"},
              {"type": "video"},
              ]
      },
]

    # Set add_generation_prompt to add the "ASSISTANT: " at the end
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    device = "cuda"
    batch = processor(
        text=prompt,
        videos=None, # we have a processed video, passing it again to processor causes errors
        return_tensors="pt"
    ).to(device)
    video_clip = video_clip.to(device)
    MAX_LENGTH = 128
    out_base = model_base.generate(**batch, pixel_values_videos=video_clip, max_length=MAX_LENGTH, do_sample=True, temperature=1.0)
    generated_text_base = processor.batch_decode(out_base, skip_special_tokens=True)

    out_ft = model_ft.generate(**batch, pixel_values_videos=video_clip, max_length=MAX_LENGTH, do_sample=True, temperature=1.0)
    generated_text_ft = processor.batch_decode(out_ft, skip_special_tokens=True)
    return generated_text_base, generated_text_ft


def run(video_name, model_base, model_ft):
    gen_text_base, gen_text_ft = run_inference(video_name, model_base, model_ft)
    return gen_text_base, gen_text_ft


# Title of the web app
st.title("Pedestrian Intention Estimation (PIE) with LLaVA-Next-Video")

# pre-load models for faster response in webpage
model_base, model_ft = load_models()

if 'clicked' not in st.session_state:
    st.session_state.clicked = False
def click_button():
    st.session_state.clicked = True

if 'message1' not in st.session_state:
    st.session_state.message1 = ''
if 'message2' not in st.session_state:
    st.session_state.message2 = ''
if 'message3' not in st.session_state:
    st.session_state.message3 = ''

def reset():
    st.session_state.clicked = False
# Create a file uploader for video files
uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

# Check if a file is uploaded
if uploaded_video is not None:
    # Display the video
    st.video(uploaded_video)
    temp = 1
    if st.button("Run inference with base and fine-tuned LLaVA-Next-Video",on_click=click_button):
        if uploaded_video is not None:
            # Get the name of the uploaded file
            answer_base, answer_ft = run(uploaded_video.name[:-4], model_base, model_ft)
            st.write("Does the indicated pedestrian intend to cross the intersection in future frames of this video?\n")

            col1, col2 = st.columns(2)
            with col1:
                st.write("Base model answer:" + answer_base[0].split("ASSISTANT:")[1])

            with col2:
                st.write("Fine-tuned model answer:" + answer_ft[0].split("ASSISTANT:")[1])

            st.session_state.message1 = "Does the indicated pedestrian intend to cross the intersection in future frames of this video?" 
            st.session_state.message2 = "Base model answer:" + answer_base[0].split("ASSISTANT:")[1]
            st.session_state.message3 = "Fine-tuned model answer:" + answer_ft[0].split("ASSISTANT:")[1]
            temp = 0
        else:
                st.write("No video uploaded yet.")
    if st.session_state.clicked and temp !=0:
        st.write(st.session_state.message1)
        col1, col2 = st.columns(2)
        with col1:
            st.write(st.session_state.message2)
        with col2:
            st.write(st.session_state.message3)
    if st.button("Show complete video"):
        video_path_2 = "../crossing_full.mov" if uploaded_video.name[:-4] == 'crossing_clipped' else '../not_crossing_full.mov'
        st.video(video_path_2) 
    st.button('Reset', on_click = reset)
    

else:
    st.write("Please upload a video file.")



