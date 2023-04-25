from bark import SAMPLE_RATE, generate_audio, preload_models, save_as_prompt
from pydub import AudioSegment
import gradio as gr
import numpy as np
import torchaudio
import torch
import nltk
import os

nltk.download('punkt')
torchaudio.set_audio_backend("soundfile")

# download and load all models
preload_models()

# define your function that takes a text input and returns an audio output
def text_to_audio(text, text_temp, waveform_temp, history_prompt):

    if(history_prompt == "Unconditional"):
        history_prompt = None
        
    #segment the sentences
    text_prompts_list = nltk.sent_tokenize(text)
    
    # generate audio from text
    audio_arrays = np.array([])
    i = 1
    
    for prompt in text_prompts_list:
        print(f"{i} of {len(text_prompts_list)}")
        full_generation, audio_array = generate_audio(prompt,
                                     history_prompt,
                                     text_temp,
                                     waveform_temp,
                                     output_full = True)
        audio_arrays = np.concatenate((audio_arrays, audio_array))
                        
        save_as_prompt(os.path.join(cwd, f"bark/assets/userprompts/temp.npz"), full_generation)
        history_prompt = os.path.join(cwd, f"bark/assets/userprompts/temp.npz")
        i = i+1
    # return audio array as output
    return SAMPLE_RATE, audio_arrays

# get the list of files in the prompts folder
cwd = os.getcwd()
files = os.listdir(os.path.join(cwd, "bark/assets/prompts"))

# remove the file extension names
files = [os.path.splitext(f)[0] for f in files]

files.insert(0, "Unconditional")

# create a list of input components
inputs = [
    gr.Textbox(label="text"),
    gr.Slider(minimum=0.0, maximum=1.0, value=0.7, label="text_temp"),
    gr.Slider(minimum=0.0, maximum=1.0, value=0.7, label="waveform_temp"),
    gr.Dropdown(label="history_prompt", choices=files)
]

# create an interface object
interface = gr.Interface(text_to_audio, inputs, "audio")

# launch the interface
interface.launch()

#TODO
#Add save as prompt
