from bark import SAMPLE_RATE, generate_audio, preload_models, save_as_prompt
from pydub import AudioSegment
import gradio as gr
import numpy as np
import torch
import random
import torchaudio
import torch
import nltk
import os
nltk.download('punkt')
torchaudio.set_audio_backend("soundfile")

seed = 0


def set_seed(seed):
    seed = int(seed)
    if(seed == 0):
        seed = random.randint(0, 2**32-1)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)
    return seed

# define your function that takes a text input and returns an audio output
def text_to_audio(text, history_prompt, text_temp, coarse_temp, fine_temp, allow_early_stop, seed):

    seed = set_seed(seed)

    if(history_prompt == "Unconditional"):
        history_prompt = None
        
    #segment the sentences
    text_prompts_list = nltk.sent_tokenize(text)
    
    # generate audio from text
    audio_arrays = np.array([])
    i = 1
    audio_list = []
    full_generation = {}
    for prompt in text_prompts_list:
        print(f"{i} of {len(text_prompts_list)}")
        full_generation, audio_array = generate_audio(
                                     text = prompt,
                                     history_prompt = history_prompt,
                                     text_temp = text_temp,
                                     coarse_temp = coarse_temp,
                                     fine_temp = fine_temp,
                                     output_full = True,
                                     allow_early_stop = allow_early_stop)
        audio_arrays = np.concatenate((audio_arrays, audio_array))
                        
        #save_as_prompt(os.path.join(cwd, f"bark/assets/userprompts/{i}.npz"), full_generation)
        #history_prompt = os.path.join(cwd, f"bark/assets/userprompts/{i}.npz")        
        i=i+1
        
    # return audio array as output
    return (SAMPLE_RATE, audio_arrays), seed

# get the list of files in the prompts folder
cwd = os.getcwd()
files = os.listdir(os.path.join(cwd, "bark/assets/prompts"))

# remove the file extension names
files = [os.path.splitext(f)[0] for f in files]

files.insert(0, "Unconditional")

# create a list of input components
inputs = [
    gr.Textbox(label="text"),
    gr.Dropdown(label="history_prompt", choices=files),
    gr.Slider(minimum=0.01, maximum=1.0, value=0.7, label="text_temp", info="Lower is more consistent with input text (Less likely to um and stammer"),
    gr.Slider(minimum=0.01, maximum=1.0, value=0.6, label="coarse_temp", info="Lower is more consistent with the history_prompt, too low seems to copy the history_prompt"),
    gr.Slider(minimum=0.01, maximum=1.0, value=0.2, label="fine_temp", info="Lower is more consistent, seems to control intonation and pitch"),
    gr.Checkbox(label="Allow Early Stop", value = True, info="Unchecked, model will fill entire context length"),
    gr.Number(value=0, label="Seed", info="0 for random")
]
# create an interface object
interface = gr.Interface(text_to_audio, inputs, outputs=["audio", gr.Textbox(label="Seed Used")])



# download and load all models
preload_models()

# launch the interface
interface.launch()
#TODO
#Add save as prompt
