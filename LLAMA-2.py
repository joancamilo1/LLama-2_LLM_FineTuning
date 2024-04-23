# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 15:58:22 2024

@author: Joan Camilo tamayo

pip install --upgrade bitsandbytes

"""

# verificar buena instalacion de pytorch y compatibilidad con cuda
import torch
print(torch.cuda.is_available())
print(torch.__version__)

import os
import datasets
from datasets import load_dataset
import transformers

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)

import peft
import trl
from peft import LoraConfig, PeftModel
from trl import SFTTrainer


# Fine-tuned model name
new_model = r"C:\Users\Joan Camilo\Desktop\P\Python\LLMs\llama-2-7b-miniguanaco"

# The model that you want to train from the Hugging Face hub
model_name = "NousResearch/Llama-2-7b-chat-hf"

# Load the entire model on the GPU 0
device_map = {"": 0}


# Reload model in FP16 and merge it with LoRA weights
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
)
model = PeftModel.from_pretrained(base_model, new_model)
model = model.merge_and_unload()

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"



#################### pruebas ##########################
# Ignore warnings
logging.set_verbosity(logging.CRITICAL)

# Run text generation pipeline with our next model
prompt = "Como puedo encontrar trabajo de ingeniero?"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text'])



import time
inicio = time.time()

#################### prueba 2 ##########################
# Run text generation pipeline with our next model

txt = """Juan Pérez, un hombre de 45 años, acudió a la consulta médica quejándose 
        de dolor crónico en las articulaciones y rigidez matutina. Durante la entrevista inicial, 
        reveló antecedentes familiares de artritis. Además, describió síntomas como inflamación 
        en las articulaciones y dificultad para moverse. Tras realizar un examen físico completo,
         se diagnosticó a Juan con artritis reumatoide. retiro lo dicho, no tiene artritis
         """
txt = """ La paciente presentaba un conjunto de síntomas que sugerían una variedad de posibles 
         condiciones médicas. Se observaron signos de malestar general, pero la naturaleza exacta
         de la dolencia no era clara. La exploración física reveló algunas anomalías, pero su 
         significado era ambiguo. Se requirió una evaluación más exhaustiva para discernir si
         los síntomas eran indicativos de una enfermedad específica o si eran simplemente manifestaciones
         transitorias de estrés o fatiga. 
         con todo ello aparentemente no puedo negar la afirmacion que niega el decir que si tiene artritis
         
         """

txt = """ La paciente presentaba un conjunto de síntomas que sugerían una variedad de posibles 
         condiciones médicas. Se observaron signos de malestar general, pero la naturaleza exacta
         de la dolencia no era clara. La exploración física reveló algunas anomalías, pero su 
         significado era ambiguo. Se requirió una evaluación más exhaustiva para discernir si
         los síntomas eran indicativos de una enfermedad específica o si eran simplemente manifestaciones
         transitorias de estrés o fatiga. con ello, se descarta el cuadro clinico debido a la falta de pruebas
         
         """

prompt = f"responde solo si o no, si el paciente tiene o no tiene artritis: {txt}  " #ptte con dx ar
texto_sin_saltos = prompt.replace('\n', '')# Eliminar '\n'
prompt = ' '.join(texto_sin_saltos.split())# Eliminar espacios extras

pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
result = pipe(f"<s>[INST] {prompt} [/INST]")

# print(result[0]['generated_text'])
respuesta = result[0]['generated_text']
texto_deseado = respuesta.split("[/INST]")[1].strip()

print("Prompt: " + prompt)
print("Respuesta: " + texto_deseado)

fin = time.time()
tiempo_transcurrido = fin - inicio

print("Tiempo de ejecución:", tiempo_transcurrido, "segundos")





