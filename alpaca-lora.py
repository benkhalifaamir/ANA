from flask import Flask, request, jsonify
import torch
from peft import PeftModel
import transformers
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

app = Flask(__name__)

BASE_MODEL = "decapoda-research/llama-7b-hf"
LORA_WEIGHTS = "tloen/alpaca-lora-7b"
device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda":
    model = LlamaForCausalLM.from_pretrained(BASE_MODEL, load_in_8bit=False, torch_dtype=torch.float16, device_map="auto")
    model = PeftModel.from_pretrained(model, LORA_WEIGHTS, torch_dtype=torch.float16, force_download=True)
else:
    model = LlamaForCausalLM.from_pretrained(BASE_MODEL, device_map={"": device}, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model, LORA_WEIGHTS, device_map={"": device})

model.half()
model.eval()

tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")

def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Input:
{input}
### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Response:"""

@app.route('/evaluate', methods=['POST'])
def evaluate():
    data = request.json
    instruction = data.get('instruction')
    input_text = data.get('input')
    temperature = data.get('temperature', 0.1)
    top_p = data.get('top_p', 0.75)
    top_k = data.get('top_k', 40)
    num_beams = data.get('num_beams', 4)
    max_new_tokens = data.get('max_new_tokens', 128)

    prompt = generate_prompt(instruction, input_text)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(temperature=temperature, top_p=top_p, top_k=top_k, num_beams=num_beams)

    with torch.no_grad():
        generation_output = model.generate(input_ids=input_ids, generation_config=generation_config, return_dict_in_generate=True, output_scores=True, max_new_tokens=max_new_tokens)
    
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    return jsonify({'data': output.split("### Response:")[1].strip()})

if __name__ == '__main__':
    app.run(host='129.128.243.13', port=25500)