from flask import Flask, request, jsonify

app = Flask(__name__)

from torch import cuda, bfloat16
import transformers

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

model = transformers.AutoModelForCausalLM.from_pretrained(
    'mosaicml/mpt-30b-chat',
    trust_remote_code=True,
    load_in_8bit=True,  # this requires the `bitsandbytes` library
    max_seq_len=8192,
    init_device=device
)
model.eval()
#model.to(device)
print(f"Model loaded on {device}")
tokenizer = transformers.AutoTokenizer.from_pretrained("mosaicml/mpt-30b")
from transformers import StoppingCriteria, StoppingCriteriaList

# we create a list of stopping criteria
stop_token_ids = [
    tokenizer.convert_tokens_to_ids(x) for x in [
        ['Human', ':'], ['AI', ':']
    ]
]

import torch

stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]
from transformers import StoppingCriteria, StoppingCriteriaList

# define custom stopping criteria object
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids in stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                return True
        return False

stopping_criteria = StoppingCriteriaList([StopOnTokens()])

generate_text = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    # we pass model parameters here too
    stopping_criteria=stopping_criteria,  # without this model rambles during chat
    temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    top_p=0.15,  # select from top tokens whose probability add up to 15%
    top_k=0,  # select from top 0 tokens (because zero, relies on top_p)
    max_new_tokens=128,  # mex number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
)

res = generate_text("Explain to me the difference between nuclear fission and fusion.")
print(res[0]["generated_text"])

from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline

# template for an instruction with no input
prompt = PromptTemplate(
    input_variables=["instruction"],
    template="{instruction}"
)

llm = HuggingFacePipeline(pipeline=generate_text)

llm_chain = LLMChain(llm=llm, prompt=prompt)
print(llm_chain.predict(
    instruction="Explain to me the difference between nuclear fission and fusion."
).lstrip())

from langchain.chains.conversation.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(
    memory_key="history",  # important to align with agent prompt (below)
    k=5,
    return_only_outputs=True
)
     
from langchain.chains import ConversationChain

chat = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)
chat.prompt.template = \
"""The following is a friendly conversation between a human and an AI. The AI is conversational but concise in its responses without rambling. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
{history}
Human: {input}
AI:"""
res = chat.predict(input='hi how are you?')
chat.memory
chat.memory.chat_memory.messages[-1]
# check for double newlines (also happens often)
chat.memory.chat_memory.messages[-1].content = chat.memory.chat_memory.messages[-1].content.split('\n\n')[0]
# strip any whitespace
chat.memory.chat_memory.messages[-1].content = chat.memory.chat_memory.messages[-1].content.strip()
# check for stop text at end of output
for stop_text in ['Human:', 'AI:', '[]']:
    chat.memory.chat_memory.messages[-1].content = chat.memory.chat_memory.messages[-1].content.removesuffix(stop_text)
# strip again
chat.memory.chat_memory.messages[-1].content = chat.memory.chat_memory.messages[-1].content.strip()

chat.memory.chat_memory.messages[-1]

def chat_trim(chat_chain, query):
    # create response
    chat_chain.predict(input=query)
    # check for double newlines (also happens often)
    chat.memory.chat_memory.messages[-1].content = chat.memory.chat_memory.messages[-1].content.split('\n\n')[0]
    # strip any whitespace
    chat.memory.chat_memory.messages[-1].content = chat.memory.chat_memory.messages[-1].content.strip()
    # check for stop text at end of output
    for stop_text in ['Human:', 'AI:', '[]']:
        chat.memory.chat_memory.messages[-1].content = chat.memory.chat_memory.messages[-1].content.removesuffix(stop_text)
    # strip again
    chat.memory.chat_memory.messages[-1].content = chat.memory.chat_memory.messages[-1].content.strip()
    # return final response
    return chat_chain.memory.chat_memory.messages[-1].content
@app.route('/generate', methods=['POST'])
def generate():
    query = request.json['query']
    response_text = chat_trim(chat, query)
    return jsonify({'response': response_text})

if __name__ == '__main__':
    app.run(host='129.128.243.13', port=25500)  # listen on all public IPs