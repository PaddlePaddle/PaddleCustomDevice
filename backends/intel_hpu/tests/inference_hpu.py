import os
#os.environ['ENABLE_EXPERIMENTAL_FLAGS'] = '1'
#os.environ['VISUALIZATION_MODE'] = '0'
#os.environ['GRAPH_VISUALIZATION'] = '1'
os.environ['HABANA_LOGS'] = 'logs'
os.environ['LOG_LEVEL_ALL'] = '0'
#os.environ['GLOG_v'] = '10'

import paddle
from paddle.static import InputSpec
from paddlenlp.transformers import AutoTokenizer, AutoModelForCausalLM

#paddle.set_device("cpu")
paddle.set_device("intel_hpu")

model = "meta-llama/Llama-2-7b-chat"
tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModelForCausalLM.from_pretrained(model, dtype="float32")

input_features = tokenizer("please introduce llm", return_tensors="pd")

with paddle.amp.auto_cast(dtype="bfloat16"):
    outputs = model.generate(**input_features, max_length=128)

#outputs = model.generate(**input_features, max_length=128)

print(outputs[0])
#print(tokenizer.batch_decode(outputs[0]))

