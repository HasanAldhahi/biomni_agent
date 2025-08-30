import sys
import os
from langchain_openai import ChatOpenAI
from biomni.agent import A1

api_key =  os.environ.get("CUSTOM_MODEL_API_KEY")
base_url = os.environ.get("CUSTOM_MODEL_BASE_URL")
custom_model_name = os.environ.get("CUSTOM_MODEL_NAME")
agent = A1(
    path='/mnt/exchange-saia/protein/haldhah/biomni_datalake',
    llm='custom',
    use_tool_retriever=True,
    timeout_seconds=1000,
    base_url=base_url,
    api_key=api_key,
    custom_model_name="qwen2.5-coder-32b-instruct"    
)



print('A1 agent created successfully!')



# llm = ChatOpenAI(
#             model='gemma-3-27b-it', 
#             temperature=0.7, 
#             max_tokens=8192, 
#             stop_sequences=["</execute>", "</solution>"],
#                 base_url='https://chat-ai.academiccloud.de/v1',

#             api_key="YOUR_API_KEY_HERE"
#         )

# print(llm.invoke("Hello, how are you?"))


print("--------------------------------")
log = agent.go("""Plan a CRISPR screen to identify genes that regulate T cell exhaustion,
        measured by the change in T cell receptor (TCR) signaling between acute
        (interleukin-2 [IL-2] only) and chronic (anti-CD3 and IL-2) stimulation conditions.
        Generate 32 genes that maximize the perturbation effect.""")