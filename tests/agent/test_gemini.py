import sys
import os
from langchain_openai import ChatOpenAI
from biomni.agent import A1

api_key =  os.environ.get("CUSTOM_MODEL_API_KEY")
base_url = os.environ.get("CUSTOM_MODEL_BASE_URL")
custom_model_name = os.environ.get("CUSTOM_MODEL_NAME")


gemini_api_key =  os.environ.get("GEMINI_API_KEY")

agent = A1(
    path='/mnt/exchange-saia/protein/haldhah/biomni_datalake',
    llm='gemini-2.5-pro',
    use_tool_retriever=True,
    timeout_seconds=600,
    api_key=gemini_api_key,
    
)


print('A1 agent created successfully!')



print("--------------------------------")
log = agent.go("""Plan a CRISPR screen to identify genes that regulate T cell exhaustion,
        measured by the change in T cell receptor (TCR) signaling between acute
        (interleukin-2 [IL-2] only) and chronic (anti-CD3 and IL-2) stimulation conditions.
        Generate 32 genes that maximize the perturbation effect.""")