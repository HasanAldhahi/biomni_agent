import sys
import os
from langchain_openai import ChatOpenAI
from biomni.agent import A1

# Get model name from environment variable
model_name = os.environ.get("CURRENT_TEST_MODEL", "qwen2.5-coder-32b-instruct")

print(f"=== TESTING A1 AGENT WITH MODEL: {model_name} ===")
print(f"Timestamp: {os.popen('date').read().strip()}")
print("=" * 60)

api_key = os.environ.get("CUSTOM_MODEL_API_KEY")
base_url = os.environ.get("CUSTOM_MODEL_BASE_URL")

try:
    agent = A1(
        path='/mnt/exchange-saia/protein/haldhah/biomni_datalake',
        llm='custom',
        use_tool_retriever=True,
        timeout_seconds=1000,
        base_url=base_url,
        api_key=api_key,
        custom_model_name=model_name    
    )

    print(f'A1 agent created successfully with model: {model_name}!')
    print("--------------------------------")
    
    log = agent.go("""Plan a CRISPR screen to identify genes that regulate T cell exhaustion,
            measured by the change in T cell receptor (TCR) signaling between acute
            (interleukin-2 [IL-2] only) and chronic (anti-CD3 and IL-2) stimulation conditions.
            Generate 32 genes that maximize the perturbation effect.""")
    
    print("--------------------------------")
    print("TEST COMPLETED SUCCESSFULLY")
    print(f"Model: {model_name}")
    print(f"Agent: A1")
    print(f"End time: {os.popen('date').read().strip()}")
    
except Exception as e:
    print(f"ERROR: Test failed for model {model_name}")
    print(f"Error details: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1) 