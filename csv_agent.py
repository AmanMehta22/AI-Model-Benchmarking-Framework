import pandas as pd
import json
import urllib.request

# 1. Load your dataset
# Assumes columns are named 'question' and 'answer'
try:
    df = pd.read_csv("KCC_Call_Dataset.csv")
    # Convert CSV to a compact string format the model can read easily
    csv_context = ""
    for idx, row in df.iterrows():
        csv_context += f"Q: {row['question']}\nA: {row['answer']}\n---\n"
except Exception as e:
    print(f"Error loading CSV file: {e}")
    exit()

def query_csv_agent(user_query, model_name="phi4-mini:3.8b"):
    url = "http://localhost:11434/api/generate"
    
    # Construct a strong system prompt embedding the entire CSV document
    system_prompt = f"""
    You are an advanced, high-precision AI Knowledge Agent. 
    Below is an absolute ground-truth database containing verified Question-and-Answer pairs.
    Use ONLY this data to answer the user's prompt. If the answer cannot be inferred from the database, 
    politely state that you do not have that information.

    === GROUND TRUTH DATABASE ===
    {csv_context}
    ============================
    """
    
    # We combine system behavior instructions with the user query
    full_prompt = f"{system_prompt}\n\nUser Question: {user_query}\nAgent Answer:"
    
    data = {
        "model": model_name,
        "prompt": full_prompt,
        "options": {
            "num_ctx": 32768  # Expand Ollama's memory window to fit the data easily
        },
        "stream": False
    }
    
    req = urllib.request.Request(
        url, 
        data=json.dumps(data).encode('utf-8'), 
        headers={'Content-Type': 'application/json'}
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            res = json.loads(response.read().decode('utf-8'))
            return res.get('response', '').strip()
    except Exception as e:
        return f"Error connecting to agent backend: {e}"

# Example interactive loop inside the cluster
if __name__ == "__main__":
    print("🤖 CSV Knowledge Agent Initialized on NVIDIA H200.")
    while True:
        user_input = input("\nAsk the agent something (or type 'exit'): ")
        if user_input.lower() == 'exit':
            break
        
        # Phi-4-mini or Qwen3:4b are excellent choices for strict context adherence
        reply = query_csv_agent(user_input, model_name="phi4-mini:3.8b")
        print(f"\nResponse:\n{reply}")