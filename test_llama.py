from transformers import pipeline, BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
import torch
import os
from huggingface_hub import login, HfFolder
import sys
import json
import requests

LOCAL_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "Llama-3.2-1B-Instruct")
DEFAULT_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

def check_auth():
    """Check if user is authenticated with Hugging Face"""
    try:
        if os.path.isdir(LOCAL_MODEL_DIR):
            print(f"Local model directory found at {LOCAL_MODEL_DIR}, skipping Hugging Face login.")
            return True
        if not HfFolder.get_token():
            print("Please login to Hugging Face first:")
            os.system('huggingface-cli login')
        return True
    except Exception as e:
        print(f"Authentication error: {e}")
        return False

def create_agent_prompt(task_type="query", context=None):
    """Create prompts for different agent roles in Sense-Think-Act architecture"""
    base_prompt = {
        "query": "You are an AI assistant specialized in querying ERP databases. Format responses in clear, structured JSON.",
        "analyze": "You are an AI analyst that evaluates ERP data patterns and provides business insights.",
        "action": "You are an AI agent that recommends and executes specific actions based on ERP analysis."
    }
    system_prompt = base_prompt.get(task_type, base_prompt["query"])
    if context:
        system_prompt += f"\nContext: {context}"
    return system_prompt

def test_llama(task_type="query", query=None):
    print("Initializing Llama pipeline...")
    # Configure for Apple M1
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using Apple M1 MPS acceleration")
    else:
        device = "cpu"
        print("MPS not available, falling back to CPU")
    try:
        # Configure model with optimizations for M1 Mac
        model_name = DEFAULT_MODEL_ID
        if os.path.isdir(LOCAL_MODEL_DIR):
            model_name = LOCAL_MODEL_DIR
        # 8-bit quantization optimized for local development (if using CUDA; ignored on MPS/CPU)
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_quant_type="nf8",
            llm_int8_threshold=6.0
        )
        # Pipeline optimized for local inference
        pipe = pipeline(
            "text-generation",
            model=model_name,
            device=device,
            model_kwargs={
                "device_map": "auto",
                "torch_dtype": torch.float16,
                "low_cpu_mem_usage": True,
                "quantization_config": quantization_config,
                "max_memory": {0: "12GB"},  # Limit memory usage for stability
                "offload_folder": "offload"  # Enable disk offloading if needed
            }
        )
        # Create agent-specific prompt
        system_prompt = create_agent_prompt(task_type)
        user_query = query or "Analyze the current ERP system performance and provide insights."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
        print("\nSending prompt to model...")
        response = pipe(
            messages,
            max_length=512,
            temperature=0.5,
            num_return_sequences=1,
            do_sample=True,
            truncation=True,
            top_p=0.85,
            repetition_penalty=1.2,
            max_time=30,
            return_full_text=False
        )
        print("\nModel Response:")
        print("-" * 50)
        print(response[0]['generated_text'])
        print("-" * 50)
        # System info for monitoring
        print("\nSystem Info:")
        print(f"Device: {pipe.device}")
        print(f"Memory Usage: {torch.cuda.memory_allocated()/1024**2:.2f}MB") if torch.cuda.is_available() else None
        return {
            "response": response[0]['generated_text'],
            "model_info": {
                "device": str(pipe.device),
                "model": model_name,
                "task_type": task_type
            }
        }
    except Exception as e:
        error_msg = f"Error during model operation: {str(e)}"
        print(error_msg)
        return {"error": error_msg}

if __name__ == "__main__":
    # Check authentication first
    if not check_auth():
        print("Authentication failed. Please check your Hugging Face token.")
        sys.exit(1)
    # Example usage for different agent roles
    tasks = ["query", "analyze", "action"]
    for task in tasks:
        print(f"\nTesting {task.upper()} agent:")
        result = test_llama(task_type=task)
        if "error" not in result:
            print(f"Success: {task} agent test completed")
            # (No bitsandbytes warning here, as compatibility is handled in pipeline usage)
    # Test local API endpoint for chat completion
    print("\nTesting local API endpoint for chat completion...")
    api_url = os.getenv("LLAMA_API_URL", "http://localhost:8000/v1/chat/completions")
    api_key = os.getenv("LLAMA_API_KEY", "sk-llama-test-key")
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {"model": "Llama-3.2-1B-Instruct", "messages": [{"role": "user", "content": "Hello, how are you?"}], "max_tokens": 50, "temperature": 0.5}
        res = requests.post(api_url, json=payload, headers=headers, timeout=10)
        if res.status_code == 200:
            data = res.json()
            print("API response:", json.dumps(data, indent=2))
        else:
            print(f"API call failed (status {res.status_code}): {res.text}")
    except Exception as e:
        print(f"Failed to connect to API: {e}")
