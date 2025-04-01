from transformers import AutoModelForCausalLM, AutoTokenizer
import os

hf_token = os.getenv("HF_TOKEN") or ""  # <- Replace if not using env
model_id = "meta-llama/Llama-3.2-1B-Instruct"
target_dir = "./models/meta-llama/Llama-3.2-1B-Instruct_new"

print(f"Downloading model: {model_id} to {target_dir}")

# Download model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=hf_token,
    cache_dir=target_dir
)

# Download tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    token=hf_token,
    cache_dir=target_dir
)

print("✅ Model and tokenizer downloaded successfully.")
