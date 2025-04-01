
Curl Test

curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer <--add LLAMA server key here -->" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Llama-3.2-1B-Instruct",
    "messages": [{"role": "user", "content": "Tell me a fun fact about Jupiter."}],
    "max_tokens": 100
  }'

# llama-install

#!/bin/zsh
set -e

echo "=== Llama 3.2 3B Instruct Setup Script with Transformers (zsh) ==="
echo "This script will set up Llama 3.2 3B Instruct to run locally on your M1 Mac"

# Create a directory for our setup and store its path
SETUP_DIR=~/llama3_setup
mkdir -p $SETUP_DIR
cd $SETUP_DIR
echo "Working directory: $(pwd)"

# Install required packages
echo "=== Installing required Python packages ==="
pip3 install torch transformers huggingface_hub

echo "=== Creating Python script for model usage ==="
cat > $SETUP_DIR/use_llama.py << 'EOF'
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

# Create a text-generation pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Example usage
messages = [
    {"role": "user", "content": "Who are you?"},
]

# Convert messages to prompt format
prompt = ""
for msg in messages:
    if msg["role"] == "user":
        prompt += f"User: {msg['content']}\nAssistant: "

# Generate response
response = pipe(prompt, max_length=200, temperature=0.7)
print(response[0]['generated_text'])
EOF

echo "=== Creating README ==="
cat > $SETUP_DIR/README.md << 'EOF'
# Llama 3.2 3B Instruct Setup

This setup allows you to use the Meta Llama 3.2 3B Instruct model locally using the Hugging Face transformers library.

## Prerequisites

1. Python 3.8 or higher
2. A Hugging Face account with access to the Llama 3.2 model
3. Hugging Face token with appropriate permissions

## Setup

1. Install the required packages:
   ```bash
   pip3 install torch transformers huggingface_hub
   ```

2. Login to Hugging Face:
   ```bash
   huggingface-cli login
   ```

3. Run the model:
   ```bash
   cd ~/llama3_setup  # Make sure you're in the right directory
   python3 use_llama.py
   ```

## Important Notes

- You must have accepted the Llama 3.2 Community License Agreement on Hugging Face
- Make sure you have sufficient disk space for the model
- The model requires authentication through Hugging Face
EOF

# Verify files were created
if [[ ! -f "$SETUP_DIR/use_llama.py" ]]; then
    echo "Error: use_llama.py was not created successfully"
    exit 1
fi

if [[ ! -f "$SETUP_DIR/README.md" ]]; then
    echo "Error: README.md was not created successfully"
    exit 1
fi

echo "=== Setup Complete ==="
echo "Please follow these steps:"
echo "1. Create a Hugging Face account if you haven't already"
echo "2. Visit https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct"
echo "3. Accept the model's license agreement"
echo "4. Run 'huggingface-cli login' and enter your token"
echo "5. Run the following commands:"
echo "   cd $SETUP_DIR"
echo "   python3 use_llama.py"
echo ""
echo "Current directory: $(pwd)"
echo "Script location: $SETUP_DIR/use_llama.py"
