import os
import torch
import psutil
import gc
import signal
from transformers import pipeline
from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
from typing import Optional, List, Union, Dict, Any
import uvicorn
import logging
import platform
import sys
import time
import uuid

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Environment variables
HF_TOKEN = os.getenv("HF_TOKEN")
API_KEY = os.getenv("LLAMA_API_KEY")

# Model configuration
DEFAULT_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_PATH = os.getenv("LLAMA_MODEL_PATH")
LOCAL_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "/meta-llama/Llama-3.2-1B-Instruct_new")
if MODEL_PATH:
    model_id = MODEL_PATH
elif os.path.isdir(LOCAL_MODEL_DIR):
    model_id = LOCAL_MODEL_DIR
else:
    model_id = DEFAULT_MODEL_ID
MODEL_NAME = model_id

# Memory management
def clear_gpu_memory():
    try:
        if torch.cuda.is_available():
            logger.info("Clearing CUDA cache...")
            torch.cuda.empty_cache()
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("Clearing MPS cache...")
            torch.mps.empty_cache()
        logger.info("Running garbage collection...")
        gc.collect()
    except Exception as e:
        logger.error(f"Error clearing memory: {str(e)}")

# Device configuration with M1-specific optimizations
def get_device():
    try:
        # Check if running on M1 Mac
        is_m1 = platform.processor() == 'arm' and platform.system() == 'Darwin'
        logger.info(f"System info: processor={platform.processor()}, system={platform.system()}")
        if is_m1:
            logger.info("Detected M1 Mac")
            if torch.backends.mps.is_available():
                logger.info("MPS (Metal Performance Shaders) is available")
                torch.backends.mps.enable_fallback_to_cpu = True
                return "mps"
            else:
                logger.warning("MPS not available, falling back to CPU")
        elif torch.cuda.is_available():
            logger.info("CUDA is available")
            return "cuda"
        logger.info("Using CPU (no GPU available)")
        return "cpu"
    except Exception as e:
        logger.error(f"Error detecting device: {str(e)}")
        return "cpu"

DEVICE = get_device()

# Initialize model pipeline
pipe = None
model_loading_timeout = 300  # 5 minutes timeout

def timeout_handler(signum, frame):
    raise TimeoutError("Model loading timed out")

def init_model():
    global pipe
    if pipe is None:
        try:
            logger.info("\n" + "="*50)
            logger.info("INITIALIZING MODEL")
            logger.info("="*50)
            logger.info(f"Model: {MODEL_NAME}")
            logger.info(f"Device: {DEVICE}")
            logger.info(f"Torch version: {torch.__version__}")
            logger.info(f"Platform: {platform.platform()}")
            logger.info(f"Python version: {sys.version}")
            if not HF_TOKEN:
                logger.warning("HF_TOKEN not set. Loading model without authentication may be limited.")
            # Set timeout for model loading
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(model_loading_timeout)
            try:
                logger.info("Initializing pipeline...")
                # Clear memory before loading
                clear_gpu_memory()
                # Configure model kwargs based on device
                model_kwargs = {
                    "low_cpu_mem_usage": True,
                    "torch_dtype": torch.float16 if DEVICE in ["mps", "cuda"] else torch.float32,
                    "use_cache": True
                }
                if DEVICE == "mps":
                    model_kwargs.update({
                        "device_map": "auto"
                    })
                logger.info(f"Model kwargs: {model_kwargs}")
                if os.path.isdir(MODEL_NAME):
                    logger.info(f"Loading model from local path: {MODEL_NAME}")
                else:
                    logger.info(f"Loading model from HuggingFace Hub: {MODEL_NAME}")
                pipe = pipeline(
                    "text-generation",
                    model=MODEL_NAME,
                    token=HF_TOKEN,
                    model_kwargs=model_kwargs
                )
                # Disable timeout after successful load
                signal.alarm(0)
                logger.info("Model loaded successfully!")
                # Test the pipeline with a simple prompt
                logger.info("Testing pipeline with a simple prompt...")
                test_output = pipe(
                    "Test message",
                    max_length=10,
                    num_return_sequences=1,
                    do_sample=False,
                    temperature=0.0
                )
                logger.info(f"Test output: {test_output}")
            except TimeoutError:
                logger.error("Model loading timed out")
                raise
            finally:
                signal.alarm(0)
        except Exception as e:
            logger.error(f"ERROR INITIALIZING MODEL: {str(e)}")
            logger.error("Stack trace:")
            import traceback
            logger.error(traceback.format_exc())
            raise


# FastAPI app setup
app = FastAPI(title="Llama API Server")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stop: Optional[Union[str, List[str]]] = None

def verify_api_key(authorization: str = Header(None)):
    if API_KEY:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Invalid or missing Authorization header")
        provided_key = authorization.split("Bearer ", 1)[-1].strip()
        if provided_key != API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API key")
    else:
        logger.warning("LLAMA_API_KEY not set; skipping authentication.")
    return True

@app.get("/")
async def root():
    try:
        return {"status": "running", "info": get_system_info()}
    except Exception as e:
        logger.error(f"Error in root endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, api_key: str = Depends(verify_api_key)):
    try:
        # Validate model name
        req_model_short = request.model.split('/')[-1]
        target_model_short = MODEL_NAME.split('/')[-1] if '/' in MODEL_NAME else MODEL_NAME
        if req_model_short.lower() != target_model_short.lower():
            logger.error(f"Requested model {request.model} not available")
            raise HTTPException(status_code=404, detail=f"Model {request.model} not found")
        if pipe is None:
            init_model()
        # Clear memory before generation
        clear_gpu_memory()
        # Log request
        logger.info(f"Chat completion request - model: {request.model}, messages: {len(request.messages)}, max_tokens: {request.max_tokens}, temperature: {request.temperature}, top_p: {request.top_p}, n: {request.n}")
        # Log last user message for context
        last_user_msg = next((m.content for m in reversed(request.messages) if m.role == 'user'), None)
        if last_user_msg:
            logger.info(f"Last user message: {last_user_msg[:50].replace(chr(10),' ') + ('...' if len(last_user_msg)>50 else '')}")
        # Set generation parameters
        params = {
            'max_new_tokens': request.max_tokens,
            'temperature': request.temperature,
            'top_p': request.top_p,
            'num_return_sequences': request.n
        }
        do_sample_flag = True
        if request.temperature is not None and request.temperature == 0:
            do_sample_flag = False
            params['temperature'] = 0.0
        params['do_sample'] = do_sample_flag
        # Run generation
        start_time = time.time()
        # Convert messages to a prompt string (OpenAI-style)
        prompt = ""
        for msg in request.messages:
            if msg.role == "system":
                prompt += f"<|system|>\n{msg.content}\n"
            elif msg.role == "user":
                prompt += f"<|user|>\n{msg.content}\n"
            elif msg.role == "assistant":
                prompt += f"<|assistant|>\n{msg.content}\n"

        prompt += "<|assistant|>\n"  # Start of model's answer
        outputs = pipe(prompt, **params)

        duration = time.time() - start_time
        logger.info(f"Generation completed in {duration:.2f}s")
        # Process outputs
        choices = []
        for idx, output in enumerate(outputs):
            conv = output.get('generated_text')
            if not conv:
                raise HTTPException(status_code=500, detail="Model did not return text")
            if isinstance(conv, list):
                # conversation list of messages
                assistant_msg = conv[-1]
            else:
                assistant_msg = {'role': 'assistant', 'content': str(conv)}
            content = assistant_msg.get('content', '')
            finish_reason = 'stop'
            # Apply stop sequences if provided
            if request.stop:
                stop_list = [request.stop] if isinstance(request.stop, str) else request.stop
                for stop_seq in stop_list:
                    if stop_seq in content:
                        content = content.split(stop_seq)[0]
                        finish_reason = 'stop'
                        break
            # Check if reached max tokens limit
            if pipe.tokenizer:
                out_tokens = pipe.tokenizer(content)['input_ids']
                if len(out_tokens) >= request.max_tokens:
                    finish_reason = 'length'
            choices.append({
                'index': idx,
                'message': {'role': 'assistant', 'content': content},
                'finish_reason': finish_reason
            })
            logger.info(f"Choice {idx} content: {content[:100].replace(chr(10),' ') + ('...' if len(content)>100 else '')}")
        # Calculate usage
        usage = None
        if pipe.tokenizer:
            prompt_tokens = 0
            for msg in request.messages:
                prompt_tokens += len(pipe.tokenizer(msg.content)['input_ids'])
            completion_tokens = 0
            for choice in choices:
                completion_tokens += len(pipe.tokenizer(choice['message']['content'])['input_ids'])
            usage = {
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': prompt_tokens + completion_tokens
            }
        response = {
            'id': 'chatcmpl-' + uuid.uuid4().hex,
            'object': 'chat.completion',
            'created': int(time.time()),
            'model': target_model_short,
            'choices': choices
        }
        if usage:
            response['usage'] = usage
        logger.info("Chat completion response ready")
        clear_gpu_memory()
        return response
    except Exception as e:
        logger.error(f"Error in /v1/chat/completions: {e}")
        logger.error("Stack trace:", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

def get_system_info():
    """Get system resource information"""
    try:
        return {
            "memory_usage": psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024,  # MB
            "cpu_percent": psutil.Process(os.getpid()).cpu_percent(),
            "device": DEVICE,
            "model_name": MODEL_NAME,
            "model_loaded": pipe is not None,
            "hf_token_configured": bool(HF_TOKEN),
            "torch_version": torch.__version__,
            "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
            "cuda_available": torch.cuda.is_available(),
            "mps_enabled": hasattr(torch.backends, 'mps') and torch.backends.mps.is_built() and torch.backends.mps.is_available(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": sys.version
        }
    except Exception as e:
        logger.error(f"Error getting system info: {str(e)}")
        return {
            "error": str(e),
            "device": DEVICE,
            "model_name": MODEL_NAME
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
