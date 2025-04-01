import os
import uvicorn

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print(f"Starting LLaMA API server on port {port}...")
    uvicorn.run("llama_server:app", host="0.0.0.0", port=port, log_level="info")
