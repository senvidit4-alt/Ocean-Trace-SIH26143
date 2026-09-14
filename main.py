"""
main.py
=======
Root entrypoint for running the OceanTrace FastAPI backend server.

Usage:
  python main.py
  uvicorn main:app --host 0.0.0.0 --port 8000
"""

import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"Starting OceanTrace Backend on {host}:{port}...", flush=True)
    uvicorn.run("backend.main:app", host=host, port=port, log_level="info")
