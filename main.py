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
from backend.main import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"Starting OceanTrace Backend on {host}:{port}...")
    uvicorn.run(app, host=host, port=port)
