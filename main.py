"""
main.py
=======
Root entrypoint for running the OceanTrace FastAPI backend server.

Usage:
  python main.py
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import uvicorn
from backend.main import app

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True, reload_dirs=["backend", "modules"])
