#how to run:"
# 1. run pip install fastapi uvicorn
# 2. pip install python-multipart
# 3. uvicorn app:app --reload
# 
# 
# "








# app.py
import json
import os
import sys
import asyncio
import datetime
import pathlib
import tempfile
import subprocess
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------- Config ----------
HERE = pathlib.Path(__file__).resolve().parent
# Exact filename with parentheses as you wrote it:
GENERATOR_SCRIPT = HERE / "(1jsonupdated)grid_visit_obstacles_ui.py"

OBSTACLES_FILE = HERE / "obstacles.json"
TRACE_FILE = HERE / "movement_trace.json"

# If you want to override the script path via env var:
#   export GENERATOR_SCRIPT=/abs/path/to/(1jsonupdated)grid_visit_obstacles_ui.py
GENERATOR_SCRIPT = pathlib.Path(os.getenv("GENERATOR_SCRIPT", str(GENERATOR_SCRIPT)))

# ---------- App setup ----------
app = FastAPI(title="Movement Trace API", version="1.0.0")

# (Optional) open CORS for Postman or other tools
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory cache + lock
cache_lock = asyncio.Lock()
cached_trace: Optional[dict] = None
cached_at: Optional[str] = None


# ---------- Helpers ----------
def _run_generator(obstacles_path: pathlib.Path) -> None:
    """
    Runs the external generator script synchronously:
      python "(1jsonupdated)grid_visit_obstacles_ui.py" obstacles.json
    Expects the script to write movement_trace.json in the same directory.
    """
    if not GENERATOR_SCRIPT.exists():
        raise FileNotFoundError(f"Generator script not found: {GENERATOR_SCRIPT}")

    # Build argv list (no shell, no quoting issues, even with parentheses)
    argv = [sys.executable, str(GENERATOR_SCRIPT), str(obstacles_path)]
    # Run in project directory so the script writes movement_trace.json here
    res = subprocess.run(argv, cwd=str(HERE), capture_output=True, text=True)
    if res.returncode != 0:
        # Bubble up stderr to help debugging
        raise RuntimeError(
            f"Generator failed (exit {res.returncode}).\nSTDERR:\n{res.stderr}\nSTDOUT:\n{res.stdout}"
        )


def _load_trace_file(trace_path: pathlib.Path) -> dict:
    if not trace_path.exists():
        raise FileNotFoundError(f"Trace file not found: {trace_path}")
    with trace_path.open("r", encoding="utf-8") as f:
        return json.load(f)


async def generate_and_cache(obstacles_path: pathlib.Path) -> dict:
    global cached_trace, cached_at  # <-- move here
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _run_generator, obstacles_path)

    trace = await loop.run_in_executor(None, _load_trace_file, TRACE_FILE)

    async with cache_lock:
        cached_trace = trace
        cached_at = datetime.datetime.utcnow().isoformat() + "Z"
    return trace



# ---------- Models ----------
class ObstaclesBody(BaseModel):
    # Accept arbitrary JSON; we don’t validate its schema here
    data: dict


# ---------- Lifecycle ----------
@app.on_event("startup")
async def on_startup():
    """
    On startup:
      - If obstacles.json exists, run the generator once so /trace is ready.
      - If it doesn't exist, we just start the API (you can POST /run later).
    """
    if OBSTACLES_FILE.exists():
        try:
            await generate_and_cache(OBSTACLES_FILE)
            print("[startup] Loaded movement_trace.json into cache.")
        except Exception as e:
            # Don't crash the server; just log the error
            print(f"[startup] Warning: failed to precompute trace: {e}")
    else:
        print("[startup] No obstacles.json found; waiting for /run.")


# ---------- Routes ----------
@app.get("/status")
async def status():
    """
    Returns basic status + timestamps.
    """
    async with cache_lock:
        return {
            "generator_script": str(GENERATOR_SCRIPT),
            "obstacles_present": OBSTACLES_FILE.exists(),
            "trace_present": TRACE_FILE.exists(),
            "cached": cached_trace is not None,
            "cached_at": cached_at,
        }


@app.get("/trace")
async def get_trace():
    global cached_trace, cached_at  # <-- move here
    async with cache_lock:
        if cached_trace is not None:
            return JSONResponse(content=cached_trace)

    try:
        trace = await asyncio.get_running_loop().run_in_executor(None, _load_trace_file, TRACE_FILE)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="movement_trace.json not found. Run /run first.")
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"movement_trace.json is invalid JSON: {e}")

    async with cache_lock:
        cached_trace = trace
        cached_at = datetime.datetime.utcnow().isoformat() + "Z"
    return JSONResponse(content=trace)



@app.post("/run")
async def run_generator(
    background: bool = Body(False, embed=True),
    # Option A: send a new obstacles JSON in the body: {"data": {...}}
    obstacles: Optional[ObstaclesBody] = None,
    # Option B: upload a file instead (e.g. obstacles.json)
    obstacles_file: Optional[UploadFile] = File(None),
):
    """
    Re-runs the generator. Two ways to provide obstacles:
    - JSON body: {"data": {...}}
    - File upload: form-data field 'obstacles_file' with a JSON file

    If neither is provided and obstacles.json exists, it will reuse that.
    Set body { "background": true } to return immediately and let it compute in the background.
    """
    # Determine obstacles source
    obstacles_path = None

    # If a file uploaded
    if obstacles_file is not None:
        if not obstacles_file.filename.lower().endswith(".json"):
            raise HTTPException(status_code=400, detail="Uploaded file must be .json")
        # Save to a temp file, then atomically replace obstacles.json
        contents = await obstacles_file.read()
        try:
            parsed = json.loads(contents.decode("utf-8"))
        except Exception:
            raise HTTPException(status_code=400, detail="Uploaded file is not valid JSON.")
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json", dir=str(HERE)) as tmp:
            json.dump(parsed, tmp, ensure_ascii=False, indent=2)
            tmp_path = pathlib.Path(tmp.name)
        tmp_path.replace(OBSTACLES_FILE)
        obstacles_path = OBSTACLES_FILE

    # If JSON body provided
    elif obstacles is not None:
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json", dir=str(HERE)) as tmp:
            json.dump(obstacles.data, tmp, ensure_ascii=False, indent=2)
            tmp_path = pathlib.Path(tmp.name)
        tmp_path.replace(OBSTACLES_FILE)
        obstacles_path = OBSTACLES_FILE

    # If neither provided, reuse the existing obstacles.json if it exists
    else:
        if not OBSTACLES_FILE.exists():
            raise HTTPException(
                status_code=400,
                detail="No obstacles provided and obstacles.json not found. Upload or send JSON.",
            )
        obstacles_path = OBSTACLES_FILE

    # Run now (foreground) or schedule (background) depending on flag
    if background:
        # Fire-and-forget background task
        asyncio.create_task(generate_and_cache(obstacles_path))
        return {"message": "Generation started in background."}
    else:
        # Run and wait, then return the resulting trace
        try:
            trace = await generate_and_cache(obstacles_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Generator failed: {e}")
        return JSONResponse(content=trace)
