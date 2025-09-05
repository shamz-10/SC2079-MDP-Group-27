#!/usr/bin/env python3
# socket_trace_server.py
import asyncio
import json
import os
import sys
import time
import pathlib
import subprocess

HERE = pathlib.Path(__file__).resolve().parent
GENERATOR_SCRIPT = HERE / "(1jsonupdated)grid_visit_obstacles_ui.py"
OBSTACLES_FILE = HERE / "obstacles.json"
TRACE_FILE = HERE / "movement_trace.json"

# How long to wait for the simulator to produce movement_trace.json (seconds)
TRACE_WAIT_TIMEOUT = 600   # 10 minutes; adjust to your needs
TRACE_POLL_INTERVAL = 0.5  # seconds

def _launch_simulator_interactive() -> subprocess.Popen:
    """
    Launch the simulator so its UI window shows.
    We do NOT force MPLBACKEND=Agg, and we do NOT capture stdout/stderr.
    We do NOT wait() here; we let it run while we watch for the trace file.
    """
    if not GENERATOR_SCRIPT.exists():
        raise FileNotFoundError(f"Generator script not found: {GENERATOR_SCRIPT}")
    if not OBSTACLES_FILE.exists():
        raise FileNotFoundError("obstacles.json not found in server directory.")

    # Important: let the process inherit the current environment
    # Do NOT set MPLBACKEND=Agg (headless) — we want the UI.
    env = os.environ.copy()

    # Use python executable that launched this server
    argv = [sys.executable, str(GENERATOR_SCRIPT), str(OBSTACLES_FILE)]

    # Launch without capture_output so GUI backends aren't hindered
    # cwd=HERE ensures movement_trace.json is written next to this file
    proc = subprocess.Popen(argv, cwd=str(HERE), env=env)
    return proc

def _wait_for_trace_file(timeout_s: float) -> None:
    """
    Block (in thread) until movement_trace.json exists or timeout.
    Doesn't kill the simulator; only waits for the file to appear.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if TRACE_FILE.exists():
            return
        time.sleep(TRACE_POLL_INTERVAL)
    raise TimeoutError(
        f"Timed out waiting for {TRACE_FILE.name}. "
        "Leave the simulator open until it finishes generating."
    )

def _load_trace_dict() -> dict:
    if not TRACE_FILE.exists():
        raise FileNotFoundError("movement_trace.json not found yet.")
    with TRACE_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)

async def ensure_generated_interactive_once():
    """
    On server startup:
      - If obstacles.json exists and movement_trace.json doesn't, start the simulator (UI pops up)
      - Wait (with timeout) for movement_trace.json to appear.
    If movement_trace.json already exists, we don't relaunch the simulator.
    """
    loop = asyncio.get_running_loop()

    if not OBSTACLES_FILE.exists():
        print("[startup] No obstacles.json found; server will wait for it.")
        return

    

    print("[startup] Launching simulator (interactive) to generate movement_trace.json...")
    # Launch simulator in a background OS process (UI visible)
    proc = await loop.run_in_executor(None, _launch_simulator_interactive)

    # Now wait (in a worker thread) for the trace to appear
    try:
        await loop.run_in_executor(None, _wait_for_trace_file, TRACE_WAIT_TIMEOUT)
        print("[startup] movement_trace.json detected.")
    except Exception as e:
        print(f"[startup] Warning: {e} (simulator pid={proc.pid})")

async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    peer = writer.get_extra_info("peername")
    print(f"[client] Connected: {peer}")

    # On each connection, serve the current movement_trace.json if present;
    # If not present but obstacles.json exists, we can (re)launch simulator.
    loop = asyncio.get_running_loop()

    try:
        if not TRACE_FILE.exists() and OBSTACLES_FILE.exists():
            print("[client] No movement_trace.json yet — launching simulator (interactive).")
            await loop.run_in_executor(None, _launch_simulator_interactive)
            # Wait for the file to appear (shorter timeout for on-demand)
            await loop.run_in_executor(None, _wait_for_trace_file, TRACE_WAIT_TIMEOUT)

        trace = await loop.run_in_executor(None, _load_trace_dict)
        line = json.dumps(trace, ensure_ascii=False) + "\n"
        writer.write(line.encode("utf-8"))
        await writer.drain()
        print(f"[client] Sent {len(line)} bytes to {peer}")
    except Exception as e:
        err = json.dumps({"error": str(e)}) + "\n"
        try:
            writer.write(err.encode("utf-8"))
            await writer.drain()
        except Exception:
            pass
        print(f"[client] Error for {peer}: {e}")
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
        print(f"[client] Closed: {peer}")

async def main(host: str = "0.0.0.0", port: int = 9099):
    await ensure_generated_interactive_once()
    server = await asyncio.start_server(handle_client, host=host, port=port)
    addrs = ", ".join(str(sock.getsockname()) for sock in (server.sockets or []))
    print(f"[tcp] Serving movement_trace.json on {addrs}")
    async with server:
        await server.serve_forever()

if __name__ == "__main__":
    # Usage: python socket_trace_server.py [host] [port]
    h = sys.argv[1] if len(sys.argv) > 1 else "0.0.0.0"
    p = int(sys.argv[2]) if len(sys.argv) > 2 else 9099
    try:
        asyncio.run(main(h, p))
    except KeyboardInterrupt:
        print("\n[shutdown] Bye.")
