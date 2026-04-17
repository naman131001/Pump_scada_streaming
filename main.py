"""
main.py — FastAPI backend for Pump SCADA Simulator
═══════════════════════════════════════════════════
WebSocket streaming + REST endpoints for scenario injection.
All computation uses pump_engine.py (identical to original s.py logic).
"""

import asyncio
import json
import os
import io
import logging
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Optional

logger = logging.getLogger("pump_scada")

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

from pump_engine import (
    PumpState,
    N_PUMPS,
    CAUSE_FAILURE,
    CAUSE_LEAKAGE,
    FAILURE_LABEL,
    LEAKAGE_LABEL,
    OUTPUT_COLUMNS,
    get_utc_timestamp_floored,
)

# ── Azure Blob Storage ────────────────────────────────────────────────────
AZURE_STORAGE_CONN_STR = os.getenv("AZURE_STORAGE_CONN_STR")
AZURE_CONTAINER_NAME   = os.getenv("AZURE_CONTAINER_NAME")

# ── Azure IoT Hub (per-pump device connections) ──────────────────────────
IOT_CONN_STRINGS: dict[int, str] = {
    1: os.getenv("IOT_CONN_STRINGS_1"),
    2: os.getenv("IOT_CONN_STRINGS_2"),
    3: os.getenv("IOT_CONN_STRINGS_3"),
    4: os.getenv("IOT_CONN_STRINGS_4"),
    5: os.getenv("IOT_CONN_STRINGS_5"),
}

# IoT Hub device clients (lazily connected)
_iot_clients: dict[int, object] = {}

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL STATE
# ══════════════════════════════════════════════════════════════════════════════

class SimState:
    def __init__(self):
        self.pumps: list[PumpState] = [PumpState(pid) for pid in range(1, N_PUMPS + 1)]
        self.running: bool = False
        self.tick_count: int = 0
        self.interval_sec: float = 5.0
        self.collected_rows: list[dict] = []
        self.ws_clients: set[WebSocket] = set()
        self._task: Optional[asyncio.Task] = None

    def reset_pumps(self):
        self.pumps = [PumpState(pid) for pid in range(1, N_PUMPS + 1)]
        self.tick_count = 0
        self.collected_rows = []

sim = SimState()


# ══════════════════════════════════════════════════════════════════════════════
# STREAMING LOOP
# ══════════════════════════════════════════════════════════════════════════════

async def broadcast(data: dict):
    dead = set()
    for ws in sim.ws_clients:
        try:
            await ws.send_json(data)
        except Exception:
            dead.add(ws)
    sim.ws_clients -= dead


# ── Parquet blob writer ───────────────────────────────────────────────────
# Writes each tick as an individual Parquet file under:
#   landing/pump_scada_py/date=YYYY-MM-DD/tick_<count>_<HHMMSSffffff>.parquet

BLOB_FOLDER = "pump_scada_py"


# ── IoT Hub helpers ────────────────────────────────────────────────────────

def _get_iot_client(pump_id: int):
    """Return a connected IoTHubDeviceClient for the given pump, or None."""
    if pump_id not in IOT_CONN_STRINGS:
        return None
    if pump_id in _iot_clients:
        return _iot_clients[pump_id]
    try:
        from azure.iot.device import IoTHubDeviceClient
        client = IoTHubDeviceClient.create_from_connection_string(
            IOT_CONN_STRINGS[pump_id]
        )
        client.connect()
        _iot_clients[pump_id] = client
        logger.info(f"IoT Hub: connected pump {pump_id}")
        return client
    except Exception as e:
        logger.warning(f"IoT Hub: failed to connect pump {pump_id}: {e}")
        return None


async def _send_tick_to_iot_hub(rows: list[dict]):
    """Send each pump's row as a D2C message to its IoT Hub device."""
    loop = asyncio.get_event_loop()
    for row in rows:
        pump_id = row.get("pump_id")
        client = _get_iot_client(pump_id)
        if client is None:
            continue
        try:
            from azure.iot.device import Message
            payload = json.dumps(row)
            msg = Message(payload)
            msg.content_type = "application/json"
            msg.content_encoding = "utf-8"
            # Run blocking send_message in a thread so we don't block the event loop
            await loop.run_in_executor(None, client.send_message, msg)
        except Exception as e:
            logger.warning(f"IoT Hub send failed for pump {pump_id}: {e}")
            # Drop stale client so it reconnects on next tick
            _iot_clients.pop(pump_id, None)


async def _write_tick_to_parquet(rows: list[dict], tick_count: int):
    """Serialize rows to Parquet in-memory and upload to the date partition."""
    if not AZURE_STORAGE_CONN_STR:
        return
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        from azure.storage.blob import BlobServiceClient

        now_utc = datetime.now(timezone.utc)
        date_str = now_utc.strftime("%Y-%m-%d")
        time_str = now_utc.strftime("%H%M%S%f")

        # Hive-style partition path
        blob_path = f"{BLOB_FOLDER}/date={date_str}/tick_{tick_count:08d}_{time_str}.parquet"

        # Build PyArrow Table with explicit schema matching OUTPUT_COLUMNS
        schema = pa.schema([
            pa.field("pump_id",                              pa.int32()),
            pa.field("timestamp",                            pa.string()),
            pa.field("cycle_number",                         pa.int32()),
            pa.field("vibration",                            pa.float64()),
            pa.field("motor_temp",                           pa.float64()),
            pa.field("current_per_flow",                     pa.float64()),
            pa.field("flow_deviation",                       pa.float64()),
            pa.field("pressure_delta",                       pa.float64()),
            pa.field("wear_ratio",                           pa.float64()),
            pa.field("probability_of_failure_in_percentage", pa.float64()),
            pa.field("probability_of_leakage_in_percentage", pa.float64()),
            pa.field("leakage_flag",                         pa.int32()),
            pa.field("leakage_reason",                       pa.string()),
            pa.field("failure_flag",                         pa.int32()),
            pa.field("failure_reason",                       pa.string()),
        ])

        arrays = {col: [row[col] for row in rows] for col in OUTPUT_COLUMNS}
        table = pa.table(arrays, schema=schema)

        # Serialize to in-memory Parquet bytes
        buf = io.BytesIO()
        pq.write_table(table, buf, compression="snappy")
        buf.seek(0)

        # Upload to Azure Blob
        blob_service = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONN_STR)
        container = blob_service.get_container_client(AZURE_CONTAINER_NAME)
        try:
            container.get_container_properties()
        except Exception:
            container.create_container()

        container.upload_blob(blob_path, buf.read(), overwrite=True)

    except ImportError:
        logger.warning("pyarrow not installed — run: pip install pyarrow")
    except Exception as e:
        logger.warning(f"Parquet blob write failed (tick {tick_count}): {e}")


async def stream_loop():
    while sim.running:
        now = get_utc_timestamp_floored()
        rows = [pump.next_row(now) for pump in sim.pumps]
        sim.tick_count += 1
        sim.collected_rows.extend(rows)

        # Build pump status info
        pump_status = []
        for p in sim.pumps:
            pump_status.append({
                "pump_id": p.pump_id,
                "type": p.current_type,
                "cause": p.current_cause,
                "tick": p.tick,
                "wear": round(p.wear, 5),
            })

        payload = {
            "event": "tick",
            "tick_count": sim.tick_count,
            "rows": rows,
            "pump_status": pump_status,
        }
        await broadcast(payload)

        # Auto-save this tick's rows as Parquet to Azure Blob (non-blocking)
        asyncio.create_task(_write_tick_to_parquet(rows, sim.tick_count))

        # Send each pump's data to its IoT Hub device (non-blocking)
        asyncio.create_task(_send_tick_to_iot_hub(rows))

        await asyncio.sleep(sim.interval_sec)


# ══════════════════════════════════════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    sim.running = False
    if sim._task and not sim._task.done():
        sim._task.cancel()
    # Disconnect all IoT Hub clients on shutdown
    for pid, client in _iot_clients.items():
        try:
            client.disconnect()
        except Exception:
            pass
    _iot_clients.clear()

app = FastAPI(title="Pump SCADA Simulator", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ────────────────────────────────────────────────────────────────────

class StartRequest(BaseModel):
    interval_sec: float = 5.0

class InjectRequest(BaseModel):
    pump_id: int
    scenario_type: str   # "failure" or "leakage"
    cause: str           # e.g. "bearing", "pipe_joint"


# ── REST endpoints ────────────────────────────────────────────────────────────

@app.get("/api/status")
async def get_status():
    return {
        "running": sim.running,
        "tick_count": sim.tick_count,
        "total_rows": len(sim.collected_rows),
        "interval_sec": sim.interval_sec,
        "pumps": [
            {
                "pump_id": p.pump_id,
                "type": p.current_type,
                "cause": p.current_cause,
                "tick": p.tick,
                "wear": round(p.wear, 5),
            }
            for p in sim.pumps
        ],
    }


@app.post("/api/start")
async def start_stream(req: StartRequest):
    if sim.running:
        return {"status": "already_running"}
    sim.interval_sec = max(0.5, req.interval_sec)
    sim.running = True
    sim._task = asyncio.create_task(stream_loop())
    return {"status": "started", "interval_sec": sim.interval_sec}


@app.post("/api/stop")
async def stop_stream():
    if not sim.running:
        return {"status": "already_stopped"}
    sim.running = False
    if sim._task and not sim._task.done():
        sim._task.cancel()
    return {"status": "stopped", "tick_count": sim.tick_count}


@app.post("/api/reset")
async def reset():
    sim.running = False
    if sim._task and not sim._task.done():
        sim._task.cancel()
    sim.reset_pumps()
    await broadcast({"event": "reset"})
    return {"status": "reset"}


@app.post("/api/inject")
async def inject_scenario(req: InjectRequest):
    valid_causes = CAUSE_FAILURE + CAUSE_LEAKAGE
    if req.cause not in valid_causes:
        raise HTTPException(400, f"Invalid cause. Valid: {valid_causes}")
    if req.scenario_type not in ("failure", "leakage"):
        raise HTTPException(400, "scenario_type must be 'failure' or 'leakage'")
    if req.pump_id < 1 or req.pump_id > N_PUMPS:
        raise HTTPException(400, f"pump_id must be 1..{N_PUMPS}")

    pump = sim.pumps[req.pump_id - 1]
    pump.inject_scenario(req.scenario_type, req.cause)

    label = FAILURE_LABEL.get(req.cause) or LEAKAGE_LABEL.get(req.cause, req.cause)
    await broadcast({
        "event": "inject",
        "pump_id": req.pump_id,
        "scenario_type": req.scenario_type,
        "cause": req.cause,
        "label": label,
    })
    return {"status": "injected", "pump_id": req.pump_id, "label": label}


@app.post("/api/clear-injection")
async def clear_injection(pump_id: int = 1):
    if pump_id < 1 or pump_id > N_PUMPS:
        raise HTTPException(400, f"pump_id must be 1..{N_PUMPS}")
    sim.pumps[pump_id - 1]._inject = None
    return {"status": "cleared", "pump_id": pump_id}


@app.get("/api/scenarios")
async def list_scenarios():
    return {
        "failures": [
            {"cause": c, "label": FAILURE_LABEL[c], "type": "failure"}
            for c in CAUSE_FAILURE
        ],
        "leakages": [
            {"cause": c, "label": LEAKAGE_LABEL[c], "type": "leakage"}
            for c in CAUSE_LEAKAGE
        ],
    }


@app.post("/api/save-to-blob")
async def save_to_blob():
    """Save all collected rows as a single Parquet snapshot to pump_scada_py/."""
    if not sim.collected_rows:
        raise HTTPException(400, "No data to save")

    import pyarrow as pa
    import pyarrow.parquet as pq

    now_utc = datetime.now(timezone.utc)
    date_str = now_utc.strftime("%Y-%m-%d")
    ts_str   = now_utc.strftime("%Y%m%d_%H%M%S")
    blob_path = f"{BLOB_FOLDER}/date={date_str}/snapshot_{ts_str}.parquet"

    schema = pa.schema([
        pa.field("pump_id",                              pa.int32()),
        pa.field("timestamp",                            pa.string()),
        pa.field("cycle_number",                         pa.int32()),
        pa.field("vibration",                            pa.float64()),
        pa.field("motor_temp",                           pa.float64()),
        pa.field("current_per_flow",                     pa.float64()),
        pa.field("flow_deviation",                       pa.float64()),
        pa.field("pressure_delta",                       pa.float64()),
        pa.field("wear_ratio",                           pa.float64()),
        pa.field("probability_of_failure_in_percentage", pa.float64()),
        pa.field("probability_of_leakage_in_percentage", pa.float64()),
        pa.field("leakage_flag",                         pa.int32()),
        pa.field("leakage_reason",                       pa.string()),
        pa.field("failure_flag",                         pa.int32()),
        pa.field("failure_reason",                       pa.string()),
    ])
    arrays = {col: [row[col] for row in sim.collected_rows] for col in OUTPUT_COLUMNS}
    table = pa.table(arrays, schema=schema)

    buf = io.BytesIO()
    pq.write_table(table, buf, compression="snappy")
    buf.seek(0)
    parquet_bytes = buf.read()

    if not AZURE_STORAGE_CONN_STR:
        local_path = os.path.join(os.path.dirname(__file__), "..", "data")
        os.makedirs(local_path, exist_ok=True)
        filepath = os.path.join(local_path, f"snapshot_{ts_str}.parquet")
        with open(filepath, "wb") as f:
            f.write(parquet_bytes)
        return {
            "status": "saved_locally",
            "path": filepath,
            "rows": len(sim.collected_rows),
        }

    try:
        from azure.storage.blob import BlobServiceClient
        blob_service = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONN_STR)
        container = blob_service.get_container_client(AZURE_CONTAINER_NAME)
        try:
            container.get_container_properties()
        except Exception:
            container.create_container()
        container.upload_blob(blob_path, parquet_bytes, overwrite=True)
        return {
            "status": "saved_to_blob",
            "container": AZURE_CONTAINER_NAME,
            "blob": blob_path,
            "rows": len(sim.collected_rows),
        }
    except ImportError:
        raise HTTPException(500, "azure-storage-blob not installed")
    except Exception as e:
        raise HTTPException(500, f"Blob upload failed: {e}")


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    sim.ws_clients.add(ws)
    # Send current status on connect
    await ws.send_json({
        "event": "connected",
        "running": sim.running,
        "tick_count": sim.tick_count,
        "n_pumps": N_PUMPS,
    })
    try:
        while True:
            await ws.receive_text()  # keep alive
    except WebSocketDisconnect:
        sim.ws_clients.discard(ws)
