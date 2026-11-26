#!/usr/bin/env bash
# Launch Stage 2 MPS backend + Vite viewer together; clean shutdown on Ctrl+C.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

STAGE2_PORT="${STAGE2_PORT:-8765}"
VIEWER_PORT="${VIEWER_PORT:-5173}"
STAGE2_PID=""
VIEWER_PID=""

cleanup() {
  trap - EXIT INT TERM
  echo ""
  echo "[dev] Shutting down..."
  if [[ -n "${VIEWER_PID}" ]] && kill -0 "${VIEWER_PID}" 2>/dev/null; then
    kill "${VIEWER_PID}" 2>/dev/null || true
    wait "${VIEWER_PID}" 2>/dev/null || true
  fi
  if [[ -n "${STAGE2_PID}" ]] && kill -0 "${STAGE2_PID}" 2>/dev/null; then
    kill "${STAGE2_PID}" 2>/dev/null || true
    wait "${STAGE2_PID}" 2>/dev/null || true
  fi
  # Clear stale listeners if children ignored SIGTERM
  for port in "${STAGE2_PORT}" "${VIEWER_PORT}"; do
    if command -v lsof >/dev/null 2>&1; then
      pids="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
      if [[ -n "$pids" ]]; then
        # intentional unquoted: lsof may return multiple PIDs
        # shellcheck disable=SC2086
        kill $pids 2>/dev/null || true
      fi
    fi
  done
  echo "[dev] Done."
}
trap cleanup EXIT INT TERM

wait_http() {
  local url="$1"
  local label="$2"
  local tries="${3:-90}"
  for ((i = 1; i <= tries; i++)); do
    if curl -sf --max-time 1 "$url" >/dev/null 2>&1; then
      echo "[dev] $label ready."
      return 0
    fi
    sleep 0.5
  done
  echo "[dev] Timed out waiting for $label ($url)" >&2
  return 1
}

if lsof -tiTCP:"$STAGE2_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "[dev] Port $STAGE2_PORT already in use — reusing existing Stage 2 server."
else
  echo "[dev] Starting Stage 2 server on :${STAGE2_PORT}..."
  uv run --extra diffusion python experiments/stage2_server.py --port "$STAGE2_PORT" &
  STAGE2_PID=$!
fi

if lsof -tiTCP:"$VIEWER_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "[dev] Port $VIEWER_PORT already in use — reusing existing Vite viewer."
else
  echo "[dev] Starting Vite viewer on :${VIEWER_PORT}..."
  (cd viewer && npm run dev -- --host 127.0.0.1 --port "$VIEWER_PORT") &
  VIEWER_PID=$!
fi

wait_http "http://127.0.0.1:$VIEWER_PORT/" "Vite viewer" 60 || true

# The Stage 2 server can take minutes to start listening on a first run (uv installs PyTorch
# into this checkout's venv), so wait as long as its process is alive rather than a fixed time.
stage2_listening() {
  curl -sf --max-time 1 "http://127.0.0.1:$STAGE2_PORT/api/health" 2>/dev/null
}
echo "[dev] Waiting for the Stage 2 server (first run in a checkout installs PyTorch; this can take a few minutes)..."
waited=0
until stage2_listening >/dev/null; do
  if [[ -n "${STAGE2_PID}" ]] && ! kill -0 "${STAGE2_PID}" 2>/dev/null; then
    echo "[dev] Stage 2 server exited before it started listening." >&2
    exit 1
  fi
  sleep 1
  waited=$((waited + 1))
  if ((waited % 30 == 0)); then
    echo "[dev] ...still waiting for the Stage 2 server (${waited}s)"
  fi
done
echo "[dev] Stage 2 server is listening; waiting for the model to load..."
until stage2_listening | grep -q '"ready": true'; do
  if stage2_listening | grep -q '"error": "'; then
    echo "[dev] Stage 2 model failed to load (see the error above). The viewer still works on cached worlds." >&2
    break
  fi
  if [[ -n "${STAGE2_PID}" ]] && ! kill -0 "${STAGE2_PID}" 2>/dev/null; then
    echo "[dev] Stage 2 server exited while loading the model." >&2
    exit 1
  fi
  sleep 1
done

echo ""
echo "[dev] TerraGrok ready:"
echo "      Planet / Explorer → http://127.0.0.1:$VIEWER_PORT/planet.html"
echo "      Stage 2 API       → http://127.0.0.1:$STAGE2_PORT/api/health"
echo "      Ctrl+C to stop."
echo ""

# Keep foreground alive while children run
while true; do
  if [[ -n "$STAGE2_PID" ]] && ! kill -0 "$STAGE2_PID" 2>/dev/null; then
    echo "[dev] Stage 2 server exited unexpectedly." >&2
    exit 1
  fi
  if [[ -n "$VIEWER_PID" ]] && ! kill -0 "$VIEWER_PID" 2>/dev/null; then
    echo "[dev] Vite viewer exited unexpectedly." >&2
    exit 1
  fi
  # If both were pre-existing, just idle until Ctrl+C
  if [[ -z "$STAGE2_PID" && -z "$VIEWER_PID" ]]; then
    sleep 3600
  else
    sleep 2
  fi
done
