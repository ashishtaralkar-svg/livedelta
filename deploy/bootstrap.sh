#!/usr/bin/env bash
# One-shot server setup for the Delta bot on a fresh Ubuntu 22.04/24.04 host.
#
# Run it from the repo root AFTER you have created .env with your LIVE keys:
#     cp .env.example .env && nano .env      # paste DELTA_API_KEY / DELTA_API_SECRET
#     bash deploy/bootstrap.sh
#
# It installs Docker (if missing), enables it on boot, then builds and starts the
# bot. `restart: always` in docker-compose.yml plus the enabled Docker service
# means the container survives crashes AND host reboots — true 24/7.

set -euo pipefail

cd "$(dirname "$0")/.."   # repo root, regardless of where this is invoked from

# --- Guard: .env must exist and contain real keys, never the placeholder. ---
if [[ ! -f .env ]]; then
  echo "ERROR: .env not found. Run: cp .env.example .env && nano .env (paste your LIVE keys), then re-run." >&2
  exit 1
fi
if grep -q "your_api_key_here" .env; then
  echo "ERROR: .env still has placeholder keys. Edit .env and paste your real DELTA_API_KEY/SECRET." >&2
  exit 1
fi

# --- 1. Install Docker Engine + compose plugin if missing. ---
if ! command -v docker >/dev/null 2>&1; then
  echo ">> Installing Docker Engine..."
  sudo apt-get update
  sudo apt-get install -y ca-certificates curl gnupg
  sudo install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  sudo chmod a+r /etc/apt/keyrings/docker.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
    | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
  sudo apt-get update
  sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi

# --- 2. Enable Docker on boot (so the bot comes back after a host reboot). ---
sudo systemctl enable --now docker

# --- 3. Lock down .env permissions (it holds live trading credentials). ---
chmod 600 .env

# --- 4. Build the image and start the bot detached. ---
echo ">> Building and starting deltabot..."
sudo docker compose up -d --build

echo ""
echo ">> Done. The bot is running 24/7."
echo "   Tail logs:   sudo docker compose logs -f"
echo "   Restart:     sudo docker compose restart"
echo "   Stop:        sudo docker compose down   (with CLOSE_ON_SHUTDOWN=true it buys back any open option)"
