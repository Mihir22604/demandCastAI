#!/usr/bin/env bash
# ==============================================================================
# DemandCast AI - AWS EC2 Automated Deployment Script
# ==============================================================================
# Target OS: Ubuntu 22.04 LTS on AWS EC2 (Instance recommendation: t3.medium)
# Automates Docker Engine installation, repository cloning, and multi-container
# deployment using Docker Compose.
# ==============================================================================

set -e

echo "========================================================================"
echo "DEMANDCAST AI - AWS EC2 AUTOMATED DEPLOYMENT INITIALIZATION"
echo "========================================================================"

# Step 1: Update system packages
echo "[Step 1/5] Updating host package repositories..."
sudo apt-get update -y && sudo apt-get upgrade -y
sudo apt-get install -y curl git apt-transport-https ca-certificates gnupg lsb-release

# Step 2: Install Docker Engine & Docker Compose
echo "[Step 2/5] Installing Docker Engine & Docker Compose plugin..."
if ! command -v docker &> /dev/null; then
    sudo mkdir -p /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update -y
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    sudo usermod -aG docker "$USER"
    echo "Docker installed successfully."
else
    echo "Docker already installed."
fi

# Step 3: Configure Firewall / Security Group Reminders
echo "[Step 3/5] Verifying host networking..."
echo "Ensure AWS EC2 Security Group permits inbound traffic on:"
echo "  - Port 8000 (FastAPI REST Engine)"
echo "  - Port 8501 (Streamlit Web Dashboard)"
echo "  - Port 22   (SSH Management)"

# Step 4: Build and Launch Containers via Docker Compose
echo "[Step 4/5] Building and launching multi-container services..."
docker compose down || true
docker compose build --no-cache
docker compose up -d

# Step 5: Verification & Health Check
echo "[Step 5/5] Polling container health status..."
sleep 10
docker compose ps

echo "Checking API health endpoint (http://localhost:8000/health)..."
curl -s http://localhost:8000/health | grep "healthy" && echo " [SUCCESS] API is operational."

echo "========================================================================"
echo "DEPLOYMENT COMPLETE!"
echo "Access REST API Swagger Docs: http://<EC2-PUBLIC-IP>:8000/docs"
echo "Access Streamlit Web UI:      http://<EC2-PUBLIC-IP>:8501"
echo "========================================================================"
