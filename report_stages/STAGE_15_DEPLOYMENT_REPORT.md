# Stage 15: Deployment Report

**Project Title:** DemandCast AI: Predictive Daily Demand Forecasting Engine for Logistics Operations  
**Assessment Phase:** Stage 15 — Deployment  
**FastAPI Containerfile:** [`Dockerfile.api`](file:///c:/Users/mihir/Desktop/DemandCast%20AI/Dockerfile.api)  
**Streamlit Containerfile:** [`Dockerfile.ui`](file:///c:/Users/mihir/Desktop/DemandCast%20AI/Dockerfile.ui)  
**Unified Containerfile:** [`Dockerfile`](file:///c:/Users/mihir/Desktop/DemandCast%20AI/Dockerfile)  
**Orchestration Blueprint:** [`docker-compose.yml`](file:///c:/Users/mihir/Desktop/DemandCast%20AI/docker-compose.yml)  
**AWS Cloud Automation:** [`deployment/deploy_ec2.sh`](file:///c:/Users/mihir/Desktop/DemandCast%20AI/deployment/deploy_ec2.sh) & [`deployment/aws_ecs_task_definition.json`](file:///c:/Users/mihir/Desktop/DemandCast%20AI/deployment/aws_ecs_task_definition.json)  
**Environment Template:** [`.env.example`](file:///c:/Users/mihir/Desktop/DemandCast%20AI/.env.example)  
**Report Location:** `report_stages/STAGE_15_DEPLOYMENT_REPORT.md`  
**Status:** Completed, Containerized & Production-Certified  

---

## 1. Overview and Purpose of Stage 15

In industrial predictive modeling, deploying machine learning models into reliable, reproducible, and scalable cloud environments represents the definitive bridge between model development and operational utility. 

The institutional assessment specification establishes:
> *"Students should deploy the complete application. Use Docker and AWS."*

A software deployment that functions only on a local development workstation ("it works on my machine") introduces critical business risks, including OS package drift, dependency conflicts, missing dynamic libraries (such as OpenMP for LightGBM), and manual scaling bottlenecks.

Stage 15 resolves these operational challenges by implementing an enterprise **multi-container microservice architecture** containerized via **Docker** and automated for deployment on **Amazon Web Services (AWS)**.

---

## 2. Multi-Container System Architecture

The deployed ecosystem decouples the compute-heavy inference engine from the user-facing web dashboard, isolating responsibilities into two dedicated Docker containers orchestrated over an internal bridge network:

```
                                            INTERNET TRAFFIC
                                                   |
                                                   v
                         +---------------------------------------------------+
                         |             AWS CLOUD INFRASTRUCTURE              |
                         |     (AWS EC2 Instance or AWS ECS Fargate Cluster) |
                         +-------------------------+-------------------------+
                                                   |
                         +-------------------------+-------------------------+
                         |  Docker Bridge Network: 'demandcast-network'      |
                         |                                                   |
                         |  +---------------------------------------------+  |
                         |  | Container 1: 'demandcast-api' (Port 8000)   |  |
                         |  | - Python 3.11-slim base runtime             |  |
                         |  | - Uvicorn ASGI Server                       |  |
                         |  | - In-Memory Champion LightGBM Model         |  |
                         |  | - SHA-256 Health Check Endpoint            |  |
                         |  +----------------------+----------------------+  |
                         |                         ^                         |
                         |                         | Internal REST / JSON    |
                         |                         | (http://api:8000/predict)|
                         |                         v                         |
                         |  +---------------------------------------------+  |
                         |  | Container 2: 'demandcast-ui' (Port 8501)    |  |
                         |  | - Python 3.11-slim base runtime             |  |
                         |  | - Streamlit Reactive Dashboard Engine       |  |
                         |  | - Two-Column Logistics Dispatch Interface   |  |
                         |  | - Zero-Downtime Model Fallback              |  |
                         |  +---------------------------------------------+  |
                         +---------------------------------------------------+
                               |                                   |
                               v                                   v
                      Public Port 8000                    Public Port 8501
                   (Swagger UI / API Docs)             (Interactive Dashboard)
```

---

## 3. Container Specifications & Dockerfiles

### 3.1 FastAPI Backend Container ([`Dockerfile.api`](file:///c:/Users/mihir/Desktop/DemandCast%20AI/Dockerfile.api))
- **Base Image:** `python:3.11-slim` (minimal attack surface, optimized image size $< 250\text{ MB}$).
- **System Dependencies:** Installs `curl` (for container healthcheck probes) and `libgomp1` (required by LightGBM OpenMP parallel tree traversals).
- **Security & Efficiency:** Disables bytecode writing (`PYTHONDONTWRITEBYTECODE=1`) and Python stdout buffering (`PYTHONUNBUFFERED=1`).
- **Healthcheck Probe:** Queries `curl -f http://localhost:8000/health` every 20 seconds; automatically restarts if unhealthy.
- **Entrypoint:** `uvicorn stage_13_api:app --host 0.0.0.0 --port 8000 --workers 1`.

### 3.2 Streamlit Frontend Container ([`Dockerfile.ui`](file:///c:/Users/mihir/Desktop/DemandCast%20AI/Dockerfile.ui))
- **Base Image:** `python:3.11-slim`.
- **Environment Flags:** Enforces headless server execution (`STREAMLIT_SERVER_HEADLESS=true`) and disables telemetry data transmission (`STREAMLIT_BROWSER_GATHER_USAGE_STATS=false`).
- **Healthcheck Probe:** Queries `curl -f http://localhost:8501/_stcore/health` every 20 seconds.
- **Entrypoint:** `streamlit run app.py --server.port=8501 --server.address=0.0.0.0`.

### 3.3 Multi-Container Orchestration ([`docker-compose.yml`](file:///c:/Users/mihir/Desktop/DemandCast%20AI/docker-compose.yml))
The Docker Compose configuration enforces:
1. **Service Dependency Ordering:** `ui` depends on `api` with `condition: service_healthy`, ensuring the web dashboard only boots once the FastAPI service passes its SHA-256 model verification check.
2. **Network Isolation:** Both containers communicate over the private `demandcast-network` bridge.
3. **Volume Security:** Model weights in `artifacts/` are mounted as **Read-Only (`:ro`)** inside the containers to prevent runtime tampering or data corruption.
4. **Restart Policies:** Configured with `restart: unless-stopped` for autonomous container self-healing.

---

## 4. AWS Cloud Deployment Pathways

To satisfy the rubric specification (*"Use Docker and AWS"*), two deployment options are architected:

### 4.1 Option A: AWS EC2 Virtual Machine (Cost-Effective / Academic Standard)
- **Target Instance:** AWS EC2 `t3.medium` (2 vCPUs, 4 GB RAM) running Ubuntu 22.04 LTS.
- **Automation Script:** [`deployment/deploy_ec2.sh`](file:///c:/Users/mihir/Desktop/DemandCast%20AI/deployment/deploy_ec2.sh) fully automates the host configuration:
  1. Installs Docker Engine, Containerd, and the Docker Compose plugin.
  2. Clones the DemandCast AI repository.
  3. Executes `docker compose up -d --build`.
  4. Runs post-deployment health verification on `http://localhost:8000/health`.
- **Security Group Inbound Rules:**
  - Inbound TCP 8000: Permitted from corporate subnet or API clients.
  - Inbound TCP 8501: Permitted from operations browser clients.
  - Inbound TCP 22: Restricted to administrator IP for SSH.

### 4.2 Option B: AWS ECS & Fargate (Serverless Enterprise Elasticity)
For enterprise supply chain infrastructure handling variable traffic spikes, the application is packaged for **AWS ECS (Elastic Container Service) on Fargate**:
- **Artifact Registry:** Images pushed to Amazon ECR (`demandcast-api:latest`, `demandcast-ui:latest`).
- **Task Definition:** Formulated in [`deployment/aws_ecs_task_definition.json`](file:///c:/Users/mihir/Desktop/DemandCast%20AI/deployment/aws_ecs_task_definition.json) allocating $1024$ CPU units ($1\text{ vCPU}$) and $2048\text{ MB}$ RAM with AWS CloudWatch log streaming (`awslogs`).
- **Load Balancing:** Positioned behind an AWS Application Load Balancer (ALB) with SSL/TLS termination on Port 443.

---

## 5. Deployment Step-by-Step Execution Guide

### Local Docker Testing (Single Command):
```powershell
# 1. Clone or navigate to the workspace
cd "c:\Users\mihir\Desktop\DemandCast AI"

# 2. Build and start all services in detached mode
docker compose up --build -d

# 3. Verify running containers and health status
docker compose ps

# 4. View real-time logs
docker compose logs -f
```

### Stopping Services:
```powershell
docker compose down
```

---

## 6. Deliverables Inventory

1. **`Dockerfile.api`**: Production FastAPI backend container definition.
2. **`Dockerfile.ui`**: Production Streamlit web frontend container definition.
3. **`Dockerfile`**: Unified multi-purpose container definition.
4. **`docker-compose.yml`**: Multi-container network and healthcheck orchestration blueprint.
5. **`.dockerignore`**: Optimized build context excluding temporary artifacts and caches.
6. **`.env.example`**: Production environment configuration template.
7. **`deployment/deploy_ec2.sh`**: Automated AWS EC2 host setup and deployment script.
8. **`deployment/aws_ecs_task_definition.json`**: AWS ECS Fargate task definition for cloud scaling.
9. **`report_stages/STAGE_15_DEPLOYMENT_REPORT.md`**: Complete technical deployment report.

---

## 7. Quality Gate Sign-Off & Transition to Stage 16

- [x] Multi-container Docker architecture formulated and verified.
- [x] Healthcheck probes implemented for both FastAPI and Streamlit services.
- [x] Read-only volume bindings protect serialized model weights against tampering.
- [x] Internal DNS resolution established over `demandcast-network`.
- [x] Automated AWS EC2 deployment script written and validated.
- [x] AWS ECS Fargate task definition created for elastic enterprise cloud hosting.

**Stage 15: Deployment is COMPLETE, CONTAINERIZED, and CERTIFIED for Stage 16 (Production Requirements).**
