# CI/CD Pipeline Documentation

## Overview

This repository uses GitHub Actions for automated testing, building, and deployment. The deployment pipeline uses a **self-hosted runner** installed on a VM to deploy the API to production.

## Workflows

### 1. CI Pipeline (`CI-action.yaml`)
Runs automated tests and validation on push/pull requests.

### 2. CD Pipeline - Training (`CD-train.yaml`)
Handles model training workflows.

### 3. CD Pipeline - Deployment (`CD-deploy.yaml`)
Automated deployment pipeline that:
1. **Tests** - Validates project structure
2. **Builds & Pushes** - Builds Docker image and pushes to Docker Hub
3. **Deploys to VM** - Deploys the API to production using self-hosted runner

## Self-Hosted Runner Architecture

### How it Works

```
GitHub Actions (Cloud) ←------ Self-Hosted Runner (VM)
       |                              |
       |                              |
   Workflow                      Executes commands
   triggered                     directly on VM
       |                              |
       └──────► Job sent ─────────────┘
                                      |
                                   Docker
                                 (pulls & runs container)
```

### Key Points

- **No SSH required**: The runner on the VM actively polls GitHub for jobs
- **Private network access**: Works even if VM is on a private network
- **Direct execution**: Commands run locally on the VM as if executed manually
- **Auto-start**: The runner service starts automatically on VM boot

## Deployment Flow

When code is pushed to the repository:

1. **GitHub Actions** detects the push and triggers the workflow
2. **Test job** runs on GitHub-hosted runner (ubuntu-latest)
3. **Deploy job** builds Docker image and pushes to Docker Hub
4. **Deploy-to-VM job** is assigned to the self-hosted runner:
   - Runner on VM picks up the job
   - Logs into Docker Hub
   - Pulls latest image
   - Stops old container
   - Starts new container on port 8000
   - Cleans up old images

## Self-Hosted Runner Setup

The runner is installed on the VM at: `~/actions-runner`

### Service Management

```bash
# Check status
sudo ./svc.sh status

# Start runner
sudo ./svc.sh start

# Stop runner
sudo ./svc.sh stop

# Restart runner
sudo ./svc.sh restart
```

### Auto-start Configuration

The runner is configured as a systemd service and starts automatically when the VM boots.

Service name: `actions.runner.mlops-2526q1-mds-upc-MLOps_XOXO.mlops-vm-runner.service`

## Troubleshooting

### Runner not picking up jobs
```bash
cd ~/actions-runner
sudo ./svc.sh status
sudo ./svc.sh restart
```

### Disk space issues
```bash
# Clean Docker resources
docker system prune -a -f

# Check disk space
df -h
```

### View runner logs
```bash
sudo journalctl -u actions.runner.* -f
```

## Secrets Configuration

Required secrets in GitHub repository settings:
- `DOCKER_USERNAME` - Docker Hub username
- `DOCKER_PASSWORD` - Docker Hub password

## Container Information

- **Container name**: `xoface-api`
- **Port**: 8000
- **Image**: `<DOCKER_USERNAME>/xoface-api:latest`
- **Restart policy**: unless-stopped
