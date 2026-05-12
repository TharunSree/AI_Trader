# Oracle Cloud Free Tier AI Trader Deployment

This document contains the exact steps to deploy the AI Trading platform on Oracle Cloud's Always Free Architecture.

## 1. Provisioning the Oracle Instance
1. Log into Oracle Cloud Console.
2. Navigate to **Menu > Compute > Instances** and click **Create Instance**.
3. **Image & Shape**: 
   - Image: Canonical Ubuntu 22.04 / 24.04
   - Shape: Switch to `VM.Standard.A1.Flex` (ARM Ampere A1).
   - Resources: Drag the OCPU slider to `4` and Memory to `24GB`.
4. **Networking**: Assign a Public IP address.
5. **Keys**: Download your SSH Key Pair (Required to access the server).
6. **Boot Volume**: Increase size to `50 GB` (Always Free limit).

> [!TIP]
> Capacity for ARM compute is extremely scarce. The easiest regions to successfully provision right now are **ca-montreal-1**, **sa-saopaulo-1**, or **ap-hyderabad-1**. If you get "Out of Capacity", you must click the retry button consistently or use an auto-clicker.

## 2. Server Preparation

SSH into your new instance:
```bash
ssh -i your_key.pem ubuntu@<your_public_ip>
```

Install System Dependencies:
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip python3-venv git htop ufw -y
```

Open Firewall Ports (for Dashboard HTTP/WebSocket access):
```bash
sudo iptables -I INPUT 6 -p tcp --dport 8000 -j ACCEPT
sudo netfilter-persistent save
```

## 3. Cloning and Setup

Clone the project to the server:
```bash
git clone <your_github_repo_url> AI_Trader
cd AI_Trader
```

Construct the Virtual Environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Deploy `.env` configuration template. You **must** populate this via `nano .env`:
```text
BASE_URL="https://paper-api.alpaca.markets"
API_KEY="your-alapca-key"
SECRET_KEY="your-alpaca-secret"
GEMINI_API_KEY="your-gemini-key"
EMAIL_HOST_USER="tarunsree@gmail.com"
```

## 4. Booting the Trade engine as a Persistent Service

Create a `systemd` configuration so your trader stays alive automatically 24/7 if the server reboots:

```bash
sudo nano /etc/systemd/system/jarvis_brain.service
```

Paste the following:
```ini
[Unit]
Description=Jarvis AI Trader Core
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/AI_Trader
ExecStart=/home/ubuntu/AI_Trader/.venv/bin/python manage.py runserver 0.0.0.0:8000
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

Start the engines:
```bash
sudo systemctl daemon-reload
sudo systemctl enable jarvis_brain
sudo systemctl start jarvis_brain
```

## Verification
You can now access your hardened AI Dashboard globally via `http://<your_oracle_ip>:8000`. The engine will autonomously trade using the memory footprint, protected by the neural rewriter.
