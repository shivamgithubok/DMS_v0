# VMS guardian API

HTTP + WebSocket for the Expo guardian app. Subscribes to the Orin MQTT topics on a VPS broker. Deploy this folder to Render. Do not deploy `orin-dms` itself.

```
Orin  --MQTT :1883-->  VPS EMQX
                         ^
                         | subscribe
                      Render API  --HTTPS-->  phone
                      Render Postgres
```

## Online services

| Service | What | Cost to start |
|---|---|---|
| [Oracle Cloud](https://www.oracle.com/cloud/free/) or [Hetzner](https://www.hetzner.com/cloud) or [DigitalOcean](https://www.digitalocean.com/products/droplets) | VPS for EMQX | Oracle always-free ARM, or ~$4–6/mo |
| [Render](https://render.com) | Web Service (this API) + Postgres | Free web sleeps; free Postgres expires — upgrade later |
| [GitHub](https://github.com) | Push this repo so Render can build `cloud/` | Free |
| Firebase Cloud Messaging | Later, for SOS when the app is closed | Free |

Do not put the broker on Render. Do not put this API on the Jetson.

## 1. VPS broker

1. Create a VM (Ubuntu 22.04, 1 vCPU, 1 GB). Note the public IP. Call it `VPS_IP`.
2. Open inbound TCP **22**, **1883**, and **18083** in the cloud firewall. 18083 is the EMQX dashboard; lock it to your IP after setup.
3. SSH in and install Docker:

```bash
sudo apt update
sudo apt install -y docker.io docker-compose-v2
sudo usermod -aG docker $USER
```

4. Copy `deploy/emqx/` from this repo onto the VM (scp, or git clone the whole repo and `cd deploy/emqx`).
5. Start the broker:

```bash
cd deploy/emqx
cp .env.example .env
# edit .env: EMQX_DASHBOARD_PASSWORD=...
docker compose up -d
```

6. Open `http://VPS_IP:18083` → `admin` / password from `.env`.
7. Access Control → Authentication → add two users (password backend):
   - `dms-orin` — Jetson publisher
   - `dms-cloud` — Render subscriber
8. Prove it from your laptop: install MQTTX, connect to `VPS_IP:1883` as `dms-cloud`, subscribe `dms/#` and `cms/#`.

## 2. Point the Jetson at the VPS

On the Orin, do **not** clone a new repo. Change config only.

`config/dms.yaml`:

```yaml
mqtt:
  host: VPS_IP
  port: 1883
```

`/etc/default/dms` (see `deploy/dms.cloud.env.example`):

```bash
DMS_DEVICE_ID=orin-nano-01
DMS_MQTT_USERNAME=dms-orin
DMS_MQTT_PASSWORD=the-password-from-emqx
```

Restart DMS. MQTTX should show `dms/orin-nano-01/status` then alerts. If that works, the vehicle side is done.

## 3. GitHub + Render API + Postgres

1. Push this git repo to GitHub (or create a repo that contains the `cloud/` folder).
2. [Render Dashboard](https://dashboard.render.com) → **New** → **Blueprint**.
3. Connect the GitHub repo.
4. Set **Root Directory** to `cloud`.
5. Apply `render.yaml`. That creates:
   - Postgres **vms-db**
   - Web service **vms-api**
6. In **vms-api** → Environment, set the values Blueprint left blank:
   - `MQTT_HOST` = `VPS_IP`
   - `MQTT_USERNAME` = `dms-cloud`
   - `MQTT_PASSWORD` = the EMQX password
7. Wait for deploy. Open:

```
https://vms-api-XXXX.onrender.com/health
```

You want `"mqtt": {"connected": true}`. If connected is false, the VPS firewall is blocking Render → 1883, or the password is wrong.

8. Copy the Render URL. That is the only URL the phone needs.

Free Render web services sleep after ~15 minutes. The first SOS after sleep can take ~30–50 s. A paid Starter web service stays up. The broker on the VPS must never sleep.

## 4. Phone (VMS Expo repo)

In `Driver-Monitoring-System-` `.env`:

```bash
EXPO_PUBLIC_TELEMETRY_URL=https://vms-api-XXXX.onrender.com
EXPO_PUBLIC_MQTT_URL=
```

Wire SOS with `cloud/phone-createServices.snippet.ts`. Restart Expo.

Test:

```bash
curl -X POST https://vms-api-XXXX.onrender.com/v1/rides/ride-1842/sos \
  -H 'content-type: application/json' \
  -d '{"passenger":"Meera Shah","latitude":12.9716,"longitude":77.5946}'
```

Then:

```
GET /v1/rides/ride-1842/alerts
```

## What goes where

| Platform | Push / run |
|---|---|
| VPS | `deploy/emqx` only (`docker compose up`). No Orin code. |
| Render | Root `cloud/`. Postgres created by the Blueprint. |
| Jetson | Existing `orin-dms`. Config + env only. |
| Phone | Existing Expo repo. `.env` + SOS fetch. |

## Local test (before any VPS / Render)

Three terminals from `cloud/`:

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt

# 1) MQTT broker
python scripts/run_local_broker.py

# 2) API
MQTT_HOST=127.0.0.1 DATABASE_URL=sqlite:///./data.db \
  uvicorn app.main:app --host 127.0.0.1 --port 8000

# 3) Fake Orin + phone SOS
./scripts/e2e_local.sh
```

Browser: http://127.0.0.1:8000/health then http://127.0.0.1:8000/docs

If Docker works on your machine you can use Mosquitto instead of terminal 1:

```bash
docker compose -f docker-compose.local.yml up -d
```
