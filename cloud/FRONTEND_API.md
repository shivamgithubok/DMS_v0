# VMS guardian API

HTTP and WebSocket contract for the phone app. The vehicle talks MQTT to this service. The app talks only to this API.

**Base URL:** `https://<render-host>` (local: `http://127.0.0.1:8000`)

Set `EXPO_PUBLIC_TELEMETRY_URL` to that origin. Leave `EXPO_PUBLIC_MQTT_URL` empty.

OpenAPI UI (same host): `/docs`

There is no auth header today. CORS allows any origin.

Default ride id while a real trip id is not wired: `ride-1842`.

Times named `at` and `lastNotifiedAt` are Unix epoch **milliseconds**. `updatedAt` is an ISO-8601 string.

---

## Types

### Ride

```json
{
  "id": "ride-1842",
  "deviceId": "orin-nano-01",
  "passenger": "Meera Shah",
  "safetyStatus": "safe",
  "driverState": "attentive",
  "deviceState": "online",
  "etaMin": 12,
  "speedKmh": 34.5,
  "heading": 90,
  "score": 98,
  "shareLink": "vms://trip/ride-1842",
  "position": { "latitude": 12.9716, "longitude": 77.5946 },
  "seats": [
    { "id": "driver", "label": "Driver", "occupied": true },
    { "id": "front", "label": "Front", "occupied": false },
    { "id": "rear-left", "label": "Rear Left", "occupied": true },
    { "id": "rear-right", "label": "Rear Right", "occupied": false }
  ],
  "updatedAt": "2026-10-27T10:15:00+00:00"
}
```

| Field | Values |
|---|---|
| `safetyStatus` | `safe` (default), `caution` (high or critical alert), `emergency` (SOS) |
| `driverState` | `attentive` (default), `drowsy`, `distracted`, `unavailable` (no face) |
| `deviceState` | vehicle status string, often `online` or `offline` |
| `position.latitude` / `longitude` | number or `null` |
| `etaMin`, `speedKmh`, `heading` | number or `null` |
| `score` | integer, starts at 98. SOS drops it by 20 (floor 8). High/critical alerts drop it by 8 (floor 46). |

Seat `id`: `driver` \| `front` \| `rear-left` \| `rear-right`.

### Alert

```json
{
  "id": "sos-a1b2c3d4e5f6",
  "tripId": "ride-1842",
  "kind": "sos",
  "severity": "critical",
  "title": "SOS raised",
  "detail": "Meera Shah raised SOS.",
  "at": 1761560000000,
  "locationLabel": "",
  "coordinate": { "latitude": 12.9716, "longitude": 77.5946 },
  "acknowledged": false,
  "evidenceId": null,
  "origin": "device"
}
```

| Field | Values |
|---|---|
| `kind` | `distraction`, `drowsiness`, `sos`, `seat_occupancy`, `suspicious_activity`, `harsh_braking`, `route_deviation` |
| `orinType` | vehicle type string (`DISTRACTION`, `CHILD_PRESENT`, …) or `null` |
| `channel` | `dms` (driver / audio) or `cms` (cabin / VLM cabin hazards) |
| `confidence` | number or `null` |
| `severity` | `info`, `low`, `medium`, `high`, `critical`. SOS is always `critical`. Seat occupancy is `info`. |
| `origin` | `device` (phone SOS) or `vehicle` (cabin / driver alerts) |
| `evidenceId` | string or `null` |

Default titles: Driver distraction, Driver drowsiness, SOS raised, Seat occupancy, Suspicious activity, Harsh braking, Route deviation.

### Guardian

```json
{
  "id": "g-1a2b3c4d",
  "name": "Anita Shah",
  "relationship": "mother",
  "phone": "+9198...",
  "sharing": true,
  "lastNotifiedAt": 1761560000000
}
```

`lastNotifiedAt` is `null` until that guardian is included in an SOS.

### Notice

```json
{
  "id": "notice-abc123",
  "at": 1761560000000,
  "title": "SOS sent",
  "body": "Notified Anita Shah.",
  "guardianNames": ["Anita Shah"],
  "origin": "device"
}
```

---

## REST

Errors are FastAPI shape: `{ "detail": "ride not found" }` with status `404`.

### `GET /health`

Liveness. Use this to confirm the API is up and the broker link is live.

```json
{
  "ok": true,
  "mqtt": { "connected": true }
}
```

The broker host and the last connection error are not in this response.

When the API is deployed with `GUARDIAN_AUTH_REQUIRED=1`, every `/v1` route except `POST /v1/session` and the vehicle `/v1/devices/...` uploads requires `Authorization: Bearer <token>`. The live socket takes the same token as `?token=`. Mint it with `POST /v1/session` and the guardian API key. A local desk leaves `GUARDIAN_AUTH_REQUIRED` unset, and these routes stay open.

### `GET /v1/rides/{rideId}`

Current trip snapshot. `404` if the ride does not exist.

Response: **Ride**.

### `GET /v1/rides/{rideId}/analysis`

Latest VLM scene slices. Not an alert. `404` if the ride does not exist.

`?channel=dms` or `?channel=cms` returns only that side (`dms` / `cms` is `null` until the first MQTT analysis message). Unknown `channel` is `400`.

```json
{
  "deviceId": "orin-nano-01",
  "updatedAt": 1761560000000,
  "dms": {
    "scene": "Driver facing forward, both hands on the wheel.",
    "hazards": [],
    "confidence": 0.82,
    "suppressed": ["driver distracted: detector owns driver state"],
    "observations": {
      "handsOnWheel": true,
      "gaze": "forward",
      "viewBlocked": false,
      "lighting": "ok",
      "cameraBlocked": false
    },
    "at": 1761560000000
  },
  "cms": {
    "scene": "Adult in front; child in rear-left with adult.",
    "hazards": [],
    "confidence": 0.74,
    "suppressed": [],
    "observations": {
      "occupants": { "adults": 2, "children": 1 },
      "childUnattended": false,
      "aggression": "calm",
      "viewBlocked": false,
      "weaponOrSmoke": false
    },
    "at": 1761560000000
  }
}
```

MQTT in: `dms/{deviceId}/analysis` and `cms/{deviceId}/analysis` (retained). Live socket frame: `{ "type": "analysis", "channel": "dms"|"cms", "analysis": { /* side object */ } }`.

### `GET /v1/rides/{rideId}/alerts?status=all`

Newest first.

| `status` | Filter |
|---|---|
| `all` (default) | every alert |
| `open` | `acknowledged === false` |
| `emergency` | `severity` is `high` or `critical` |
| `caution` | `severity` is `info`, `low`, or `medium` |

```json
{ "alerts": [ /* Alert */ ] }
```

Unknown `status` values return the full list (same as `all`). `?channel=dms` or `?channel=cms` keeps only that channel.

### `POST /v1/alerts/{alertId}/ack`

Marks one alert acknowledged. No body. `404` if missing.

Response: **Alert** with `acknowledged: true`.

### `POST /v1/rides/{rideId}/sos`

Passenger SOS from the phone. Creates a critical alert, sets `safetyStatus` to `emergency`, notifies guardians with `sharing: true`, writes a notice, and pushes a `sos` event on the live socket.

If the ride id is unknown, the server attaches the SOS to the default ride.

```json
{
  "passenger": "Meera Shah",
  "latitude": 12.9716,
  "longitude": 77.5946,
  "source": "phone"
}
```

All fields are optional. `passenger` defaults to `""`. Coordinates may be `null`.

```json
{
  "id": "sos-a1b2c3d4e5f6",
  "at": 1761560000000,
  "topic": "ride/ride-1842/sos",
  "origin": "device",
  "note": "Notified Anita Shah",
  "alert": { }
}
```

`note` is `"Notified nobody sharing"` when no guardian has sharing on. `alert` is an **Alert**.

### `POST /v1/rides/{rideId}/telemetry`

Phone location for the map. `404` if the ride does not exist. Pushes a `telemetry` event on the live socket.

JSON accepts camelCase or snake_case for the optional fields.

```json
{
  "latitude": 12.9716,
  "longitude": 77.5946,
  "heading": 90,
  "speedKmh": 34.5,
  "etaMin": 12
}
```

`latitude` and `longitude` are required. The others are optional and only overwrite the stored value when sent.

Response: **Ride**.

### `GET /v1/rides/{rideId}/cabin`

Driver and seat slice of the ride. `404` if missing.

```json
{
  "driverState": "attentive",
  "seats": [ { "id": "driver", "label": "Driver", "occupied": true } ],
  "safetyStatus": "safe"
}
```

### `GET /v1/rides/{rideId}/guardians`

```json
{ "guardians": [ /* Guardian */ ] }
```

Empty array when none are stored. Does not 404 for an unknown ride.

### `POST /v1/rides/{rideId}/guardians`

`404` if the ride does not exist.

```json
{
  "name": "Anita Shah",
  "relationship": "mother",
  "phone": "+9198...",
  "sharing": true
}
```

`name` is required. `relationship` and `phone` default to `""`. `sharing` defaults to `true`.

```json
{ "id": "g-1a2b3c4d" }
```

### `POST /v1/rides/{rideId}/guardians/{guardianId}/sharing`

Toggles `sharing`. No body. `404` if that guardian is not on this ride.

```json
{ "id": "g-1a2b3c4d", "sharing": false }
```

### `POST /v1/rides/{rideId}/share`

Refreshes the trip link and writes a notice titled `"Trip link refreshed"`. `404` if the ride does not exist. No body.

```json
{ "link": "vms://trip/ride-1842", "origin": "vehicle" }
```

### `GET /v1/rides/{rideId}/notices`

Newest first.

```json
{ "notices": [ /* Notice */ ] }
```

### `POST /v1/devices/{deviceId}/push-token`

`deviceId` must be an existing **guardian id** (`g-…`). `404` otherwise (`detail`: `"store the token against a guardian id for now"`).

```json
{ "token": "<fcm token>", "ride_id": "ride-1842" }
```

`ride_id` is accepted and ignored. The token is stored on the guardian row.

```json
{ "ok": true }
```

Push delivery is not implemented yet. This only stores the token.

### `GET /v1/rides/{rideId}/video?track=all`

Guardian live watch. **Not a call to the Jetson.** Returns a LiveKit (or dummy) viewer token if at least one guardian has `sharing: true`. `403` when sharing is off. `404` if the ride is missing.

`track`: `all` (default), `driver`, or `cabin`.

On the VPS, `LIVEKIT_URL` is `ws://<VPS>:7880` (or `wss://`). Then `mode` is `livekit` and `stream` is `webrtc`. The app connects to LiveKit and subscribes to the track named `cabin`. That track is the **driver camera**, encoded as video (640×360, 12 fps), plus the microphone on an audio track also named `cabin`. Do not play `preview` as the watch picture.

```json
{
  "url": "ws://VPS_IP:7880",
  "mode": "livekit",
  "stream": "webrtc",
  "token": "<jwt>",
  "room": "ride-1842",
  "tracks": ["cabin"],
  "video": { "track": "cabin", "camera": "driver", "width": 640, "height": 360, "fps": 12, "bitrate": 800000 },
  "audio": { "track": "cabin", "rate": 16000, "channels": 1 },
  "expiresAt": 1761560900000,
  "publisherOnline": true,
  "publish": true,
  "publishTo": "livekit",
  "preview": "/v1/rides/ride-1842/video/preview"
}
```

`preview` is the idle JPEG. `stream` is `jpeg` only when `LIVEKIT_URL` is `dummy://…` or `local://…`. Viewer JWT has `canPublish: false`.

`POST /v1/rides/{rideId}/video/cancel` clears the watch flag. The Jetson drops the LiveKit room on its next media-token poll.

### `GET /v1/rides/{rideId}/video/preview?track=driver`

Latest idle JPEG (`driver` or `cabin`). Same sharing gate as `/video`. `404` until the vehicle has posted a frame. This is not the live watch picture when `stream` is `webrtc`.

### `POST /v1/devices/{deviceId}/media-token`

Jetson only. Body `{ "key": "<MEDIA_DEVICE_KEY>" }`. Returns a **publisher** JWT (`canPublish: true`, `canSubscribe: false`) and marks the publisher online.

When a guardian is watching and `mode` is `livekit`, `publish` is true, `publishTo` is `livekit`, and `tracks` is `["cabin"]`. The Jetson then publishes the driver camera as that video track and does not upload JPEGs until the watch ends. Otherwise `publish` is false and `publishTo` is `POST /v1/devices/{deviceId}/video/frame`.

### `POST /v1/devices/{deviceId}/video/frame`

Idle JPEG still. Body `{ "key": "…", "track": "driver"|"cabin", "at": 1761560000000, "image_b64": "<jpeg>" }`. Sent while nobody is watching. `track` `driver` is stored as channel `dms`, `cabin` as `cms`. Guardian reads `/video/preview` and `GET /v1/rides/{rideId}/frames/live?channel=dms|cms`.

### `POST /v1/devices/{deviceId}/video/live`

Local-mode fallback only (`LIVEKIT_URL` starts with `local`). One JPEG (`image_b64`) plus PCM (`audio_b64`). Not used when `mode` is `livekit`. Phone read: `GET /v1/rides/{rideId}/video/live`.

### `GET /v1/history?deviceId=`

Past and current trips for that vehicle, newest `startedAt` first.

```json
{ "rides": [ { "id": "ride-1842", "deviceId": "orin-nano-01", "status": "active", "startedAt": 1761560000000, "endedAt": null, "safetyStatus": "safe", "score": 98, "alertCount": 1 } ] }
```

### `GET /v1/rides/{rideId}/frames/live?channel=dms|cms`

Latest still. `at` is the capture time in epoch ms. Image: `GET …/frames/live/image?channel=dms`. Sharing off → `403`.

### `POST /v1/devices/{deviceId}/clips`

Short danger clip (not the whole ride, and not the live watch stream). Body:

```json
{
  "key": "<MEDIA_DEVICE_KEY>",
  "channel": "dms",
  "alert_id": "<alert id>",
  "start_at": 1761561196000,
  "end_at": 1761561204000,
  "frames": [ { "at": 1761561196000, "image_b64": "<jpeg>" } ],
  "video_b64": "<mp4>"
}
```

`frames` is up to 24 JPEGs from about 4 seconds before the alert to 4 seconds after. `video_b64` is one H.264 MP4 of those same frames. At least one of `frames` or `video_b64` is required. Links `clipId` onto that alert when `alert_id` matches.

### `GET /v1/rides/{rideId}/clips/{clipId}`

```json
{ "id": "clip-abc", "channel": "dms", "alertId": "…", "startAt": 1, "endAt": 2, "frames": [ { "at": 1, "url": "/v1/media/clip-abc/0.jpg" } ], "videoUrl": "/v1/media/clip-abc/clip.mp4" }
```

`GET /v1/media/{clipId}/{file}` returns the JPEG or `video/mp4` for `clip.mp4`. Sharing off → `403`.

---

## WebSocket

`GET ws(s)://<host>/v1/rides/{rideId}/live`

Connect and keep the socket open. The server accepts the connection and then ignores any text the client sends. Close the socket to leave the room.

Server frames are JSON text:

| `type` | When | Payload |
|---|---|---|
| `telemetry` | `POST .../telemetry` | `{ "type": "telemetry", "ride": Ride }` |
| `status` | vehicle heartbeat / cabin status | `{ "type": "status", "ride": Ride }` |
| `alert` | vehicle alert | `{ "type": "alert", "alert": Alert, "ride": Ride }` |
| `analysis` | latest VLM slice | `{ "type": "analysis", "channel": "dms"\|"cms", "analysis": object }` |
| `sos` | phone SOS, or a vehicle alert whose kind is `sos` | `{ "type": "sos", "alert": Alert, "ride": Ride, "notified"?: string[] }` |

`notified` is present only for phone SOS. It is the list of guardian names who had sharing on. A vehicle SOS is sent twice: once as `alert`, then again as `sos` (without `notified`).

Replace the ride object in local state on `telemetry` and `status`. Append the alert on `alert` and `sos`, and set safety from `ride.safetyStatus`.

Example:

```js
const ws = new WebSocket(
  `${base.replace(/^http/, "ws")}/v1/rides/${rideId}/live`
);
ws.onmessage = (ev) => {
  const msg = JSON.parse(ev.data);
  if (msg.type === "telemetry" || msg.type === "status") {
    setRide(msg.ride);
  } else if (msg.type === "alert" || msg.type === "sos") {
    setRide(msg.ride);
    prependAlert(msg.alert);
  }
};
```

---

## Suggested screens

| Screen | Calls |
|---|---|
| Trip home | `GET /v1/rides/{id}` once, then the live socket |
| Map | `POST /v1/rides/{id}/telemetry` on location updates |
| Alerts | `GET /v1/rides/{id}/alerts`, then socket `alert` / `sos`. Ack with `POST /v1/alerts/{id}/ack` |
| SOS button | `POST /v1/rides/{id}/sos` |
| Cabin | `GET /v1/rides/{id}/cabin` or `ride.seats` + `ride.driverState` from the socket |
| VLM scene | `GET /v1/rides/{id}/analysis` (`?channel=dms` or `cms`), then socket `analysis` |
| Driver vs cabin alerts | `GET /v1/rides/{id}/alerts?channel=dms` or `cms` |
| Guardians | list, `POST` to add, `POST .../sharing` to toggle |
| Share | `POST /v1/rides/{id}/share`, then `GET .../notices` |
| Push | `POST /v1/devices/{guardianId}/push-token` after the guardian exists |
| Live video | `GET /v1/rides/{id}/video`. If `stream` is `webrtc`, play LiveKit track `cabin` (driver camera). `preview` is the idle JPEG only. 403 if sharing is off |
| Alert clip | `GET /v1/rides/{id}/clips/{clipId}` then `frames[].url` (JPEG) and `videoUrl` (MP4) |

---

## Smoke test

```bash
curl -s https://<host>/health

curl -s -X POST https://<host>/v1/rides/ride-1842/sos \
  -H 'content-type: application/json' \
  -d '{"passenger":"Meera Shah","latitude":12.9716,"longitude":77.5946,"source":"phone"}'

curl -s https://<host>/v1/rides/ride-1842
curl -s https://<host>/v1/rides/ride-1842/alerts
```
