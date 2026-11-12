# Guardian app API

Contract for the phone app. The app talks only to this HTTP API and, for live video, to the LiveKit URL returned by this API. Do not connect the app to MQTT, the Jetson, or port `7880` with a secret of your own.

Interactive list of the same routes: `GET {BASE}/docs`

## Setup

```bash
EXPO_PUBLIC_TELEMETRY_URL=http://YOUR_VPS_IP:8000
EXPO_PUBLIC_MQTT_URL=
```

No trailing slash on the base URL. Leave `EXPO_PUBLIC_MQTT_URL` empty.

| Value | Current production |
|---|---|
| Base URL | `http://YOUR_VPS_IP:8000` |
| Ride id | `ride-1842` |
| Device id | `orin-nano-01` |

There is no auth header. CORS allows any origin. Unknown rides return `404` with `{ "detail": "ride not found" }`. Sharing off returns `403` with `{ "detail": "sharing is off" }` on video, frames, clips, recordings, and snaps.

Times named `at`, `startedAt`, `endedAt`, `expiresAt`, and `lastNotifiedAt` are Unix epoch **milliseconds**. `updatedAt` is an ISO-8601 string.

Do not put these in the app: `MEDIA_DEVICE_KEY`, `LIVEKIT_API_SECRET`, MQTT password, EMQX dashboard password.

Share this file. The new calls are Speak (`POST /video/talk`), guardian recordings and snaps (lists filtered by `guardianId`), alerts for one trip (`GET /alerts?startedAt=`), and push after `POST /push-token`. Alert clip id is still `alerts[].clipId`. Live video is `GET /video`, then two LiveKit tracks: `driver` and `cabin`.

---

## What the app calls

| Screen | Call |
|---|---|
| Trip home | `GET /v1/rides/{rideId}` once, then WebSocket `/v1/rides/{rideId}/live` |
| Device online | `ride.deviceState` and `ride.publisherOnline` from the ride or the `status` socket event |
| Map | `POST /v1/rides/{rideId}/telemetry` |
| Alerts | `GET /v1/rides/{rideId}/alerts`, then socket `alert` / `sos`. Ack with `POST /v1/alerts/{alertId}/ack`. A past trip uses `?startedAt=` |
| Speak | `POST /v1/rides/{rideId}/video/talk` publishes the phone mic as `guardian-mic`. `POST .../video/talk/cancel` stops it |
| VLM text | `GET /v1/rides/{rideId}/analysis?channel=dms` or `cms`, then socket `analysis` |
| Live still | `GET /v1/rides/{rideId}/frames/live?channel=dms` or `cms`, then load `url` |
| Live video | `GET /v1/rides/{rideId}/video` starts it. `POST /v1/rides/{rideId}/video/cancel` stops it |
| Past trips | `GET /v1/history?deviceId=orin-nano-01`. Several rows share `id: ride-1842`. Switch trips with `startedAt`. |
| Past trip map | `GET /v1/rides/ride-1842?startedAt={startedAt}` and draw `route` |
| Alert clip | `alerts[].clipId` from `GET /v1/rides/{rideId}/alerts`, then `GET /v1/rides/{rideId}/clips/{clipId}` |
| Record | `POST /v1/rides/{rideId}/recordings` with `{"action":"start","guardianId":"..."}`. Stop with `POST .../recordings/{id}/cancel`. List with `GET .../recordings?guardianId=` |
| Snap | `POST /v1/rides/{rideId}/snapshots`. List with `GET .../snapshots?guardianId=` |
| SOS | `POST /v1/rides/{rideId}/sos` |
| Cabin seats | `GET /v1/rides/{rideId}/cabin` |
| Guardians | list, `POST` to add, `POST .../sharing` to toggle |
| Share link | `POST /v1/rides/{rideId}/share` |
| Notices | `GET /v1/rides/{rideId}/notices` |
| Push token | `POST /v1/devices/{guardianId}/push-token` |

There is no `PATCH` or `DELETE` for guardians, rides, or alerts.

---

## Ride

`GET /v1/rides/{rideId}`

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
  "updatedAt": "2026-10-28T10:15:00+00:00",
  "publisherOnline": true,
  "status": "active",
  "startedAt": 1761560000000,
  "endedAt": null,
  "route": [
    { "latitude": 12.9716, "longitude": 77.5946 },
    { "latitude": 12.9800, "longitude": 77.6100 }
  ]
}
```

| Field | Values |
|---|---|
| `safetyStatus` | `safe`, `caution`, `emergency` |
| `driverState` | `attentive`, `drowsy`, `distracted`, `unavailable` |
| `deviceState` | `online` or `offline`. This is the vehicle heartbeat. |
| `publisherOnline` | `true` when a video frame arrived in the last minute |
| `status` | `active` or `ended` |
| `startedAt` / `endedAt` | epoch ms, or `null`. On a history open, these are the selected trip, not a new ride id. |
| `route` | Ordered GPS points for the map. Empty array when that trip has no fix. One point is a pin, not a line. At most 200 points. |
| `position` | Last known point. Use it for the car marker. On a history open this is the trip's end pin. |
| `score` | integer. Starts at 98. SOS drops it by 20 (floor 8). High or critical alerts drop it by 8 (floor 46). |

Seat `id`: `driver`, `front`, `rear-left`, `rear-right`.

`GET /v1/rides/{rideId}/cabin` returns only:

```json
{ "driverState": "attentive", "seats": [], "safetyStatus": "safe" }
```

---

## History

The phone keeps one ride id, `ride-1842`. Each time the vehicle runs, the API stores another trip under that same id. The trip key is `startedAt` (epoch ms). Do not mint a new ride id per trip, and do not key the list on `id` alone: every row has `id: "ride-1842"`.

The list still loads when the Jetson is offline. `deviceState` on the live ride tells you the car is offline. The saved trips stay.

### List

`GET /v1/history?deviceId=orin-nano-01`

Newest `startedAt` first. Omit `deviceId` to list every device.

```json
{
  "rides": [
    {
      "id": "ride-1842",
      "deviceId": "orin-nano-01",
      "passenger": "Meera Shah",
      "status": "ended",
      "startedAt": 1761560000000,
      "endedAt": 1761563600000,
      "safetyStatus": "caution",
      "score": 82,
      "alertCount": 2,
      "start": { "latitude": 12.9716, "longitude": 77.5946 },
      "end": { "latitude": 12.9800, "longitude": 77.6100 }
    },
    {
      "id": "ride-1842",
      "deviceId": "orin-nano-01",
      "passenger": "Meera Shah",
      "status": "active",
      "startedAt": 1761567200000,
      "endedAt": null,
      "safetyStatus": "safe",
      "score": 98,
      "alertCount": 0,
      "start": { "latitude": 12.9800, "longitude": 77.6100 },
      "end": { "latitude": 12.9850, "longitude": 77.6200 }
    }
  ]
}
```

| Field | Use on the history row |
|---|---|
| `startedAt` | Identity of this trip. Pass it when the user opens the row. |
| `endedAt` | `null` while that trip is still the open one. |
| `status` | `active` or `ended` |
| `start` / `end` | Pins for the row. Either can be `null` when GPS was missing. |
| `alertCount` | Number on the row. |

### Map for the trip the user opened

`GET /v1/rides/ride-1842?startedAt=1761560000000`

Same Ride object as the live screen, plus the selected trip's `startedAt`, `endedAt`, `status`, `score`, `safetyStatus`, and `route`.

```json
{
  "id": "ride-1842",
  "startedAt": 1761560000000,
  "endedAt": 1761563600000,
  "status": "ended",
  "position": { "latitude": 12.9800, "longitude": 77.6100 },
  "route": [
    { "latitude": 12.9716, "longitude": 77.5946 },
    { "latitude": 12.9750, "longitude": 77.6020 },
    { "latitude": 12.9800, "longitude": 77.6100 }
  ]
}
```

Draw `route` in order as the polyline. Put the car marker on `position` (the last point). `start` from the history row is the other pin.

| `route` | What to draw |
|---|---|
| `[]` | No GPS for that trip. Show the trip card without a line. |
| one point | A single marker. Do not draw a line. |
| two or more | Polyline from first to last, marker on the last point. |

An unknown `startedAt` still returns `200` and `route: []`. It does not 404.

Leave `startedAt` off (`GET /v1/rides/ride-1842`) for the live home screen. That returns the newest trip that already has a GPS point, which may be the open trip.

`GET /v1/rides/{id}/alerts?startedAt={ms}` returns only alerts from that trip. An unknown `startedAt` returns `200` and `alerts: []`. Leave `startedAt` off for every alert on the ride. Drop a map pin where `coordinate.latitude` and `coordinate.longitude` are not null.

---

## Alerts

`GET /v1/rides/{rideId}/alerts?status=all&channel=dms&startedAt={ms}`

`startedAt` is optional. When it is set, the list is that trip only (`at` from `startedAt` through `endedAt`, or still open if the trip has not ended).

`status`: `all` (default), `open`, `emergency` (`high` or `critical`), `caution` (`info`, `low`, `medium`). Unknown `status` returns every alert.

`channel`: `dms` (driver) or `cms` (cabin). Omit it for both.

```json
{
  "alerts": [
    {
      "id": "alert-1",
      "tripId": "ride-1842",
      "kind": "distraction",
      "orinType": "DISTRACTION",
      "channel": "dms",
      "severity": "high",
      "title": "Driver distraction",
      "detail": "head turned",
      "at": 1761560000000,
      "locationLabel": "head_turned",
      "coordinate": { "latitude": null, "longitude": null },
      "acknowledged": false,
      "evidenceId": null,
      "origin": "vehicle",
      "confidence": 0.9,
      "clipId": "clip-abc123"
    }
  ]
}
```

| Field | Values |
|---|---|
| `kind` | `distraction`, `drowsiness`, `sos`, `seat_occupancy`, `suspicious_activity`, `harsh_braking`, `route_deviation` |
| `channel` | `dms` or `cms` |
| `severity` | `info`, `low`, `medium`, `high`, `critical` |
| `origin` | `vehicle` or `device` (phone SOS) |
| `clipId` | id of the alert clip, or `null`. This is the only place the app reads the clip id |

`POST /v1/alerts/{alertId}/ack` with no body. Returns the same alert with `acknowledged: true`. `404` if missing.

---

## VLM analysis

Not an alert. Latest description only. DMS is the driver scene. CMS is the cabin scene.

`GET /v1/rides/{rideId}/analysis`

`GET /v1/rides/{rideId}/analysis?channel=dms`

`GET /v1/rides/{rideId}/analysis?channel=cms`

```json
{
  "deviceId": "orin-nano-01",
  "updatedAt": 1761560000000,
  "dms": {
    "scene": "Driver facing forward, both hands on the wheel.",
    "hazards": [],
    "confidence": 0.82,
    "suppressed": [],
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
    "scene": "Adult in front. Child in rear-left.",
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

A side that has not arrived yet is `null`. Both sides stay `null` while the car is running with the VLM off. `aggression` is `calm`, `argument`, or `panic`. Show `dms` on the driver page and `cms` on the cabin page. Driver alerts still come from the detector (`kind` `distraction` / `drowsiness`), not from `dms.hazards`.

---

## Live stills

One current JPEG per channel, with the capture time.

`GET /v1/rides/{rideId}/frames/live?channel=dms`

`GET /v1/rides/{rideId}/frames/live?channel=cms`

```json
{
  "rideId": "ride-1842",
  "channel": "dms",
  "at": 1761561800500,
  "url": "/v1/rides/ride-1842/frames/live/image?channel=dms"
}
```

Load the picture from `{BASE}` + `url`. That URL is `GET /v1/rides/{rideId}/frames/live/image?channel=dms` (or `cms`) and the body is the JPEG, not JSON. `404` until the first frame. `403` when sharing is off. If `at` is more than about 15 seconds old, show the still as stale.

`dms` is the driver camera. `cms` is the cabin camera. This is the idle still on the trip page. It is not the Watch stream and not the alert clip. While Watch is publishing, the car stops these JPEG uploads, so `at` goes stale on purpose. Keep showing the LiveKit tracks during Watch. After `POST /video/cancel`, the stills start again.

---

## Live video

Watch is two pictures plus the cabin microphone. The app does not call the car.

1. `GET /video` tells the car to start. Until this call, the car stays on JPEG stills and does not join the room.
2. Connect a LiveKit client to `url` with `token`.
3. Subscribe to the video tracks named `driver` and `cabin`, and the audio track named `cabin`.
4. `POST /video/cancel` tells the car to leave the room. The stream stays up until that cancel.

`GET /v1/rides/{rideId}/video?track=all`

`track` may be `all` (default), `driver`, or `cabin`. Use `all` for Watch.

```json
{
  "url": "ws://YOUR_VPS_IP:7880",
  "mode": "livekit",
  "stream": "webrtc",
  "token": "<viewer jwt>",
  "room": "ride-1842",
  "tracks": ["driver", "cabin"],
  "expiresAt": 1761560900000,
  "publisherOnline": true,
  "publish": true,
  "preview": "/v1/rides/ride-1842/video/preview"
}
```

Use `url` exactly as returned. It may be `ws://`, not `wss://`.

| Track name | What the car sends |
|---|---|
| `driver` | Driver camera. On the current Orin this is the USB webcam. |
| `cabin` | Cabin camera. On the current Orin this is the cabin video file until a second camera is plugged in. |
| `cabin` (audio) | Cabin microphone. Same name as the cabin video track. Subscribe to the audio track, not a second video. |

| Field | Meaning |
|---|---|
| `publish` | `true` after this call. The car joins the room and sends both video tracks until cancel. |
| `stream` | `webrtc` while Watch is on. `jpeg` when nobody is watching. |
| `token` | Viewer token. It cannot publish. Refresh by calling `GET /video` again before `expiresAt`. That call does not stop the stream. |
| `publisherOnline` | `true` when a JPEG arrived in the last minute. During Watch the car pauses JPEGs, so this flag can go `false` while LiveKit is still playing. Keep the player up from the room connection. Use this flag on the trip page, before Watch. |

| `mode` | What to do |
|---|---|
| `livekit` | Connect to `url` with `token`. Subscribe to video `driver`, video `cabin`, and audio `cabin`. |
| `local` | No LiveKit server. `GET /v1/rides/{rideId}/video/live` returns the latest sample as JSON (`width`, `height`, `samples`, `rate`, `image_b64`, `audio_b64`, `seq`). This is for a local test, not the production player. |
| `dummy` | No moving stream. `publish` is `false`. Show the JPEG from `{BASE}{preview}?track=driver` or `track=cabin`. |

`403` when sharing is off.

`POST /v1/rides/{rideId}/video/cancel` has no body.

```json
{ "ok": true, "publish": false, "room": "ride-1842" }
```

`404` if the ride does not exist. The next time the car asks for a token it leaves the room and JPEG stills resume.

## Speak

This is the phone microphone played on the cabin speaker. It is separate from Watch. Watch sends the cabin microphone to the phone.

`POST /v1/rides/{rideId}/video/talk` has no body. The app publishes one audio track named `guardian-mic` and does not subscribe.

```json
{
  "url": "ws://YOUR_VPS_IP:7880",
  "token": "<jwt canPublish microphone only>",
  "room": "ride-1842",
  "expiresAt": 1761560900000,
  "tracks": ["guardian-mic"]
}
```

`POST /v1/rides/{rideId}/video/talk/cancel` stops it. The car drops the speaker on its next poll.

```json
{ "ok": true, "talk": false, "room": "ride-1842" }
```

`403` when sharing is off. `409` when the server is not in LiveKit mode. Cancel does not stop Watch.

`GET /v1/rides/{rideId}/video/preview?track=driver` returns a JPEG body (`track` is `driver` or `cabin`). Show that JPEG before Watch connects and if the room fails. The trip-page still is `frames/live`. The moving stream is the two LiveKit tracks when `mode` is `livekit`.

---

## Alert clip

Only a few seconds around a danger alert are stored. The full trip is not a video.

When `alert.clipId` is set:

`GET /v1/rides/{rideId}/clips/{clipId}`

```json
{
  "id": "clip-abc123",
  "rideId": "ride-1842",
  "alertId": "alert-1",
  "channel": "dms",
  "startAt": 1761561196000,
  "endAt": 1761561204000,
  "frames": [
    { "at": 1761561196000, "url": "/v1/media/clip-abc123/0.jpg" },
    { "at": 1761561200000, "url": "/v1/media/clip-abc123/1.jpg" }
  ],
  "videoUrl": "/v1/media/clip-abc123/clip.mp4"
}
```

Play `{BASE}` + `videoUrl` (`GET /v1/media/{clipId}/clip.mp4`). The body is `video/mp4`, about 8 seconds: 4 seconds before the alert and 4 seconds after. That file is what the guardian watches for this alert on this trip. `frames` are the same moment as stills. `channel` `dms` is the driver camera. `cms` is the cabin camera. `videoUrl` is `null` only when the car could not encode the file. `403` when sharing is off. `404` when that clip is not on this ride.

The alert socket event does not include the picture. MQTT does not either. `clipId` is `null` until the car finishes the upload, which can take up to about 30 seconds. When the first payload has `clipId: null`, read `GET /alerts` again before showing "no video". Do not try to open `snapshot_path`; that path exists only on the Jetson.

---

## WebSocket

`ws://YOUR_VPS_IP:8000/v1/rides/{rideId}/live`

Replace `http` with `ws`. The server ignores text you send. Close the socket to leave.

| `type` | Payload |
|---|---|
| `telemetry` | `{ "type": "telemetry", "ride": Ride }` |
| `status` | `{ "type": "status", "ride": Ride }` |
| `alert` | `{ "type": "alert", "alert": Alert, "ride": Ride }` |
| `sos` | `{ "type": "sos", "alert": Alert, "ride": Ride, "notified"?: string[] }` |
| `analysis` | `{ "type": "analysis", "channel": "dms" or "cms", "analysis": { scene, hazards, confidence, suppressed, observations, at } }` |

On `telemetry` and `status`, replace the ride in local state. On `alert` and `sos`, replace the ride and prepend the alert. On `analysis`, replace only that channel's scene.

```js
const ws = new WebSocket(`${base.replace(/^http/, "ws")}/v1/rides/${rideId}/live`);
ws.onmessage = (ev) => {
  const msg = JSON.parse(ev.data);
  if (msg.type === "telemetry" || msg.type === "status") setRide(msg.ride);
  else if (msg.type === "alert" || msg.type === "sos") {
    setRide(msg.ride);
    prependAlert(msg.alert);
  } else if (msg.type === "analysis") setAnalysis(msg.channel, msg.analysis);
};
```

---

## SOS

`POST /v1/rides/{rideId}/sos`

```json
{
  "passenger": "Meera Shah",
  "latitude": 12.9716,
  "longitude": 77.5946,
  "source": "phone"
}
```

All fields are optional.

```json
{
  "id": "sos-a1b2c3d4e5f6",
  "at": 1761560000000,
  "topic": "ride/ride-1842/sos",
  "origin": "device",
  "note": "Notified Priya Shah",
  "pushed": 1,
  "alert": {}
}
```

`note` is `"Notified nobody sharing"` when no guardian has sharing on. `alert` is an Alert. `pushed` is how many phones were sent a notification. This also pushes a `sos` event on the live socket.

---

## Telemetry

`POST /v1/rides/{rideId}/telemetry`

`latitude` and `longitude` are required. The other fields are optional and only overwrite the stored value when sent. CamelCase or snake_case is accepted for the optional names.

```json
{
  "latitude": 12.9716,
  "longitude": 77.5946,
  "heading": 90,
  "speedKmh": 34.5,
  "etaMin": 12
}
```

Response is a Ride. Also pushes `telemetry` on the live socket.

---

## Guardians

`GET /v1/rides/{rideId}/guardians`

```json
{
  "guardians": [
    {
      "id": "g-priya",
      "name": "Priya Shah",
      "relationship": "Mother",
      "phone": "+91 98450 11008",
      "sharing": true,
      "lastNotifiedAt": null
    }
  ]
}
```

`lastNotifiedAt` is `null` until that guardian is included in an SOS. An unknown ride returns an empty list, not `404`.

`POST /v1/rides/{rideId}/guardians`

```json
{ "name": "Anita Shah", "relationship": "mother", "phone": "+9198...", "sharing": true }
```

`name` is required. Response: `{ "id": "g-1a2b3c4d" }`. `404` if the ride does not exist.

`POST /v1/rides/{rideId}/guardians/{guardianId}/sharing`

No body. Flips `sharing`. Response: `{ "id": "g-priya", "sharing": false }`. `404` if that guardian is not on this ride.

You cannot edit the name or phone, and you cannot delete a guardian.

---

## Share and notices

`POST /v1/rides/{rideId}/share` with no body.

```json
{ "link": "vms://trip/ride-1842", "origin": "vehicle" }
```

`GET /v1/rides/{rideId}/notices`

```json
{
  "notices": [
    {
      "id": "notice-abc123",
      "at": 1761560000000,
      "title": "Trip link refreshed",
      "body": "vms://trip/ride-1842",
      "guardianNames": ["Priya Shah"],
      "origin": "device"
    }
  ]
}
```

---

## Push token

`POST /v1/devices/{guardianId}/push-token`

`guardianId` must be an existing guardian id (`g-…`), not the Jetson device id. `404` otherwise.

```json
{ "token": "<fcm token>", "ride_id": "ride-1842" }
```

`ride_id` is accepted and ignored. Response: `{ "ok": true }`.

SOS and alerts with severity `high` or `critical` are sent to every sharing guardian who has a token. An Expo token (`ExponentPushToken[...]`) is delivered by Expo. Any other token is delivered by FCM when `FCM_SERVER_KEY` is set on the API. The notification `data` is `rideId`, `alertId`, `kind`, and `severity`. Lower alerts stay on the WebSocket only.

---

## Health

`GET /health`

```json
{
  "ok": true,
  "mqtt": { "host": "emqx", "connected": true, "error": "" },
  "ride": "ride-1842",
  "livekit": { "url": "ws://YOUR_VPS_IP:7880", "configured": true, "mode": "livekit", "watch": "webrtc" }
}
```

Use this to confirm the API is up. The app does not need to poll it.

---

## Guardian recordings and snaps

These are what the guardian starts. They are not alert clips. Alert clips stay on `alerts[].clipId`.

`POST /v1/rides/{rideId}/recordings`

```json
{ "action": "start", "guardianId": "g-priya" }
```

```json
{
  "id": "rec-ab12cd34",
  "rideId": "ride-1842",
  "guardianId": "g-priya",
  "status": "recording",
  "startedAt": 1761560000000,
  "endedAt": null,
  "videoUrl": null,
  "audio": true
}
```

The car records the cabin picture and the cabin microphone until she stops it, or for 10 minutes. `POST /v1/rides/{rideId}/recordings/{id}/cancel` sets `status` to `stopping`. When the file is uploaded, `status` is `ready` and `videoUrl` is `/v1/media/recordings/{id}/clip.mp4`. Play `{BASE}` + `videoUrl`. The MP4 contains the video and the cabin audio.

`GET /v1/rides/{rideId}/recordings?guardianId=g-priya` lists her takes, newest first. Omit `guardianId` for every recording on the trip.

`POST /v1/rides/{rideId}/snapshots`

```json
{ "guardianId": "g-priya", "channel": "dms" }
```

`channel` is `dms` or `cms`. This copies the current live still. `404` if no still has arrived yet.

```json
{
  "id": "snap-ab12cd34",
  "rideId": "ride-1842",
  "guardianId": "g-priya",
  "channel": "dms",
  "at": 1761560000000,
  "url": "/v1/media/snapshots/snap-ab12cd34/snap.jpg"
}
```

`GET /v1/rides/{rideId}/snapshots?guardianId=g-priya` is her snap list. Load `{BASE}` + `url`.

`403` when sharing is off. One recording can be in progress per ride. A second start returns that same recording.

---

## Do not call from the app

These are for the Jetson. They require `MEDIA_DEVICE_KEY`, which the phone must not have.

- `POST /v1/devices/{deviceId}/media-token`
- `POST /v1/devices/{deviceId}/video/frame`
- `POST /v1/devices/{deviceId}/video/live`
- `POST /v1/devices/{deviceId}/clips`
- `POST /v1/devices/{deviceId}/recordings/{id}`
