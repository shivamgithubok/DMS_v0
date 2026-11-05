# Guardian app: how to decode video and images

The cabin device encodes. The Android app decodes. There is no decoder API. If the Guardian screen stays blank, the bytes arrived in a form the player was not given.

There are three different things. Use a different view for each.

## 1. Preview stills

Source: `POST /v1/devices/{id}/video/frame`, about every 2 seconds.

JSON field: `image_b64`
Also: `track` is `driver` or `cabin`.

What it is: base64 of a JPEG. After base64 decode, the bytes start with `FF D8`.

Show it like this:

1. Base64-decode `image_b64` to a byte array.
2. Decode that array as a JPEG (`BitmapFactory`, Coil, or Glide from bytes).
3. Put the bitmap in an `ImageView`.

Do not give this string to `VideoView` or ExoPlayer. It is a picture, not an MP4.

## 2. Live watch

When the guardian is watching, the device joins LiveKit and publishes:

- Video track `driver`
- Video track `cabin` (640×360, about 12 fps, 800 kbps, H.264 from LiveKit)
- Audio track `cabin` (16 kHz, mono)

Show it like this:

1. Join the same LiveKit room with the subscriber token.
2. Subscribe to the remote tracks named `driver` and `cabin`.
3. Attach each video track to its own `SurfaceViewRenderer` or `TextureViewRenderer`.
4. Attach the `cabin` audio track for sound.

LiveKit decodes the video. Do not JPEG-decode these frames, and do not write them to an MP4.

Stills stop while these tracks are live. If the guardian is no longer marked as watching, the device leaves the room and the last frame stays on screen. That is expected.

## 3. Recorded clip

JSON field: `video_b64`
What it is: base64 of one complete MP4 file.

Inside the file:

- Video: H.264, `yuv420p`
- Audio: AAC, only if the cabin mic had audio
- `faststart` is set, so playback can start before the full download finishes

Show it like this:

1. Base64-decode `video_b64` to bytes.
2. Confirm the bytes are an MP4 (they contain `ftyp` near the start).
3. Write them to a `.mp4` file, or pass the raw bytes to the player.
4. Play with ExoPlayer (H.264 + AAC). `VideoView` is fine for a normal MP4.

Do not set the base64 text as the video URL. The player must receive the MP4 bytes. If the download is cut short, the player cannot open the file. Check that the decoded size matches what the API sent.

## If the screen is blank

| Field or track you got | What to do |
|---|---|
| `image_b64` | Base64-decode, then JPEG-decode into an `ImageView` |
| LiveKit `driver` / `cabin` | Subscribe and render the track. LiveKit is the decoder |
| `video_b64` | Base64-decode to an `.mp4`, then play with ExoPlayer |

Base64 only turns binary into text for JSON. After that step, stills are JPEG and clips are H.264/AAC MP4.
