// Paste over createMqtt + createSos in the VMS repo services/createServices.ts
// after EXPO_PUBLIC_TELEMETRY_URL points at the Render API.

function createSos(telemetryUrl: string): SosService {
  return {
    async raise({ rideId, coordinate, passenger }) {
      const response = await fetch(`${telemetryUrl}/v1/rides/${rideId}/sos`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          passenger,
          latitude: coordinate.latitude,
          longitude: coordinate.longitude,
          source: "phone",
        }),
      });
      if (!response.ok) {
        throw new Error(`SOS failed (${response.status})`);
      }
      const body = await response.json();
      return {
        id: body.id,
        at: body.at,
        topic: body.topic,
        origin: "device",
        note: body.note,
      };
    },
  };
}
