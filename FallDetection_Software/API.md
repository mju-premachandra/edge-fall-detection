# Local integration contract

Base URL when running: `http://127.0.0.1:5050`. JSON requests use
`Content-Type: application/json`. Validation errors return HTTP 400 and
`{"error":"explanation"}`. This is a same-machine demo, bound to loopback;
a separate team server requires deliberate networking and deployment work.

## Prediction

`POST /api/predict` accepts a complete recent sequence:

```json
{"samples":[{"time_s":0,"acc_x_g":0,"acc_y_g":1,"acc_z_g":0,"gyro_x_deg_s":0,"gyro_y_deg_s":0,"gyro_z_deg_s":0}]}
```

This shows the **schema of one row only**; a one-row request is rejected.
Send at least a one-second sequence satisfying README timing rules. See
`examples/predict_request.json` for a full runnable request using a real
training excerpt. The response contains `model_id`, `run_id`, `decision_id`,
`time_s`, `start_s`, `samples`, `features`, `fall_score`, `suspected_fall` and
`label`. It returns the last eligible window, not the maximum score in the file.

`POST /api/analyze` returns all eligible windows, sensor chart trace and counts.
Choose one body format:

- `{"samples":[...]}`: sensor objects.
- `{"csv":"time_s,acc_x_g,...\n..."}`: CSV text.
- `{"sample_id":"Subject1Activity1Trial1_belt"}`: an included recording.
- Multipart form with a CSV file field named `file`.

The browser uses `/api/analyze`. Batch analysis selects the first suspected-fall
window for display and starts its alert immediately after analysis. Replay is a
visualization of already computed results, not live acquisition. No activity
labels, filenames or source identifiers affect scores.

## Alerts

For a decision where `suspected_fall` is true, call `POST /api/events`:

```json
{"decision_id":"COPY_FROM_PREDICTION","worker_id":"DEMO-001","cancellation_seconds":30}
```

An actual backend-generated decision ID is required; client-supplied fall labels
cannot open an event. `cancellation_seconds` is editable from 1–120. Response:
HTTP 201 with `{"event":{...},"duplicate":false}`. Repeat requests for the same
analysis run return the same event, preventing duplicate alerts during replay.

| Request | Action |
|---|---|
| `GET /api/events` | Latest 100 events and transition histories; reconcile expired deadlines |
| `POST /api/events/{id}/cancel`, body `{}` | Cancel while pending; HTTP 409 after expiry |
| `POST /api/events/{id}/acknowledge`, body `{}` | Acknowledge an escalated event; HTTP 409 in other states |
| `GET /api/events/export.csv` | Export all stored events |

State transitions: pending -> cancelled, or pending -> escalated -> acknowledged.
Repeated cancellation or acknowledgement returns the existing outcome. The
server enforces the deadline. The event records score, model ID, source, worker
ID, timestamps and history. No email, SMS or outbound network request is sent.

## Other endpoints

| Request | Response |
|---|---|
| `GET /api/health` | Backend availability and loaded model ID |
| `GET /api/model` | Actual model metadata, policy and evaluation report |
| `GET /api/samples` | Included recording IDs and split metadata |
| `GET /api/samples/{id}/csv` | Download sensor-only input CSV |
| `GET /api/examples/walking` or `/fall` | Short known training excerpt as CSV text |

To connect a future sensor sender, buffer recent readings into the same schema
and send sequences to `/api/predict`. An adapter must convert the sensor’s
units, timestamps, axes and placement to the training convention. The current
code does not implement serial, Bluetooth, MQTT or device firmware.
