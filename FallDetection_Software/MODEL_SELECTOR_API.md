# Model selection update

The existing API still defaults to `fall-tree-v1`. An installed forest is exposed
as `fall-forest-v1`. Models and reports load at backend startup; restart after retraining.

| Request | Behaviour |
|---|---|
| `GET /api/models` | Available model IDs, names, thresholds, default ID and loading messages |
| `GET /api/model?model_id=fall-forest-v1` | Selected model metadata and evaluation adapted to the UI schema |
| `POST /api/analyze` | Accepts optional `model_id` with the existing CSV, samples or recording request |
| `POST /api/predict` | Accepts optional `model_id` alongside `samples`; returns last eligible prediction |

Example batch request:

```json
{"model_id":"fall-forest-v1","sample_id":"Subject1Activity1Trial1_belt"}
```

For file uploads, include a multipart text field `model_id` alongside `file`.
Unknown or unavailable model IDs return HTTP 400, rather than substituting another model.
The response and every prediction include the actual model ID.

Alert creation still requires an actual positive decision ID. Its model ID is
read from the stored decision. Switching the frontend selection cannot relabel
an existing event. The original cancellation, escalation, acknowledgement,
idempotency and CSV export rules remain in effect.

Existing SQLite histories are migrated by adding a `model_id` column to the
decisions table, with `fall-tree-v1` for legacy decisions. No rows are removed.

The forest's evaluation uses the reused subjects 5-6 comparison. Its second
table row compares against the original decision tree. When the original tree
is selected, its original evaluation compares against the acceleration baseline.
