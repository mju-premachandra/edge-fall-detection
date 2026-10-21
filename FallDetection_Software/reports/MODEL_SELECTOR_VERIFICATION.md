# Model selector update verification — 18 October 2026

19 unittest cases passed: original backend flow, forest inference, both model routes, invalid IDs, event model identity, cancellation, timeout, restart, CSV export and migration of the previous SQLite schema.

JavaScript syntax checked with node --check.

## Browser verification

Chromium 140 / Playwright 1.55, actual Flask HTTP server with temporary SQLite history.

- Both trained models available; default Decision Tree remains compatible
- Forest selection updates actual model ID, metrics and reused-data labels
- Walking sensor sequence runs the forest and displays its actual score
- Forest fall prediction opens an alert with correct model identity and cancellation works
- Switching to Decision Tree clears stale results and runs its own prediction/alert
- CSV upload forwards selected forest model to backend
- Complete recording runs forest; real timeout and acknowledgement preserve its identity
- Forest recording replay produces no duplicate event
- Reload preserves history and model identity for both models
- 390px mobile layout includes usable model selector without horizontal page overflow

No JavaScript page errors observed. Desktop and 390px mobile screenshots visually inspected internally.

The update contains no model, dataset or SQLite database. Existing locally trained weights and reports are used after restarting the server. The Windows copy/restart steps were documented but not executed on a Windows machine. Physical hardware and external notifications remain outside this update.
