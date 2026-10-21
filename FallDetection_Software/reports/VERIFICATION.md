# Verification — October 16, 2026

## Backend checks

9 automated unittest cases passed under Python 3.12.14, Flask 3.1.3, pandas 2.2.3 and NumPy 2.3.5.

The checks exercise real sensor inference, CSV/JSON parity, exclusion of label columns from features, input validation, actual-prediction requirements for alerts, durable cancellation, server deadline enforcement, restart recovery, acknowledgement, export, and causal features.

## Browser checks

Chromium 140 with Playwright 1.55; actual local Flask HTTP server, temporary SQLite database. Desktop viewport 1440 x 1100; mobile viewport 390 x 844.

- Frontend loads actual model metadata and connects to backend
- Real walking input travels through API and displays non-fall prediction
- Fall input opens configurable cancellation window; cancel updates saved history
- Real elapsed timeout escalates in backend and UI acknowledgement succeeds
- CSV file upload produces actual model response
- Recorded prediction replay completes
- Cancelled and acknowledged event history survives browser reload
- Invalid input displays actionable error
- 390px mobile layout has no horizontal page overflow

No JavaScript page errors were observed. Desktop and mobile screenshots were visually inspected.

## Training checks

The portable JSON model matched scikit-learn scores on every train, validation and test window within 1e-12. Participant splits and model evaluation are recorded in training_report.json. This verifies software behavior; it does not establish real-world fall-detection accuracy.

## Limits of verification

Windows launcher commands were reviewed for the existing .venv313 layout but were not executed on a Windows machine. Physical sensors, haptic hardware, external receivers and the team application were not connected. Browser checks used known training excerpts for workflow demonstrations.
