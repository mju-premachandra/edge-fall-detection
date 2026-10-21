# Five-minute demonstration

Open the software through `start_demo.bat`. Keep its command window open.

1. **Show the connection.** Point to “Backend connected”. Explain that the browser
   sends sensor readings to a Python Flask API, which runs a saved trained model.
2. **Normal input.** Select Manual input, Load walking window, then Analyze input.
   Show “No fall predicted”, the score and actual backend response. This is a
   known training excerpt used to demonstrate the workflow.
3. **Fall and cancellation.** Set Cancellation time to 30. Load fall window and
   Analyze input. Show the suspected-fall decision, visual prompt and countdown.
   Click Cancel alert. Show the cancelled row in Alert history.
4. **Fall and escalation.** Set Cancellation time to 3 and Analyze input again.
   Wait without cancelling. Show escalation to the local receiver, then click
   Acknowledge receipt. The stored history records both actions.
5. **Your own input.** Edit values in the pasted sequence, or upload a CSV from
   `examples`. Analyze again and expand View actual backend response. The model
   uses the entered values; edits do not guarantee the class will change.
6. **Evidence.** Show Training & evaluation, its real precision/recall and false
   alarms. Export events as CSV. Restart the server and show that history persists.

Suggested explanation:

“We completed the software input-to-output integration before the deadline.
Recorded or manually entered accelerometer and gyroscope sequences are sent
from the frontend to the backend. The trained model returns a suspected-fall
decision. A configurable cancellation period then leads to cancellation or
local escalation and acknowledgement. Physical vibration is simulated visually.
Hardware integration and further accuracy improvement remain future work.”

## Work plan before October 22

| Date | Deliverable |
|---|---|
| Oct 16 | Launch this app and complete walking, cancellation and escalation tests |
| Oct 17 | Practise CSV upload/manual edits; understand the input fields and API |
| Oct 18 | Review training script, participant split, actual results and limits |
| Oct 19 | Agree the team’s field names and event API; connect their software if ready |
| Oct 20 | Capture screenshots, export event evidence and record a short demo |
| Oct 21 | Rehearse from a clean launch, back up the ZIP and freeze the demonstration |

If the team application is not ready, the included frontend and local receiver
still demonstrate your individual software component. Do not describe a separate
team application or physical alert device as integrated until actually tested.
