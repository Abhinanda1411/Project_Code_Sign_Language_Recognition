// app.js - shows what the server sees and sends button presses back.
const $ = id => document.getElementById(id);
let lastText = "";
let lastSpeak = null;
let cameraOn = true;
let lastSugg = "";

async function post(name, value = "") {
  try {
    const r = await fetch(`/action/${name}`, {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({value})
    });
    return await r.json();
  } catch (e) { return null; }
}

function currentWord(text) { return text.split(" ").pop(); }

function render(s) {
  const chip = $("status");
  cameraOn = s.camera_on;
  $("btn-camera").textContent = s.camera_on ? "Turn camera off" : "Turn camera on";
  $("btn-camera").classList.toggle("off", !s.camera_on);
  document.querySelector(".camera").classList.toggle("is-off", !s.camera_on);
  if (!s.camera_on && !s.error) { chip.textContent = "Camera off"; chip.className = "chip"; }
  else if (s.error) { chip.textContent = s.error; chip.className = "chip bad"; }
  else if (!s.hand) { chip.textContent = "No hand detected"; chip.className = "chip warn"; }
  else if (s.state === "RECORDING") { chip.textContent = "Reading J / Z movement…"; chip.className = "chip ok"; }
  else { chip.textContent = `Ready · ${s.fps} fps`; chip.className = "chip ok"; }

  $("letter").textContent = s.prediction && s.prediction !== "REST" ? s.prediction
                          : (s.prediction === "REST" ? "rest" : "–");
  $("holdbar").style.width = Math.round(s.hold * 100) + "%";
  $("dynamic").textContent = s.dynamic && s.dynamic !== "NONE" ? `Movement: ${s.dynamic}` : "";

  $("text").textContent = s.text;
  $("clearbar").style.width = Math.round(s.clear_hold * 100) + "%";
  $("palms").textContent = `Open palms: ${s.palms}/2`;
  const cm = $("clearmsg");
  cm.textContent = s.cleared ? "Cleared" : "Clear: both open palms";
  cm.className = s.cleared ? "label done" : "label";

  $("speakbar").style.width = Math.round(s.speak_hold * 100) + "%";
  $("thumb").textContent = `Thumbs up: ${s.thumb ? "yes" : "no"}`;
  const sm = $("speakmsg");
  sm.textContent = s.speaking ? "Speaking" : "Speak: thumbs up";
  sm.className = s.speaking ? "label done" : "label";

  // a thumbs up held long enough makes the server raise speak_seq -> speak the text
  if (lastSpeak === null) lastSpeak = s.speak_seq;
  else if (s.speak_seq !== lastSpeak) { lastSpeak = s.speak_seq; Speech.speak(s.text); }

  const box = $("suggestions");
const key = (s.suggestions || []).join("|");

if (key !== lastSugg) {
    lastSugg = key; // Fixed: was astSugg = key
    box.innerHTML = "";

    (s.suggestions || []).forEach(w => {
        const b = document.createElement("button");
        b.type = "button";
        b.textContent = w;

        b.addEventListener("click", async () => {
            b.disabled = true;

            try {
                await post("accept", w);

                // Refresh the displayed text immediately after accepting
                const r = await fetch("/state");
                if (r.ok) {
                    render(await r.json());
                }
            } catch (e) {
                console.error("Could not accept suggestion:", e);
            } finally {
                b.disabled = false;
            }
        });

        box.appendChild(b);
    });
}

  // speak the finished word when Space was pressed (optional)
  if ($("autospeak").checked && s.text.length > lastText.length
      && s.text.endsWith(" ") && !lastText.endsWith(" ")) {
    const words = s.text.trim().split(" ");
    Speech.speak(words[words.length - 1]);
  }
  lastText = s.text;
}

async function poll() {
  try {
    const r = await fetch("/state");
    render(await r.json());
  } catch (e) {
    $("status").textContent = "Server not reachable";
    $("status").className = "chip bad";
  }
  setTimeout(poll, 100);
}

$("btn-space").onclick = () => post("space");
$("btn-back").onclick = () => post("backspace");
$("btn-clear").onclick = () => post("clear");
$("btn-camera").onclick = () => post(cameraOn ? "camera_off" : "camera_on");
$("btn-speak").onclick = () => Speech.speak($("text").textContent);
document.addEventListener("keydown", e => {
  if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT") return;
  if (e.code === "Space") { e.preventDefault(); post("space"); }
  if (e.code === "Backspace") { e.preventDefault(); post("backspace"); }
});

if (!Speech.supported) {
  $("speech-note").textContent = "This browser has no speech support. Try Chrome or Edge.";
} else {
  $("speech-note").textContent = "Voices come from your device, so they differ between computers.";
}
poll();