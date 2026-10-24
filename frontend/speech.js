// speech.js - text-to-speech using the browser's built-in voices.
// Browsers only allow speech after the user has clicked something once,
// so the first press of "Speak" (or any button) unlocks it.
const Speech = (() => {
  const supported = "speechSynthesis" in window;
  let voices = [];

  function loadVoices() {
    if (!supported) return;
    voices = window.speechSynthesis.getVoices();
    const select = document.getElementById("voice");
    const keep = select.value;
    select.innerHTML = "";
    voices.forEach((v, i) => {
      const opt = document.createElement("option");
      opt.value = i;
      opt.textContent = `${v.name} (${v.lang})`;
      select.appendChild(opt);
    });
    // prefer an English voice by default
    const en = voices.findIndex(v => v.lang.toLowerCase().startsWith("en"));
    select.value = keep || (en >= 0 ? en : 0);
  }

  function speak(text) {
    if (!supported || !text.trim()) return;
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text.toLowerCase());
    const i = parseInt(document.getElementById("voice").value, 10);
    if (voices[i]) { u.voice = voices[i]; u.lang = voices[i].lang; }
    u.rate = parseFloat(document.getElementById("rate").value);
    window.speechSynthesis.speak(u);
  }

  // Browsers block speech until the user has interacted with the page once.
  // The first click or key press "unlocks" it with a silent utterance.
  function prime() {
    if (!supported) return;
    window.speechSynthesis.speak(new SpeechSynthesisUtterance(""));
  }
  ["click", "keydown", "touchstart"].forEach(ev =>
    document.addEventListener(ev, prime, { once: true }));

  if (supported) {
    loadVoices();
    window.speechSynthesis.onvoiceschanged = loadVoices;
  }
  return { speak, supported };
})();