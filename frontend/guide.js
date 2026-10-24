// guide.js - the Sign Guide page: letter cards + live highlight of what the camera sees.
const LETTERS = [
  ["A", "Fist, thumb resting against the side of the index finger."],
  ["B", "Flat hand, fingers straight up and together, thumb folded across the palm."],
  ["C", "Fingers and thumb curved to make a C shape."],
  ["D", "Index finger points up; the other fingers and thumb curve to touch, like a small circle."],
  ["E", "Fingertips curled down onto the palm, thumb tucked under the fingers."],
  ["F", "Thumb and index finger touch to make a circle; the other three fingers point up."],
  ["G", "Index finger and thumb point sideways, parallel, like a small pinch; other fingers closed."],
  ["H", "Index and middle fingers stretched out sideways together; thumb and other fingers closed."],
  ["I", "Fist with the little finger pointing up."],
  ["J", "Make an I, then draw a J in the air with the little finger.", "motion"],
  ["K", "Index and middle fingers up in a V, thumb touching the base of the middle finger."],
  ["L", "Index finger up, thumb out to the side: an L shape."],
  ["M", "Thumb tucked under the first three fingers (fingers lie over the thumb)."],
  ["N", "Thumb tucked under the first two fingers."],
  ["O", "All fingertips touch the thumb, making an O."],
  ["P", "Like K, but pointing downwards."],
  ["Q", "Like G, but pointing downwards."],
  ["R", "Index and middle fingers crossed; other fingers closed."],
  ["S", "Fist with the thumb across the front of the fingers."],
  ["T", "Fist with the thumb tucked between the index and middle fingers."],
  ["U", "Index and middle fingers up and together; thumb holds the other fingers down."],
  ["V", "Index and middle fingers up and apart: a V (peace sign)."],
  ["W", "Index, middle and ring fingers up and spread; thumb holds the little finger down."],
  ["X", "Fist with the index finger bent into a hook."],
  ["Y", "Thumb and little finger stretched out, other fingers folded."],
  ["Z", "Draw a Z in the air with the index finger.", "motion"],
];

const grid = document.getElementById("grid");
const cards = {};

LETTERS.forEach(([letter, text, kind]) => {
  const card = document.createElement("article");
  card.className = "lcard";
  card.innerHTML = `
    <div class="lhead"><span class="lbig">${letter}</span>${kind ? '<span class="tag">motion</span>' : ""}</div>
    <div class="lpic"><img alt="Sign for ${letter}"></div>
    <p>${text}</p>`;
  const img = card.querySelector("img");
  // try your own photo first, then the drawing from the recordings, else hide the picture
  const sources = [`signs/photos/${letter}.jpg`, `signs/photos/${letter}.png`, `signs/${letter}.svg`];
  let i = 0;
  img.onerror = () => { i += 1; if (i < sources.length) img.src = sources[i]; else card.querySelector(".lpic").style.display = "none"; };
  img.src = sources[0];
  grid.appendChild(card);
  cards[letter] = card;
});

// ---- live highlight: the letter the translator currently sees ----
const seen = document.getElementById("seen");
let active = null;

function highlight(letter) {
  if (active && cards[active]) cards[active].classList.remove("active");
  active = letter && cards[letter] ? letter : null;
  if (active) cards[active].classList.add("active");
}

async function poll() {
  try {
    const s = (await (await fetch("/state")).json());
    const dyn = s.dynamic && s.dynamic !== "NONE" ? s.dynamic : "";
    const letter = dyn || (s.prediction && s.prediction !== "REST" ? s.prediction : "");
    seen.textContent = letter ? `Live: ${letter}` : (s.hand ? "Live: hand seen" : "Live: no hand");
    seen.className = "chip ok";
    highlight(letter);
  } catch (e) {
    seen.textContent = "Live: not connected";
    seen.className = "chip";
    highlight(null);
  }
  setTimeout(poll, 250);
}
poll();