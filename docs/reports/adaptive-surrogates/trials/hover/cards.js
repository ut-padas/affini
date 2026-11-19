// Show a symbol's card beside it on hover or keyboard focus, and light up every other
// occurrence of the same symbol, like an editor's "highlight references".
(() => {
  const card = document.createElement("div");
  card.id = "card";
  card.hidden = true;
  card.setAttribute("role", "tooltip");
  document.body.appendChild(card);
  let hideTimer = null;

  function show(el) {
    const key = el.dataset.sym;
    const tpl = document.getElementById("card-" + key);
    if (!tpl) return;
    clearTimeout(hideTimer);
    card.replaceChildren(tpl.content.cloneNode(true));
    // Where the symbol is defined: the glossary entry links to its section.
    const back = document.createElement("div");
    back.className = "card-where";
    back.innerHTML = `<a href="#gloss-${key}">glossary entry</a>`;
    card.appendChild(back);
    card.hidden = false;
    const r = el.getBoundingClientRect();
    const top = window.scrollY + r.bottom + 6;
    let left = window.scrollX + r.left - 12;
    left = Math.min(left, window.scrollX + document.documentElement.clientWidth - card.offsetWidth - 12);
    card.style.top = `${top}px`;
    card.style.left = `${Math.max(8, left)}px`;
    document.querySelectorAll(".sym").forEach((s) => {
      s.classList.toggle("lit", s.dataset.sym === key);
      s.classList.toggle("here", s === el);
    });
  }

  function hide() {
    hideTimer = setTimeout(() => {
      card.hidden = true;
      document.querySelectorAll(".sym.lit, .sym.here").forEach((s) => s.classList.remove("lit", "here"));
    }, 120);
  }

  document.querySelectorAll(".sym").forEach((el) => {
    el.addEventListener("mouseenter", () => show(el));
    el.addEventListener("mouseleave", hide);
    el.addEventListener("focus", () => show(el));
    el.addEventListener("blur", hide);
  });
  card.addEventListener("mouseenter", () => clearTimeout(hideTimer));
  card.addEventListener("mouseleave", hide);
})();
