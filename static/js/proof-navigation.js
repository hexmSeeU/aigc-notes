// A direct heading link should also reveal the folded proof containing it.
(() => {
  function revealHash() {
    let id;
    try { id = decodeURIComponent(location.hash.slice(1)); } catch { return; }
    const target = id && document.getElementById(id);
    if (!target) return;
    let ancestor = target.parentElement;
    let revealed = false;
    while (ancestor) {
      if (ancestor.matches('details.proof') && !ancestor.open) {
        ancestor.open = true;
        revealed = true;
      }
      ancestor = ancestor.parentElement;
    }
    if (revealed) requestAnimationFrame(() => target.scrollIntoView());
  }
  addEventListener('hashchange', revealHash);
  addEventListener('DOMContentLoaded', revealHash);
  document.addEventListener('click', event => {
    const link = event.target.closest?.('a[href^="#"]');
    if (link && link.hash === location.hash) revealHash();
  });
})();
