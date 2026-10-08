(() => {
  const cover = document.querySelector('[data-home-cover]');
  if (!cover) return;

  const toggle = cover.querySelector('[data-home-cover-toggle]');
  const label = cover.querySelector('[data-home-cover-toggle-label]');
  const motion = window.matchMedia('(prefers-reduced-motion: reduce)');
  let paused = false;
  let visible = true;

  const update = () => {
    toggle.hidden = motion.matches;
    toggle.setAttribute('aria-pressed', String(paused));
    toggle.setAttribute('aria-label', paused ? '播放封面动画' : '暂停封面动画');
    label.textContent = paused ? '播放' : '暂停';
    cover.dataset.animation = motion.matches ? 'reduced'
      : paused ? 'paused'
      : document.hidden || !visible ? 'suspended'
      : 'running';
  };

  toggle.addEventListener('click', () => {
    paused = !paused;
    update();
  });
  document.addEventListener('visibilitychange', update);
  motion.addEventListener('change', update);

  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver(entries => {
      visible = entries[0].isIntersecting;
      update();
    });
    observer.observe(cover);
  }

  update();
  cover.dataset.enhanced = '';
})();
