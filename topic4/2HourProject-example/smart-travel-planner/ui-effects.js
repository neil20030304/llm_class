/**
 * TripCast UI Effects — ripple, parallax, counters, form interactions.
 *
 * Separated from trip-planner logic (script.js) to keep files under 600 lines.
 * Load this BEFORE script.js in index.html.
 */

// ==========================================================================
// BUTTON RIPPLE EFFECT
// ==========================================================================

function createRipple(event) {
  const button = event.currentTarget;

  const existingRipple = button.querySelector('.ripple');
  if (existingRipple) existingRipple.remove();

  const ripple = document.createElement('span');
  ripple.classList.add('ripple');

  const rect = button.getBoundingClientRect();
  const size = Math.max(rect.width, rect.height);

  ripple.style.width = ripple.style.height = `${size}px`;
  ripple.style.left = `${event.clientX - rect.left - size / 2}px`;
  ripple.style.top = `${event.clientY - rect.top - size / 2}px`;

  button.appendChild(ripple);
  ripple.addEventListener('animationend', () => ripple.remove());
}

document.querySelectorAll('.btn-primary').forEach(button => {
  button.style.position = 'relative';
  button.style.overflow = 'hidden';
  button.addEventListener('click', createRipple);
});

const rippleStyles = document.createElement('style');
rippleStyles.textContent = `
  .ripple {
    position: absolute;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.3);
    transform: scale(0);
    animation: ripple-animation 0.6s linear;
    pointer-events: none;
  }
  @keyframes ripple-animation {
    to { transform: scale(4); opacity: 0; }
  }
`;
document.head.appendChild(rippleStyles);

// ==========================================================================
// PARALLAX FLOATING CARDS
// ==========================================================================

const floatingCards = document.querySelectorAll('.floating-card');

if (floatingCards.length > 0) {
  document.addEventListener('mousemove', (e) => {
    const mouseX = e.clientX / window.innerWidth - 0.5;
    const mouseY = e.clientY / window.innerHeight - 0.5;

    floatingCards.forEach((card, index) => {
      const speed = (index + 1) * 10;
      card.style.transform = `translate(${mouseX * speed}px, ${mouseY * speed}px)`;
    });
  });
}

// ==========================================================================
// PRICING CARD HOVER EFFECT
// ==========================================================================

const pricingCards = document.querySelectorAll('.pricing-card');

pricingCards.forEach(card => {
  card.addEventListener('mouseenter', () => {
    pricingCards.forEach(c => { if (c !== card) c.style.opacity = '0.7'; });
  });
  card.addEventListener('mouseleave', () => {
    pricingCards.forEach(c => { c.style.opacity = '1'; });
  });
});

// ==========================================================================
// COUNTER ANIMATION FOR STATS
// ==========================================================================

function animateCounter(element, target, duration = 2000) {
  let start = 0;
  const increment = target / (duration / 16);

  function tick() {
    start += increment;
    if (start < target) {
      element.textContent = Math.floor(start).toLocaleString();
      requestAnimationFrame(tick);
    } else {
      element.textContent = target.toLocaleString();
    }
  }
  tick();
}

const statValues = document.querySelectorAll('.stat-value');
const statsObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      const text = entry.target.textContent;
      if (text.includes('+')) {
        const num = parseInt(text.replace(/[^0-9]/g, ''));
        entry.target.textContent = '0+';
        animateCounter(entry.target, num);
        entry.target.textContent = num.toLocaleString() + '+';
      }
      statsObserver.unobserve(entry.target);
    }
  });
}, { threshold: 0.5 });

statValues.forEach(stat => statsObserver.observe(stat));

// ==========================================================================
// FORM INTERACTIONS
// ==========================================================================

function initForms() {
  document.querySelectorAll('input, textarea').forEach(input => {
    input.addEventListener('focus', () => input.parentElement?.classList.add('focused'));
    input.addEventListener('blur', () => {
      if (!input.value) input.parentElement?.classList.remove('focused');
      if (input.hasAttribute('required') && !input.value) {
        input.classList.add('error');
      } else {
        input.classList.remove('error');
      }
    });
  });
}

if (document.querySelector('form')) initForms();

// ==========================================================================
// CONSOLE EASTER EGG
// ==========================================================================

console.log(`
%c🌤️ TRIPCAST %c

Welcome, curious developer! 👋
Powered by OpenWeatherMap + OpenAI.
Built with ❤️ for the Smart Travel Planner project.
`,
'color: #00D4AA; font-size: 24px; font-weight: bold;',
'color: #888; font-size: 12px;'
);
