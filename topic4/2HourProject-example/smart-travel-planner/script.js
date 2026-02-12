/**
 * TripCast — Interactive Elements
 * Trip planner form, results display, scroll effects, mobile nav.
 */

// ==========================================================================
// CONFIGURATION
// ==========================================================================

const API_BASE_URL = '/api';  // Relative URL since served from same origin

// ==========================================================================
// MOBILE NAVIGATION
// ==========================================================================

const mobileMenuBtn = document.querySelector('.mobile-menu-btn');
const mobileNav = document.querySelector('.mobile-nav');
const mobileNavLinks = document.querySelectorAll('.mobile-nav a');

if (mobileMenuBtn && mobileNav) {
  mobileMenuBtn.addEventListener('click', () => {
    mobileNav.classList.toggle('open');
    mobileMenuBtn.classList.toggle('active');
    document.body.style.overflow = mobileNav.classList.contains('open') ? 'hidden' : '';
  });

  // Close mobile nav when clicking a link
  mobileNavLinks.forEach(link => {
    link.addEventListener('click', () => {
      mobileNav.classList.remove('open');
      mobileMenuBtn.classList.remove('active');
      document.body.style.overflow = '';
    });
  });

  // Close on escape key
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && mobileNav.classList.contains('open')) {
      mobileNav.classList.remove('open');
      mobileMenuBtn.classList.remove('active');
      document.body.style.overflow = '';
    }
  });
}

// ==========================================================================
// SCROLL ANIMATIONS (Intersection Observer)
// ==========================================================================

const observerOptions = {
  root: null,
  rootMargin: '0px',
  threshold: 0.1
};

const animationObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.classList.add('visible');
      // Optionally unobserve after animation to save resources
      // animationObserver.unobserve(entry.target);
    }
  });
}, observerOptions);

// Observe all elements with animation classes
const animatedElements = document.querySelectorAll('.fade-in, .stagger-children');
animatedElements.forEach(el => animationObserver.observe(el));

// ==========================================================================
// HEADER SCROLL EFFECT
// ==========================================================================

const header = document.querySelector('.header');
let lastScrollY = window.scrollY;
let ticking = false;

function updateHeader() {
  const currentScrollY = window.scrollY;
  
  // Add shadow on scroll
  if (currentScrollY > 10) {
    header.style.boxShadow = '0 4px 20px rgba(0, 0, 0, 0.3)';
  } else {
    header.style.boxShadow = 'none';
  }
  
  // Hide/show header on scroll direction (optional - uncomment to enable)
  /*
  if (currentScrollY > lastScrollY && currentScrollY > 100) {
    header.style.transform = 'translateY(-100%)';
  } else {
    header.style.transform = 'translateY(0)';
  }
  */
  
  lastScrollY = currentScrollY;
  ticking = false;
}

window.addEventListener('scroll', () => {
  if (!ticking) {
    requestAnimationFrame(updateHeader);
    ticking = true;
  }
});

// ==========================================================================
// SMOOTH SCROLL FOR ANCHOR LINKS
// ==========================================================================

document.querySelectorAll('a[href^="#"]').forEach(anchor => {
  anchor.addEventListener('click', function(e) {
    const href = this.getAttribute('href');
    
    // Skip if it's just "#"
    if (href === '#') return;
    
    const target = document.querySelector(href);
    if (target) {
      e.preventDefault();
      
      const headerHeight = header ? header.offsetHeight : 0;
      const targetPosition = target.getBoundingClientRect().top + window.scrollY - headerHeight - 20;
      
      window.scrollTo({
        top: targetPosition,
        behavior: 'smooth'
      });
    }
  });
});

// ==========================================================================
// TRY IT OUT - TRIP PLANNER FORM
// ==========================================================================

const tripForm = document.getElementById('trip-form');
const destinationInput = document.getElementById('destination');
const startDateInput = document.getElementById('start-date');
const endDateInput = document.getElementById('end-date');
const unitsSelect = document.getElementById('units');
const resultsContainer = document.getElementById('results');
const weatherResults = document.getElementById('weather-results');
const packingResults = document.getElementById('packing-results');
const activityResults = document.getElementById('activity-results');
const resultsTitle = document.getElementById('results-title');
const errorContainer = document.getElementById('error-container');
const errorMessage = document.getElementById('error-message');
const clearResultsBtn = document.getElementById('clear-results');
const retryBtn = document.getElementById('retry-btn');
const cityButtons = document.querySelectorAll('.city-btn');

// Default the date pickers to today → +5 days
if (startDateInput && endDateInput) {
  const today = new Date();
  const inFive = new Date(today);
  inFive.setDate(inFive.getDate() + 5);

  const fmt = d => d.toISOString().split('T')[0];
  startDateInput.value = fmt(today);
  startDateInput.min = fmt(today);
  endDateInput.value = fmt(inFive);
  endDateInput.min = fmt(today);

  // Keep end ≥ start
  startDateInput.addEventListener('change', () => {
    if (endDateInput.value < startDateInput.value) {
      endDateInput.value = startDateInput.value;
    }
    endDateInput.min = startDateInput.value;
  });
}

// Form submission handler
if (tripForm) {
  tripForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    const destination = destinationInput.value.trim();
    const units = unitsSelect.value;
    const startDate = startDateInput?.value || '';
    const endDate = endDateInput?.value || '';

    if (!destination) {
      showError('Please enter a destination city.');
      return;
    }

    await fetchTripPlan(destination, units, startDate, endDate);
  });
}

// Quick city buttons
cityButtons.forEach(btn => {
  btn.addEventListener('click', async () => {
    const city = btn.dataset.city;
    destinationInput.value = city;
    await fetchTripPlan(city, unitsSelect.value);
  });
});

// Clear results
if (clearResultsBtn) {
  clearResultsBtn.addEventListener('click', () => {
    hideResults();
    destinationInput.value = '';
    destinationInput.focus();
  });
}

// Retry button
if (retryBtn) {
  retryBtn.addEventListener('click', () => {
    hideError();
    destinationInput.focus();
  });
}

// Fetch trip plan from API (agent → OpenAI service → hardcoded fallback)
async function fetchTripPlan(city, units = 'metric', startDate = '', endDate = '') {
  showLoading(true);
  hideError();
  hideResults();

  const params = new URLSearchParams({ city, units });
  if (startDate) params.set('start_date', startDate);
  if (endDate) params.set('end_date', endDate);

  try {
    const response = await fetch(`${API_BASE_URL}/travel-plan?${params}`);
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || 'Failed to fetch weather data');
    }

    displayResults(city, data);

  } catch (error) {
    console.error('Error fetching trip plan:', error);
    showError(error.message || 'Unable to fetch weather data. Please try again.');
  } finally {
    showLoading(false);
  }
}

/**
 * Safely render a list of strings into a container via DOM APIs.
 * Avoids innerHTML to prevent XSS from LLM-generated content.
 */
function renderList(container, items) {
  container.textContent = '';           // clear previous content
  const ul = document.createElement('ul');
  for (const text of items) {
    const li = document.createElement('li');
    li.textContent = text;              // textContent is XSS-safe
    ul.appendChild(li);
  }
  container.appendChild(ul);
}

// Display results — uses AI data when available, hardcoded fallback otherwise
function displayResults(city, data) {
  // Title with optional date range
  let title = `🌤️ Trip Plan for ${city}`;
  if (data.start_date && data.end_date) {
    title += ` (${data.start_date} → ${data.end_date})`;
  }
  resultsTitle.textContent = title;

  // Always show the detailed day-by-day weather forecast
  weatherResults.textContent = data.weather || 'No weather data available';

  // AI summary (built with safe DOM methods — no innerHTML with API data)
  const existingSummary = document.getElementById('ai-summary');
  if (existingSummary) existingSummary.remove();

  if (data.ai_powered && data.summary) {
    const summaryEl = document.createElement('div');
    summaryEl.id = 'ai-summary';
    summaryEl.className = 'ai-summary';

    const p = document.createElement('p');
    const badge = document.createElement('span');
    badge.className = 'ai-badge';
    badge.textContent = '✨ AI-Powered';
    p.appendChild(badge);
    p.append(' ' + data.summary);
    summaryEl.appendChild(p);

    weatherResults.after(summaryEl);
  }

  // Packing list — prefer AI, fall back to hardcoded
  const packingItems = (data.ai_powered && data.packing && data.packing.length)
    ? data.packing
    : generatePackingListItems(data);
  renderList(packingResults.querySelector('.results-content'), packingItems);

  // Activities — prefer AI, fall back to hardcoded
  const activityItems = (data.ai_powered && data.activities && data.activities.length)
    ? data.activities
    : generateActivityItems(data);
  renderList(activityResults.querySelector('.results-content'), activityItems);

  resultsContainer.hidden = false;
  resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ---------------------------------------------------------------------------
// FALLBACK generators — used when OpenAI is unavailable
// ---------------------------------------------------------------------------

// Returns a flat array of packing item strings
function generatePackingListItems(data) {
  const items = [];
  const weather = data.weather || '';
  
  const tempMatch = weather.match(/(-?\d+\.?\d*).*?(-?\d+\.?\d*).*?°/);
  let avgTemp = 20;
  if (tempMatch) {
    avgTemp = (parseFloat(tempMatch[1]) + parseFloat(tempMatch[2])) / 2;
  }
  
  const hasRain = weather.toLowerCase().includes('rain') || 
                  weather.match(/\d{2,3}%.*rain/i);
  
  if (avgTemp < 10) {
    items.push('Heavy winter coat', 'Thermal layers', 'Warm sweater', 'Gloves & scarf', 'Warm boots', 'Beanie/warm hat');
  } else if (avgTemp < 20) {
    items.push('Light jacket or fleece', 'Long pants/jeans', 'Layerable tops', 'Closed-toe shoes', 'Light sweater');
  } else if (avgTemp < 30) {
    items.push('Light clothing', 'Shorts & t-shirts', 'Comfortable sandals', 'Light cardigan', 'Walking shoes');
  } else {
    items.push('Very light fabrics', 'Shorts & tank tops', 'Wide-brim sun hat', 'Light colored clothing', 'Breathable shoes');
  }
  
  if (hasRain) {
    items.push('Compact umbrella', 'Rain jacket', 'Waterproof shoes');
  }
  items.push('Sunscreen (SPF 30+)', 'Sunglasses');
  return items;
}

// Returns a flat array of activity strings
function generateActivityItems(data) {
  const activities = [];
  const weather = data.weather || '';
  
  const hasRain = weather.toLowerCase().includes('rain') || 
                  weather.match(/\d{2,3}%.*rain/i);
  const isSunny = weather.toLowerCase().includes('clear') || 
                  weather.toLowerCase().includes('sunny');
  
  const tempMatch = weather.match(/(-?\d+\.?\d*).*?(-?\d+\.?\d*).*?°/);
  let avgTemp = 20;
  if (tempMatch) {
    avgTemp = (parseFloat(tempMatch[1]) + parseFloat(tempMatch[2])) / 2;
  }
  
  if (hasRain) {
    activities.push('Visit museums & galleries', 'Explore local cafés', 'Indoor shopping', 'Cooking class', 'Spa day', 'Catch a local show');
  }
  if (isSunny && avgTemp > 15) {
    activities.push('Walking city tour', 'Park picnic', 'Outdoor markets', 'Rooftop bars', 'Bike rental', 'Scenic viewpoints');
  }
  if (avgTemp < 10) {
    activities.push('Hot springs or thermal baths', 'Cozy restaurant hopping', 'Indoor cultural sites', 'Local coffee shops');
  }
  if (avgTemp > 28) {
    activities.push('Swimming/beach time', 'Early morning sightseeing', 'Air-conditioned museums', 'Evening outdoor dining');
  }
  if (activities.length === 0) {
    activities.push('Explore the city center', 'Visit local landmarks', 'Try local cuisine', 'People-watch at a café', 'Take photos at scenic spots');
  }
  return activities;
}

// Show/hide loading state
function showLoading(isLoading) {
  const submitBtn = tripForm?.querySelector('button[type="submit"]');
  if (!submitBtn) return;
  
  const btnText = submitBtn.querySelector('.btn-text');
  const btnLoading = submitBtn.querySelector('.btn-loading');
  const btnArrow = submitBtn.querySelector('.btn-arrow');
  
  if (isLoading) {
    btnText.hidden = true;
    btnLoading.hidden = false;
    btnArrow.hidden = true;
    submitBtn.disabled = true;
  } else {
    btnText.hidden = false;
    btnLoading.hidden = true;
    btnArrow.hidden = false;
    submitBtn.disabled = false;
  }
}

// Show error
function showError(message) {
  if (errorContainer && errorMessage) {
    errorMessage.textContent = message;
    errorContainer.hidden = false;
    resultsContainer.hidden = true;
  }
}

// Hide error
function hideError() {
  if (errorContainer) {
    errorContainer.hidden = true;
  }
}

// Hide results
function hideResults() {
  if (resultsContainer) {
    resultsContainer.hidden = true;
  }
}

// ==========================================================================
// PERFORMANCE: Defer non-critical operations
// ==========================================================================

window.addEventListener('load', () => {
  // Preload fonts for better performance
  const fonts = [
    'DM Sans',
    'Libre Baskerville'
  ];
  
  fonts.forEach(font => {
    document.fonts.ready.then(() => {
      console.log(`Font loaded: ${font}`);
    });
  });
});