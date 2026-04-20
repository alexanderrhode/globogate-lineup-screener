/**
 * GLOBOGATE Lineup Screener — v4 Dropout Risk Scoring
 * Client-side Logistic Regression with API-backed candidate search.
 * Fetches person data from GLOBOGATE External API via Netlify proxy,
 * auto-maps fields to model features, and scores dropout risk.
 */

// ═══════════════════════════════════════════════════════════
//  MODEL COEFFICIENTS (v4 LR)
// ═══════════════════════════════════════════════════════════

const MODELS = {
  Philippines: {
    intercept: 0.24108842150516707,
    features: {
      age: -0.037970460328578835,
      age_sq: 0.0016924022863235293,
      is_male: 0.283417164937886,
      is_married: -0.10000744086235193,
      is_single: -0.14993612454247787,
      has_icu: 0.3221129416273702,
      years_exp: -0.3115989562047567,
      years_exp_sq: 0.014059300138390364,
      hosp_Tertiary: -0.24774047268760166,
      hosp_Primary: 0.10400686880579482,
      'cat_General Ward': -0.021326044436436195,
      cat_ICU: -0.6534048219082038,
      cat_Med_Surg: -0.1993545238431912,
      cat_OR: -0.3148245820166344,
      cat_ER: -0.26596314513169683,
      cat_Geriatrics: -0.5653121644749984,
      cat_Pediatrics: -0.3221616479541042,
      cat_Other: 0.20175042425524914,
      cat_OB_GYN: -0.0755136281224192,
      cat_Dialysis: -0.030185067966975645,
      cat_Psych: -0.033029927697600235,
      region_dropout: 2.6611975395320533,
    },
    hospitals: ['Tertiary', 'Secondary', 'Primary'],
    categories: ['General Ward', 'ICU', 'ER', 'OR', 'Med-Surg', 'Geriatrics', 'Pediatrics', 'OB-GYN', 'Dialysis', 'Psych', 'Other'],
    hasRegion: true,
    baselineDropout: 0.37,
    // Quartil-Schwellen (P25/P50/P75 ueber alle ~16k PH-Kandidaten)
    tiers: [0.497, 0.591, 0.703],
    countryFilter: ['Philippines'],
  },
  Uzbekistan: {
    intercept: 0.1856603545859344,
    features: {
      age: -0.16562280756972808,
      age_sq: 0.0031631051469227373,
      is_male: -0.25330722053313465,
      is_married: -0.00646283701846286,
      is_single: -0.1807191209182684,
      has_icu: -0.4929705963387055,
      years_exp: 0.10669944107563821,
      years_exp_sq: -0.004312758126933724,
      hosp_Regional: -0.20687943412460366,
      hosp_Republican: -0.09852933175809717,
      hosp_City_District: 0.4379471865881832,
      'cat_General Ward': 0.22952661090431986,
      cat_ICU: 0.5202117479348337,
      cat_Psych: 0.16271796393088042,
      cat_ER: -0.3837574715815463,
      cat_Other: -0.1629204579834048,
      cat_OB_GYN: -1.3456464731371562,
      cat_Pediatrics: 0.1348619774462137,
      region_dropout: 2.117005899199757,
    },
    hospitals: ['City/District', 'Regional', 'Republican'],
    categories: ['General Ward', 'ICU', 'ER', 'Psych', 'OB-GYN', 'Pediatrics', 'Other'],
    hasRegion: true,
    baselineDropout: 0.59,
    // Quartil-Schwellen (P25/P50/P75 ueber alle ~3.2k UZ-Kandidaten)
    tiers: [0.492, 0.605, 0.692],
    countryFilter: ['Uzbekistan'],
  },
  Colombia: {
    intercept: -7.435911361237017,
    features: {
      age: 0.4355943775097663,
      age_sq: -0.004803381419454712,
      is_male: 0.17737077044112393,
      is_married: -0.19075779215148392,
      is_single: -0.2254199115623465,
      has_icu: -0.5037186103515986,
      years_exp: -0.13364707404016132,
      years_exp_sq: 0.002913016247678206,
      'cat_General Ward': -0.48584220707890174,
      cat_ICU: -0.34842391160209873,
    },
    hospitals: [],
    categories: ['General Ward', 'ICU', 'Other'],
    hasRegion: false,
    baselineDropout: 0.68,
    // Quartil-Schwellen (P25/P50/P75 ueber alle ~800 CO-Kandidaten)
    tiers: [0.506, 0.577, 0.650],
    countryFilter: ['Colombia'],
  },
};

// ═══════════════════════════════════════════════════════════
//  REGION LOOKUP + CLASSIFICATION
// ═══════════════════════════════════════════════════════════

const REGIONS = {
  Philippines: {
    'NCR': 0.355525965379494,
    'Central Luzon': 0.4186746987951807,
    'CALABARZON': 0.39490445859872614,
    'Western Visayas': 0.392226148409894,
    'CAR': 0.41379310344827586,
    'Ilocos': 0.32547169811320753,
    'Zamboanga': 0.391304347826087,
    'Bicol': 0.3611111111111111,
    'Cagayan Valley': 0.37755102040816324,
    'Davao': 0.5357142857142857,
    'Central Visayas': 0.5,
    'Eastern Visayas': 0.4411764705882353,
    'Northern Mindanao': 0.4117647058823529,
    'Other': 0.2895662368112544,
  },
  Uzbekistan: {
    'Tashkent': 0.5602409638554217,
    'Fergana': 0.5730337078651685,
    'Andijan': 0.7074468085106383,
    'Namangan': 0.5510204081632653,
    'Samarkand': 0.6266666666666667,
    'Bukhara': 0.6666666666666666,
    'Kashkadarya': 0.6304347826086957,
    'Surkhandarya': 0.7368421052631579,
    'Khorezm': 0.5,
    'Navoi': 0.6153846153846154,
    'Jizzakh': 0.6666666666666666,
    'Sirdarya': 0.6666666666666666,
    'Karakalpakstan': 0.7894736842105263,
    'Other': 0.6190476190476191,
  },
  Colombia: {},
};

// PH city → region classification (keyword-based, same as training)
const PH_REGION_KEYWORDS = {
  'NCR': ['manila', 'makati', 'quezon city', 'pasig', 'taguig', 'mandaluyong', 'caloocan',
           'pasay', 'paranaque', 'muntinlupa', 'marikina', 'valenzuela', 'navotas', 'malabon',
           'san juan', 'pateros', 'las pinas', 'ncr', 'metro manila'],
  'Central Luzon': ['pampanga', 'bulacan', 'tarlac', 'nueva ecija', 'zambales', 'bataan', 'aurora',
                     'angeles', 'olongapo', 'san fernando', 'malolos', 'meycauayan', 'clark'],
  'CALABARZON': ['cavite', 'laguna', 'batangas', 'rizal', 'quezon province', 'calabarzon',
                  'antipolo', 'bacoor', 'imus', 'dasmarinas', 'lucena', 'lipa', 'calamba', 'san pablo',
                  'sta rosa', 'binan', 'cabuyao', 'santa rosa'],
  'Western Visayas': ['iloilo', 'bacolod', 'negros occidental', 'capiz', 'antique', 'aklan', 'guimaras'],
  'CAR': ['baguio', 'benguet', 'mountain province', 'ifugao', 'kalinga', 'apayao', 'abra', 'cordillera'],
  'Ilocos': ['pangasinan', 'la union', 'ilocos norte', 'ilocos sur', 'dagupan', 'san carlos', 'vigan', 'laoag'],
  'Zamboanga': ['zamboanga', 'pagadian', 'dipolog', 'dapitan', 'isabela city'],
  'Bicol': ['albay', 'camarines', 'sorsogon', 'catanduanes', 'masbate', 'legazpi', 'naga', 'bicol'],
  'Cagayan Valley': ['cagayan', 'isabela', 'nueva vizcaya', 'quirino', 'batanes', 'tuguegarao', 'santiago'],
  'Davao': ['davao', 'tagum', 'digos', 'panabo', 'samal'],
  'Central Visayas': ['cebu', 'bohol', 'siquijor', 'negros oriental', 'dumaguete', 'mandaue', 'lapu-lapu', 'talisay'],
  'Eastern Visayas': ['leyte', 'samar', 'tacloban', 'ormoc', 'eastern samar', 'northern samar', 'southern leyte', 'biliran'],
  'Northern Mindanao': ['misamis', 'bukidnon', 'lanao del norte', 'cagayan de oro', 'iligan', 'malaybalay', 'valencia'],
};

// UZ city → region classification
const UZ_REGION_KEYWORDS = {
  'Tashkent': ['tashkent', 'toshkent'],
  'Fergana': ['fergana', 'fargona', 'ferghana'],
  'Andijan': ['andijan', 'andijon'],
  'Namangan': ['namangan'],
  'Samarkand': ['samarkand', 'samarqand'],
  'Bukhara': ['bukhara', 'buxoro'],
  'Kashkadarya': ['kashkadarya', 'qashqadaryo', 'karshi'],
  'Surkhandarya': ['surkhandarya', 'surxondaryo', 'termez'],
  'Khorezm': ['khorezm', 'xorazm', 'urgench'],
  'Navoi': ['navoi', 'navoiy'],
  'Jizzakh': ['jizzakh', 'jizzax'],
  'Sirdarya': ['sirdarya', 'syrdarya', 'guliston'],
  'Karakalpakstan': ['karakalpakstan', 'nukus', 'qoraqalpog'],
};

function classifyRegionPH(city) {
  if (!city) return 'Other';
  const c = city.toLowerCase().trim();
  for (const [region, keywords] of Object.entries(PH_REGION_KEYWORDS)) {
    if (keywords.some(kw => c.includes(kw))) return region;
  }
  return 'Other';
}

function classifyRegionUZ(city) {
  if (!city) return 'Other';
  const c = city.toLowerCase().trim();
  for (const [region, keywords] of Object.entries(UZ_REGION_KEYWORDS)) {
    if (keywords.some(kw => c.includes(kw))) return region;
  }
  return 'Other';
}

function classifyRegion(city, country) {
  if (country === 'Philippines') return classifyRegionPH(city);
  if (country === 'Uzbekistan') return classifyRegionUZ(city);
  return '';
}

// ═══════════════════════════════════════════════════════════
//  CATEGORY SIMPLIFICATION (same as training)
// ═══════════════════════════════════════════════════════════

function simplifyCategory(cat) {
  if (!cat) return 'Other';
  const c = cat.toLowerCase().trim();
  if (c.includes('intensive') || c.includes('icu')) return 'ICU';
  if (c.includes('emergency') || c.includes('er ') || c === 'er') return 'ER';
  if (c.includes('operating') || c.includes('theater') || c.includes('theatre')) return 'OR';
  if (c.includes('obstetric') || c.includes('labor') || c.includes('gynaecol') || c.includes('gynecol')) return 'OB-GYN';
  if (c.includes('pediatric') || c.includes('paediatric') || c.includes('neonatal')) return 'Pediatrics';
  if (c.includes('psychiatric') || c.includes('mental')) return 'Psych';
  if (c.includes('dialysis') || c.includes('nephrol')) return 'Dialysis';
  if (c.includes('surgical') || c.includes('surgery')) return 'Med-Surg';
  if (c.includes('geriatric')) return 'Geriatrics';
  if (c.includes('general ward') || c.includes('general nursing') || c.includes('medical ward')) return 'General Ward';
  if (c.includes('cardiol')) return 'Other';
  return 'Other';
}

// ═══════════════════════════════════════════════════════════
//  HOSPITAL NORMALIZATION
// ═══════════════════════════════════════════════════════════

function normalizeHospital(val) {
  if (!val) return '';
  const v = val.toLowerCase().trim();
  if (v === 'tertiary') return 'Tertiary';
  if (v === 'secondary') return 'Secondary';
  if (v === 'primary') return 'Primary';
  if (v === 'regional') return 'Regional';
  if (v === 'republican') return 'Republican';
  if (v.includes('city') || v.includes('district')) return 'City/District';
  return val;
}

// ═══════════════════════════════════════════════════════════
//  FEATURE LABELS (German, for UI)
// ═══════════════════════════════════════════════════════════

const FEATURE_LABELS = {
  age: 'Alter', age_sq: 'Alter (quadr.)', is_male: 'Maennlich',
  is_married: 'Verheiratet', is_single: 'Ledig', has_icu: 'ICU-Erfahrung',
  years_exp: 'Berufsjahre', years_exp_sq: 'Berufsjahre (quadr.)',
  hosp_Tertiary: 'Tertiaerkrankenhaus', hosp_Primary: 'Primaerversorgung',
  hosp_Regional: 'Regionalkrankenhaus', hosp_Republican: 'Republikkrankenhaus',
  hosp_City_District: 'Stadt-/Bezirkskrankenhaus',
  'cat_General Ward': 'Allgemeinstation', cat_ICU: 'Intensivstation',
  cat_Med_Surg: 'Chirurgie', cat_OR: 'OP-Saal', cat_ER: 'Notaufnahme',
  cat_Geriatrics: 'Geriatrie', cat_Pediatrics: 'Paediatrie',
  cat_OB_GYN: 'Geburtshilfe/Gynaekologie', cat_Dialysis: 'Dialyse',
  cat_Psych: 'Psychiatrie', cat_Other: 'Andere Fachrichtung',
  region_dropout: 'Regionale Dropout-Rate',
};

// ═══════════════════════════════════════════════════════════
//  TIER CONFIGURATION
// ═══════════════════════════════════════════════════════════

const TIERS = {
  LOW:      { label: 'Low',      css: 'low' },
  MEDIUM:   { label: 'Medium',   css: 'medium' },
  ELEVATED: { label: 'Elevated', css: 'elevated' },
  HIGH:     { label: 'High',     css: 'high' },
};

function getTier(p, country) {
  const t = MODELS[country]?.tiers || [0.50, 0.60, 0.70];
  if (p < t[0]) return 'LOW';
  if (p < t[1]) return 'MEDIUM';
  if (p < t[2]) return 'ELEVATED';
  return 'HIGH';
}

// ═══════════════════════════════════════════════════════════
//  SCORING ENGINE
// ═══════════════════════════════════════════════════════════

function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

function scoreCandidate(country, candidate) {
  const model = MODELS[country];
  if (!model) return null;

  const age = candidate.age || 30;
  const featureValues = {
    age, age_sq: age * age,
    is_male: candidate.gender === 'Male' ? 1 : 0,
    is_married: candidate.maritalStatus === 'Married' ? 1 : 0,
    is_single: candidate.maritalStatus === 'Single' ? 1 : 0,
    has_icu: candidate.icuExperience ? 1 : 0,
    years_exp: candidate.yearsExperience || 0,
    years_exp_sq: (candidate.yearsExperience || 0) ** 2,
    region_dropout: model.hasRegion && candidate.region
      ? (REGIONS[country]?.[candidate.region] ?? REGIONS[country]?.['Other'] ?? 0)
      : 0,
  };

  // Hospital dummies
  const h = candidate.hospital || '';
  if (h === 'Tertiary') featureValues.hosp_Tertiary = 1;
  if (h === 'Primary') featureValues.hosp_Primary = 1;
  if (h === 'Regional') featureValues.hosp_Regional = 1;
  if (h === 'Republican') featureValues.hosp_Republican = 1;
  if (h === 'City/District') featureValues.hosp_City_District = 1;

  // Category dummies
  const cat = candidate.category || '';
  if (cat === 'General Ward') featureValues['cat_General Ward'] = 1;
  else if (cat) featureValues['cat_' + cat.replace(/-/g, '_')] = 1;

  let logit = model.intercept;
  const contributions = {};
  for (const [feature, coef] of Object.entries(model.features)) {
    const val = featureValues[feature] || 0;
    const contribution = coef * val;
    logit += contribution;
    if (Math.abs(contribution) > 0.001) {
      contributions[feature] = { coef, value: val, contribution };
    }
  }

  const probability = sigmoid(logit);
  return { probability, tier: getTier(probability, country), logit, contributions };
}

// ═══════════════════════════════════════════════════════════
//  API DATA + PERSON MAPPING
// ═══════════════════════════════════════════════════════════

const PROXY_URL = '/.netlify/functions/api-proxy';
const personsCache = {}; // { country: Person[] }
let allPersonsRaw = null; // raw API response cached

async function fetchPersons() {
  if (allPersonsRaw) return allPersonsRaw;

  const statusEl = document.getElementById('loading-status');
  statusEl.style.display = '';
  statusEl.textContent = 'Lade Kandidaten aus der API...';

  try {
    const url = `${PROXY_URL}?endpoint=${encodeURIComponent('/persons?state=all')}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`API Fehler: ${res.status}`);
    const json = await res.json();

    // API returns { original: { "0": {...}, "1": {...} } } or similar
    let persons;
    if (json.original) {
      persons = Array.isArray(json.original) ? json.original : Object.values(json.original);
    } else if (Array.isArray(json)) {
      persons = json;
    } else {
      persons = Object.values(json);
    }

    allPersonsRaw = persons;
    statusEl.style.display = 'none';
    return persons;
  } catch (err) {
    statusEl.textContent = `Fehler beim Laden: ${err.message}`;
    statusEl.className = 'loading-status error';
    throw err;
  }
}

async function getPersonsForCountry(country) {
  if (personsCache[country]) return personsCache[country];

  const raw = await fetchPersons();
  const countryFilter = MODELS[country].countryFilter;

  const mapped = raw
    .filter(p => countryFilter.includes(p.country))
    .map(p => mapPersonFromApi(p, country));

  personsCache[country] = mapped;
  return mapped;
}

function mapPersonFromApi(p, country) {
  // Calculate age
  let age = null;
  if (p.person_birth_date) {
    const birth = new Date(p.person_birth_date);
    const now = new Date();
    age = Math.floor((now - birth) / (365.25 * 24 * 60 * 60 * 1000));
    if (age < 18 || age > 70) age = null;
  }

  // Classify region from city
  const region = classifyRegion(p.person_city || p.person_birth_place || '', country);

  // ICU experience
  const icuRaw = String(p.person_icu_category || '').toLowerCase();
  const icuExperience = ['1', '1.0', 'true', 'yes'].includes(icuRaw);

  return {
    personId: p.person_id,
    referenceId: p.reference_id || '',
    name: p.person_name || '(Unbekannt)',
    country: p.country,
    age,
    gender: p.person_gender || 'Female',
    maritalStatus: p.marital_status || '',
    hospital: normalizeHospital(p.person_hospital || ''),
    category: simplifyCategory(p.person_categories || ''),
    rawCategory: p.person_categories || '',
    icuExperience,
    yearsExperience: parseFloat(p.total_years_experience_rn) || 0,
    region,
    city: p.person_city || '',
    // Status info for display
    processStep: p.process_step || '',
    classId: p.class_id,
    hasDropped: !!(p.dropout_date_fin || p.dropout_reason),
    hasArrived: !!p.arrival_fin,
  };
}

// ═══════════════════════════════════════════════════════════
//  APPLICATION STATE
// ═══════════════════════════════════════════════════════════

let state = {
  country: 'Philippines',
  candidates: [],      // scored lineup candidates
  filter: 'ALL',
  nextId: 1,
  persons: [],         // loaded from API for current country
  loading: false,
  searchOpen: false,
};

// ═══════════════════════════════════════════════════════════
//  UI CONTROLLER
// ═══════════════════════════════════════════════════════════

function init() {
  renderCountryTabs();
  renderResults();
  renderSummary();

  // Event listeners
  document.getElementById('csv-export-btn').addEventListener('click', exportCsv);
  document.getElementById('clear-all-btn').addEventListener('click', clearAll);

  // Training-Data-Export
  document.getElementById('export-btn').addEventListener('click', downloadTrainingCsv);
  document.getElementById('export-subset').addEventListener('change', updateExportCount);
  updateExportCount();

  // Search input
  const searchInput = document.getElementById('search-input');
  searchInput.addEventListener('input', onSearchInput);
  searchInput.addEventListener('focus', () => {
    if (searchInput.value.length >= 2) onSearchInput();
  });
  document.addEventListener('click', (e) => {
    if (!e.target.closest('.search-container')) closeDropdown();
  });
  searchInput.addEventListener('keydown', onSearchKeydown);

  // Load data for initial country
  loadCountryData();
}

async function loadCountryData() {
  state.loading = true;
  updateSearchPlaceholder();
  try {
    state.persons = await getPersonsForCountry(state.country);
    state.loading = false;
    updateSearchPlaceholder();
    updatePersonCount();
    updateExportCount();
  } catch (e) {
    state.loading = false;
    updateSearchPlaceholder();
  }
}

function updateSearchPlaceholder() {
  const input = document.getElementById('search-input');
  if (state.loading) {
    input.placeholder = 'Lade Kandidaten...';
    input.disabled = true;
  } else {
    input.placeholder = `Name oder Referenz-Nr. eingeben (${state.persons.length} Kandidaten)`;
    input.disabled = false;
  }
}

function updatePersonCount() {
  const el = document.getElementById('person-count');
  if (el) {
    const active = state.persons.filter(p => !p.hasDropped && !p.hasArrived).length;
    el.textContent = `${state.persons.length} Personen geladen, ${active} aktiv in Pipeline`;
  }
}

function renderCountryTabs() {
  const container = document.getElementById('country-tabs');
  container.innerHTML = '';
  for (const country of Object.keys(MODELS)) {
    const btn = document.createElement('button');
    btn.className = 'country-tab' + (country === state.country ? ' active' : '');
    btn.textContent = country;
    btn.addEventListener('click', () => {
      if (country === state.country) return;
      state.country = country;
      state.candidates = [];
      state.filter = 'ALL';
      state.nextId = 1;
      renderCountryTabs();
      renderResults();
      renderSummary();
      document.getElementById('search-input').value = '';
      closeDropdown();
      loadCountryData();
    });
    container.appendChild(btn);
  }
}

// ═══════════════════════════════════════════════════════════
//  SEARCH / AUTOCOMPLETE
// ═══════════════════════════════════════════════════════════

let searchHighlightIdx = -1;

function onSearchInput() {
  const query = document.getElementById('search-input').value.trim().toLowerCase();
  if (query.length < 2) { closeDropdown(); return; }

  const alreadyAdded = new Set(state.candidates.map(c => c.personId));

  const results = state.persons
    .filter(p => {
      if (alreadyAdded.has(p.personId)) return false;
      const nameMatch = p.name.toLowerCase().includes(query);
      const refMatch = p.referenceId && p.referenceId.toString().includes(query);
      return nameMatch || refMatch;
    })
    .slice(0, 20);

  renderDropdown(results, query);
}

function onSearchKeydown(e) {
  const dropdown = document.getElementById('search-dropdown');
  const items = dropdown.querySelectorAll('.dropdown-item');
  if (!items.length) return;

  if (e.key === 'ArrowDown') {
    e.preventDefault();
    searchHighlightIdx = Math.min(searchHighlightIdx + 1, items.length - 1);
    updateDropdownHighlight(items);
  } else if (e.key === 'ArrowUp') {
    e.preventDefault();
    searchHighlightIdx = Math.max(searchHighlightIdx - 1, 0);
    updateDropdownHighlight(items);
  } else if (e.key === 'Enter' && searchHighlightIdx >= 0) {
    e.preventDefault();
    items[searchHighlightIdx].click();
  } else if (e.key === 'Escape') {
    closeDropdown();
  }
}

function updateDropdownHighlight(items) {
  items.forEach((item, i) => {
    item.classList.toggle('highlighted', i === searchHighlightIdx);
  });
  if (searchHighlightIdx >= 0 && items[searchHighlightIdx]) {
    items[searchHighlightIdx].scrollIntoView({ block: 'nearest' });
  }
}

function renderDropdown(results, query) {
  const dropdown = document.getElementById('search-dropdown');
  searchHighlightIdx = -1;

  if (results.length === 0) {
    dropdown.innerHTML = '<div class="dropdown-empty">Keine Treffer</div>';
    dropdown.style.display = '';
    return;
  }

  dropdown.innerHTML = '';
  dropdown.style.display = '';

  for (const p of results) {
    const item = document.createElement('div');
    item.className = 'dropdown-item';

    const statusClass = p.hasDropped ? 'dropped' : p.hasArrived ? 'arrived' : 'active';
    const statusLabel = p.hasDropped ? 'Dropout' : p.hasArrived ? 'Arrived' : (p.processStep || 'Aktiv');

    item.innerHTML = `
      <div class="dropdown-item-main">
        <span class="dropdown-name">${highlightMatch(esc(p.name), query)}</span>
        <span class="dropdown-ref">${p.referenceId ? '#' + esc(p.referenceId) : ''}</span>
      </div>
      <div class="dropdown-item-meta">
        ${p.age ? p.age + 'J' : '?'} &middot;
        ${p.gender === 'Male' ? 'M' : 'W'} &middot;
        ${esc(p.category || '-')} &middot;
        ${esc(p.hospital || '-')} &middot;
        ${p.yearsExperience ? p.yearsExperience + ' Jahre' : '-'}
        <span class="dropdown-status ${statusClass}">${esc(statusLabel)}</span>
      </div>
    `;

    item.addEventListener('click', () => addPersonToLineup(p));
    dropdown.appendChild(item);
  }
}

function highlightMatch(text, query) {
  if (!query) return text;
  const regex = new RegExp(`(${query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
  return text.replace(regex, '<mark>$1</mark>');
}

function closeDropdown() {
  document.getElementById('search-dropdown').style.display = 'none';
  searchHighlightIdx = -1;
}

// ═══════════════════════════════════════════════════════════
//  ADD PERSON TO LINEUP
// ═══════════════════════════════════════════════════════════

function addPersonToLineup(person) {
  const candidate = {
    id: state.nextId++,
    personId: person.personId,
    referenceId: person.referenceId,
    name: person.name,
    age: person.age,
    gender: person.gender,
    maritalStatus: person.maritalStatus,
    hospital: person.hospital,
    category: person.category,
    rawCategory: person.rawCategory,
    icuExperience: person.icuExperience,
    yearsExperience: person.yearsExperience,
    region: person.region,
    city: person.city,
  };

  const result = scoreCandidate(state.country, candidate);
  candidate.score = result.probability;
  candidate.tier = result.tier;
  candidate.contributions = result.contributions;

  state.candidates.push(candidate);

  // Clear search
  document.getElementById('search-input').value = '';
  closeDropdown();

  renderResults();
  renderSummary();

  // Focus back on search for quick successive adds
  document.getElementById('search-input').focus();
}

function removeCandidate(id) {
  state.candidates = state.candidates.filter(c => c.id !== id);
  renderResults();
  renderSummary();
}

function clearAll() {
  if (state.candidates.length === 0) return;
  if (!confirm(`Alle ${state.candidates.length} Kandidaten entfernen?`)) return;
  state.candidates = [];
  state.nextId = 1;
  renderResults();
  renderSummary();
}

function toggleFactors(id) {
  const row = document.getElementById(`factors-${id}`);
  if (row) row.classList.toggle('open');
}

// ═══════════════════════════════════════════════════════════
//  RENDER: Summary Panel
// ═══════════════════════════════════════════════════════════

function renderSummary() {
  const panel = document.getElementById('summary-panel');
  if (state.candidates.length === 0) { panel.style.display = 'none'; return; }

  panel.style.display = '';
  const n = state.candidates.length;
  const expectedArrivals = state.candidates.reduce((sum, c) => sum + (1 - c.score), 0);
  const expectedDropouts = n - expectedArrivals;
  const avgProb = state.candidates.reduce((sum, c) => sum + c.score, 0) / n;
  const survivalRate = expectedArrivals / n;
  const overRecruit = Math.max(0, Math.ceil(n / survivalRate) - n);

  const tiers = { LOW: 0, MEDIUM: 0, ELEVATED: 0, HIGH: 0 };
  state.candidates.forEach(c => tiers[c.tier]++);

  const baseline = MODELS[state.country].baselineDropout;

  document.getElementById('summary-total').textContent = n;
  document.getElementById('summary-arrivals').textContent = expectedArrivals.toFixed(1);
  document.getElementById('summary-dropouts').textContent = expectedDropouts.toFixed(1);
  document.getElementById('summary-overrecruit').textContent = '+' + overRecruit;

  document.getElementById('tier-low-count').textContent = tiers.LOW;
  document.getElementById('tier-medium-count').textContent = tiers.MEDIUM;
  document.getElementById('tier-elevated-count').textContent = tiers.ELEVATED;
  document.getElementById('tier-high-count').textContent = tiers.HIGH;

  document.getElementById('summary-avg-prob').textContent = (avgProb * 100).toFixed(1) + '%';
  document.getElementById('summary-baseline').textContent = state.country + ': ' + (baseline * 100).toFixed(0) + '%';
  document.getElementById('summary-survival').textContent = (survivalRate * 100).toFixed(0) + '%';

  const recEl = document.getElementById('summary-recommendation');
  if (overRecruit > 0) {
    recEl.style.display = '';
    recEl.textContent = `Empfehlung: ${overRecruit} zusaetzliche Kandidaten rekrutieren, um die Ziel-Arrivals zu sichern.`;
  } else {
    recEl.style.display = 'none';
  }
}

// ═══════════════════════════════════════════════════════════
//  RENDER: Filter Bar + Results Table
// ═══════════════════════════════════════════════════════════

function renderResults() {
  renderFilterBar();
  renderTable();
}

function renderFilterBar() {
  const bar = document.getElementById('filter-bar');
  const tiers = { ALL: state.candidates.length, LOW: 0, MEDIUM: 0, ELEVATED: 0, HIGH: 0 };
  state.candidates.forEach(c => tiers[c.tier]++);

  bar.querySelectorAll('.filter-btn').forEach(btn => {
    const f = btn.dataset.filter;
    btn.className = 'filter-btn' + (f === state.filter ? ' active' : '');
    const countEl = btn.querySelector('.count');
    if (countEl) countEl.textContent = tiers[f] || 0;
  });
}

function setFilter(filter) {
  state.filter = filter;
  renderResults();
}

function renderTable() {
  const tbody = document.getElementById('results-body');
  const empty = document.getElementById('empty-state');
  const tableWrap = document.getElementById('results-table-wrap');

  const filtered = state.filter === 'ALL'
    ? state.candidates
    : state.candidates.filter(c => c.tier === state.filter);

  if (filtered.length === 0) {
    tableWrap.style.display = 'none';
    empty.style.display = '';
    return;
  }

  tableWrap.style.display = '';
  empty.style.display = 'none';

  const sorted = [...filtered].sort((a, b) => b.score - a.score);
  tbody.innerHTML = '';

  for (const c of sorted) {
    const tierCfg = TIERS[c.tier];
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td class="col-name">${esc(c.name)}${c.referenceId ? ' <span class="col-ref">#' + esc(c.referenceId) + '</span>' : ''}</td>
      <td>${c.age || '-'}</td>
      <td>${c.gender === 'Male' ? 'M' : 'W'}</td>
      <td>${esc(c.category || '-')}</td>
      <td>${esc(c.hospital || '-')}</td>
      <td>${c.yearsExperience || '-'}</td>
      <td>${esc(c.region || '-')}</td>
      <td class="col-score" style="color: var(--color-${tierCfg.css})">${(c.score * 100).toFixed(1)}%</td>
      <td><span class="tier-badge ${tierCfg.css}">${tierCfg.label}</span></td>
      <td>
        <button class="factor-toggle" onclick="toggleFactors(${c.id})">
          Faktoren <span class="arrow">&#9660;</span>
        </button>
      </td>
      <td>
        <button class="delete-btn" onclick="removeCandidate(${c.id})" title="Entfernen">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/></svg>
        </button>
      </td>
    `;
    tbody.appendChild(tr);

    const ftr = document.createElement('tr');
    ftr.className = 'factor-row';
    ftr.id = `factors-${c.id}`;
    const ftd = document.createElement('td');
    ftd.colSpan = 11;
    ftd.innerHTML = renderFactorChips(c.contributions);
    ftr.appendChild(ftd);
    tbody.appendChild(ftr);
  }
}

function renderFactorChips(contributions) {
  if (!contributions || Object.keys(contributions).length === 0) {
    return '<span style="color:var(--color-text-light);font-size:12px;">Keine signifikanten Faktoren</span>';
  }

  const sorted = Object.entries(contributions)
    .sort((a, b) => Math.abs(b[1].contribution) - Math.abs(a[1].contribution));

  let html = '<div class="factor-list">';
  for (const [feature, data] of sorted) {
    const label = FEATURE_LABELS[feature] || feature;
    const isPositive = data.contribution > 0;
    const cls = isPositive ? 'positive' : 'negative';
    const arrow = isPositive ? '&#9650;' : '&#9660;';
    const sign = isPositive ? '+' : '';
    html += `<span class="factor-chip ${cls}">
      <span class="arrow">${arrow}</span>
      ${esc(label)} (${sign}${data.contribution.toFixed(2)})
    </span>`;
  }
  html += '</div>';
  return html;
}

function esc(str) {
  const div = document.createElement('div');
  div.textContent = String(str);
  return div.innerHTML;
}

// ═══════════════════════════════════════════════════════════
//  CSV EXPORT
// ═══════════════════════════════════════════════════════════

function exportCsv() {
  if (state.candidates.length === 0) { alert('Keine Kandidaten zum Exportieren.'); return; }

  const sep = ';';
  const headers = ['Referenz', 'Name', 'Alter', 'Geschlecht', 'Familienstand', 'Krankenhaus',
    'Fachrichtung', 'ICU', 'Berufsjahre', 'Region', 'Score (%)', 'Tier', 'Empfehlung'];

  const rows = state.candidates.map(c => {
    let empfehlung = '';
    if (c.tier === 'HIGH') empfehlung = 'Hohes Risiko - genau pruefen';
    else if (c.tier === 'ELEVATED') empfehlung = 'Erhoehtes Risiko - beobachten';
    else if (c.tier === 'MEDIUM') empfehlung = 'Mittleres Risiko';
    else empfehlung = 'Niedriges Risiko';

    return [
      c.referenceId || '', c.name, c.age || '', c.gender === 'Male' ? 'M' : 'W',
      c.maritalStatus || '', c.hospital || '', c.category || '',
      c.icuExperience ? 'Ja' : 'Nein', c.yearsExperience || '', c.region || '',
      (c.score * 100).toFixed(1).replace('.', ','), TIERS[c.tier].label, empfehlung,
    ].map(v => `"${v}"`).join(sep);
  });

  const bom = '\uFEFF';
  const csv = bom + headers.join(sep) + '\n' + rows.join('\n');
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `lineup-screener-${state.country.toLowerCase()}-${new Date().toISOString().slice(0, 10)}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

// ═══════════════════════════════════════════════════════════
//  TRAINING DATA EXPORT (fuer Modell-Retraining)
// ═══════════════════════════════════════════════════════════

/**
 * Liefert eine gefilterte Liste von rohen API-Personen-Records fuer das aktuelle Land.
 * Subset-Filter:
 *   'with_outcome' → arrival_fin ODER dropout_date_fin
 *   'arrived'      → arrival_fin
 *   'dropped'      → dropout_date_fin
 *   'all'          → alle Kandidaten des Landes
 */
function filterPersonsForExport(rawPersons, country, subset) {
  const filtered = rawPersons.filter(p => p.country === country);
  if (subset === 'arrived')       return filtered.filter(p => p.arrival_fin);
  if (subset === 'dropped')       return filtered.filter(p => p.dropout_date_fin);
  if (subset === 'with_outcome')  return filtered.filter(p => p.arrival_fin || p.dropout_date_fin);
  return filtered;
}

function yearsBetween(dateStrA, dateStrB) {
  if (!dateStrA || !dateStrB) return '';
  const a = new Date(dateStrA);
  const b = new Date(dateStrB);
  if (isNaN(a) || isNaN(b)) return '';
  return ((b - a) / (365.25 * 24 * 60 * 60 * 1000)).toFixed(1);
}

function csvEscape(v) {
  if (v === null || v === undefined) return '';
  const s = String(v);
  if (/[;"\n\r]/.test(s)) return '"' + s.replace(/"/g, '""') + '"';
  return s;
}

function buildTrainingCsv(country, subset, options) {
  if (!allPersonsRaw) return null;
  const persons = filterPersonsForExport(allPersonsRaw, country, subset);

  const columns = [
    'person_id', 'reference_id', 'name', 'gender',
    'birth_date', 'age_at_event',
    'marital_status',
  ];
  if (options.kidsCount) columns.push('kids_count');
  columns.push(
    'years_experience',
    'origin_city', 'current_city', 'region',
    'hospital_type', 'category', 'icu_category',
    'arrival_date', 'dropout_date', 'dropout_reason',
    'outcome', 'v4_predicted_prob',
  );
  if (options.notes) columns.push('notes');

  const rows = persons.map(p => {
    // Outcome klassifizieren
    let outcome;
    if (p.arrival_fin && p.dropout_date_fin)      outcome = 'dropped_after_arrival';
    else if (p.arrival_fin)                       outcome = 'arrived';
    else if (p.dropout_date_fin)                  outcome = 'dropped_before_arrival';
    else                                           outcome = 'active';

    // Event-Datum fuer Alter
    const eventDate = p.arrival_fin || p.dropout_date_fin || new Date().toISOString().slice(0, 10);
    const age = yearsBetween(p.person_birth_date, eventDate);

    // Region — origin bevorzugen (birth_place), current_city als Fallback
    const currentCity = p.person_city || '';
    const birthPlace = p.person_birth_place || '';
    const regionFromBirth = classifyRegion(birthPlace, country);
    const regionFromCity = classifyRegion(currentCity, country);
    const region = regionFromBirth !== 'Other' ? regionFromBirth : regionFromCity;
    const originCity = birthPlace || currentCity;

    // v4 Score fuer diesen Kandidaten
    const candidate = mapPersonFromApi(p, country);
    const scoreResult = scoreCandidate(country, candidate);
    const v4Prob = scoreResult ? scoreResult.probability.toFixed(4) : '';

    const row = {
      person_id: p.person_id ?? '',
      reference_id: p.reference_id ?? '',
      name: p.person_name ?? '',
      gender: p.person_gender ?? '',
      birth_date: (p.person_birth_date || '').slice(0, 10),
      age_at_event: age,
      marital_status: p.marital_status ?? '',
      kids_count: '',  // leer, manuell zu fuellen
      years_experience: p.total_years_experience_rn ?? '',
      origin_city: originCity,
      current_city: currentCity,
      region,
      hospital_type: normalizeHospital(p.person_hospital || ''),
      category: simplifyCategory(p.person_categories || ''),
      icu_category: p.person_icu_category ?? '',
      arrival_date: (p.arrival_fin || '').slice(0, 10),
      dropout_date: (p.dropout_date_fin || '').slice(0, 10),
      dropout_reason: p.dropout_reason || '',
      outcome,
      v4_predicted_prob: v4Prob,
      notes: '',
    };
    return columns.map(c => csvEscape(row[c])).join(';');
  });

  const bom = '\uFEFF';
  return {
    csv: bom + columns.join(';') + '\n' + rows.join('\n'),
    count: persons.length,
  };
}

async function downloadTrainingCsv() {
  // Sicherstellen, dass die API-Daten geladen sind
  if (!allPersonsRaw) {
    try { await fetchPersons(); } catch (e) { alert('API-Daten konnten nicht geladen werden: ' + e.message); return; }
  }

  const subset = document.getElementById('export-subset').value;
  const options = {
    kidsCount: document.getElementById('export-opt-kids').checked,
    notes:     document.getElementById('export-opt-notes').checked,
  };

  const result = buildTrainingCsv(state.country, subset, options);
  if (!result || result.count === 0) { alert('Keine Kandidaten fuer diesen Filter.'); return; }

  const today = new Date().toISOString().slice(0, 10);
  const subsetSlug = subset === 'with_outcome' ? 'outcomes' : subset;
  const filename = `${state.country.toLowerCase()}_dropout_dataset_${subsetSlug}_${today}.csv`;

  const blob = new Blob([result.csv], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename; a.click();
  URL.revokeObjectURL(url);
}

async function updateExportCount() {
  const countEl = document.getElementById('export-count');
  if (!countEl) return;

  if (!allPersonsRaw) {
    countEl.textContent = 'Daten werden geladen...';
    return;
  }

  const subset = document.getElementById('export-subset').value;
  const persons = filterPersonsForExport(allPersonsRaw, state.country, subset);
  countEl.textContent = `${persons.length.toLocaleString('de-DE')} Kandidaten (${state.country})`;
}

// ═══════════════════════════════════════════════════════════
//  INIT
// ═══════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', init);
