/**
 * GLOBOGATE Lineup Screener — v4 Dropout Risk Scoring
 * Client-side Logistic Regression: P = 1 / (1 + exp(-logit))
 * Trained on 4,906 resolved pipeline journeys (no recruiter, no school).
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
  },
};

// ═══════════════════════════════════════════════════════════
//  REGION LOOKUP TABLES
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

// ═══════════════════════════════════════════════════════════
//  FEATURE LABELS (German, for UI)
// ═══════════════════════════════════════════════════════════

const FEATURE_LABELS = {
  age: 'Alter',
  age_sq: 'Alter (quadr.)',
  is_male: 'Maennlich',
  is_married: 'Verheiratet',
  is_single: 'Ledig',
  has_icu: 'ICU-Erfahrung',
  years_exp: 'Berufsjahre',
  years_exp_sq: 'Berufsjahre (quadr.)',
  hosp_Tertiary: 'Tertiaerkrankenhaus',
  hosp_Primary: 'Primaerversorgung',
  hosp_Secondary: 'Sekundaerversorgung',
  hosp_Regional: 'Regionalkrankenhaus',
  hosp_Republican: 'Republikkrankenhaus',
  hosp_City_District: 'Stadt-/Bezirkskrankenhaus',
  'cat_General Ward': 'Allgemeinstation',
  cat_ICU: 'Intensivstation',
  cat_Med_Surg: 'Chirurgie',
  cat_OR: 'OP-Saal',
  cat_ER: 'Notaufnahme',
  cat_Geriatrics: 'Geriatrie',
  cat_Pediatrics: 'Paediatrie',
  cat_OB_GYN: 'Geburtshilfe/Gynaekologie',
  cat_Dialysis: 'Dialyse',
  cat_Psych: 'Psychiatrie',
  cat_Other: 'Andere Fachrichtung',
  region_dropout: 'Regionale Dropout-Rate',
};

// ═══════════════════════════════════════════════════════════
//  TIER CONFIGURATION
// ═══════════════════════════════════════════════════════════

const TIERS = {
  LOW:      { label: 'Low',      min: 0,    max: 0.30, css: 'low' },
  MEDIUM:   { label: 'Medium',   min: 0.30, max: 0.50, css: 'medium' },
  ELEVATED: { label: 'Elevated', min: 0.50, max: 0.70, css: 'elevated' },
  HIGH:     { label: 'High',     min: 0.70, max: 1.01, css: 'high' },
};

function getTier(probability) {
  if (probability < 0.30) return 'LOW';
  if (probability < 0.50) return 'MEDIUM';
  if (probability < 0.70) return 'ELEVATED';
  return 'HIGH';
}

// ═══════════════════════════════════════════════════════════
//  SCORING ENGINE
// ═══════════════════════════════════════════════════════════

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function scoreCandidate(country, candidate) {
  const model = MODELS[country];
  if (!model) return null;

  const age = candidate.age || 30;
  const ageSq = age * age;
  const isMale = candidate.gender === 'Male' ? 1 : 0;
  const isMarried = candidate.maritalStatus === 'Married' ? 1 : 0;
  const isSingle = candidate.maritalStatus === 'Single' ? 1 : 0;
  const hasIcu = candidate.icuExperience ? 1 : 0;
  const yearsExp = candidate.yearsExperience || 0;
  const yearsExpSq = yearsExp * yearsExp;

  // Hospital dummies
  const hospital = candidate.hospital || '';
  const hospFeatures = {};
  if (hospital === 'Tertiary') hospFeatures.hosp_Tertiary = 1;
  if (hospital === 'Primary') hospFeatures.hosp_Primary = 1;
  if (hospital === 'Secondary') hospFeatures.hosp_Secondary = 1;
  if (hospital === 'Regional') hospFeatures.hosp_Regional = 1;
  if (hospital === 'Republican') hospFeatures.hosp_Republican = 1;
  if (hospital === 'City/District') hospFeatures.hosp_City_District = 1;

  // Category dummies
  const category = candidate.category || '';
  const catKey = 'cat_' + category.replace(/-/g, '_');
  const catFeatures = {};
  catFeatures[catKey] = 1;
  // Special case for "General Ward" with space
  if (category === 'General Ward') {
    delete catFeatures[catKey];
    catFeatures['cat_General Ward'] = 1;
  }

  // Region dropout rate
  const regionRate = model.hasRegion && candidate.region
    ? (REGIONS[country]?.[candidate.region] ?? REGIONS[country]?.['Other'] ?? 0)
    : 0;

  // Build feature vector
  const featureValues = {
    age, age_sq: ageSq, is_male: isMale,
    is_married: isMarried, is_single: isSingle,
    has_icu: hasIcu, years_exp: yearsExp, years_exp_sq: yearsExpSq,
    ...hospFeatures, ...catFeatures,
    region_dropout: regionRate,
  };

  // Calculate logit
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
  const tier = getTier(probability);

  return { probability, tier, logit, contributions };
}

// ═══════════════════════════════════════════════════════════
//  APPLICATION STATE
// ═══════════════════════════════════════════════════════════

let state = {
  country: 'Philippines',
  candidates: [],
  filter: 'ALL',
  nextId: 1,
};

// ═══════════════════════════════════════════════════════════
//  UI CONTROLLER
// ═══════════════════════════════════════════════════════════

function init() {
  renderCountryTabs();
  updateFormForCountry();
  renderResults();
  renderSummary();

  // Event listeners
  document.getElementById('add-btn').addEventListener('click', addCandidate);
  document.getElementById('csv-upload-btn').addEventListener('click', () => {
    document.getElementById('csv-file-input').click();
  });
  document.getElementById('csv-file-input').addEventListener('change', handleCsvUpload);
  document.getElementById('csv-export-btn').addEventListener('click', exportCsv);
  document.getElementById('clear-all-btn').addEventListener('click', clearAll);

  // Enter key in form
  document.getElementById('input-form').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); addCandidate(); }
  });
}

function renderCountryTabs() {
  const container = document.getElementById('country-tabs');
  container.innerHTML = '';
  for (const country of Object.keys(MODELS)) {
    const btn = document.createElement('button');
    btn.className = 'country-tab' + (country === state.country ? ' active' : '');
    btn.textContent = country;
    btn.addEventListener('click', () => {
      state.country = country;
      state.candidates = [];
      state.filter = 'ALL';
      state.nextId = 1;
      renderCountryTabs();
      updateFormForCountry();
      renderResults();
      renderSummary();
    });
    container.appendChild(btn);
  }
}

function updateFormForCountry() {
  const model = MODELS[state.country];

  // Hospital select
  const hospGroup = document.getElementById('hospital-group');
  const hospSelect = document.getElementById('input-hospital');
  if (model.hospitals.length === 0) {
    hospGroup.style.display = 'none';
    hospSelect.value = '';
  } else {
    hospGroup.style.display = '';
    hospSelect.innerHTML = '<option value="">-- Bitte waehlen --</option>';
    for (const h of model.hospitals) {
      const opt = document.createElement('option');
      opt.value = h;
      opt.textContent = h;
      hospSelect.appendChild(opt);
    }
  }

  // Category select
  const catSelect = document.getElementById('input-category');
  catSelect.innerHTML = '<option value="">-- Bitte waehlen --</option>';
  for (const c of model.categories) {
    const opt = document.createElement('option');
    opt.value = c;
    opt.textContent = c;
    catSelect.appendChild(opt);
  }

  // Region select
  const regionGroup = document.getElementById('region-group');
  const regionSelect = document.getElementById('input-region');
  if (model.hasRegion) {
    regionGroup.style.display = '';
    regionSelect.innerHTML = '<option value="">-- Bitte waehlen --</option>';
    const regions = Object.keys(REGIONS[state.country] || {});
    for (const r of regions) {
      const opt = document.createElement('option');
      opt.value = r;
      opt.textContent = r;
      regionSelect.appendChild(opt);
    }
  } else {
    regionGroup.style.display = 'none';
    regionSelect.value = '';
  }

  // Reset form
  document.getElementById('input-name').value = '';
  document.getElementById('input-age').value = '';
  document.getElementById('input-gender').value = 'Female';
  document.getElementById('input-marital').value = '';
  document.getElementById('input-icu').checked = false;
  document.getElementById('input-years').value = '';
}

function addCandidate() {
  const name = document.getElementById('input-name').value.trim() || `Kandidat ${state.nextId}`;
  const age = parseFloat(document.getElementById('input-age').value);
  const gender = document.getElementById('input-gender').value;
  const maritalStatus = document.getElementById('input-marital').value;
  const hospital = document.getElementById('input-hospital').value;
  const category = document.getElementById('input-category').value;
  const icuExperience = document.getElementById('input-icu').checked;
  const yearsExperience = parseFloat(document.getElementById('input-years').value) || 0;
  const region = document.getElementById('input-region').value;

  if (!age || age < 18 || age > 65) {
    alert('Bitte Alter eingeben (18-65).');
    return;
  }

  const candidate = {
    id: state.nextId++,
    name, age, gender, maritalStatus, hospital, category, icuExperience, yearsExperience, region,
  };

  const result = scoreCandidate(state.country, candidate);
  candidate.score = result.probability;
  candidate.tier = result.tier;
  candidate.contributions = result.contributions;

  state.candidates.push(candidate);

  // Clear form (keep country-specific selects)
  document.getElementById('input-name').value = '';
  document.getElementById('input-age').value = '';
  document.getElementById('input-icu').checked = false;
  document.getElementById('input-years').value = '';

  renderResults();
  renderSummary();
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

  if (state.candidates.length === 0) {
    panel.style.display = 'none';
    return;
  }

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

  const buttons = bar.querySelectorAll('.filter-btn');
  buttons.forEach(btn => {
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

  // Sort by score descending (highest risk first)
  const sorted = [...filtered].sort((a, b) => b.score - a.score);

  tbody.innerHTML = '';
  for (const c of sorted) {
    const tierCfg = TIERS[c.tier];

    // Main row
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td class="col-name">${esc(c.name)}</td>
      <td>${c.age}</td>
      <td>${c.gender === 'Male' ? 'M' : 'W'}</td>
      <td>${esc(c.maritalStatus || '-')}</td>
      <td>${esc(c.hospital || '-')}</td>
      <td>${esc(c.category || '-')}</td>
      <td>${c.yearsExperience || '-'}</td>
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

    // Factor detail row
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
  div.textContent = str;
  return div.innerHTML;
}

// ═══════════════════════════════════════════════════════════
//  CSV IMPORT
// ═══════════════════════════════════════════════════════════

function handleCsvUpload(e) {
  const file = e.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = function(evt) {
    const text = evt.target.result;
    parseCsv(text);
    e.target.value = '';
  };
  reader.readAsText(file, 'UTF-8');
}

function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/);
  if (lines.length < 2) {
    alert('CSV muss mindestens eine Headerzeile und eine Datenzeile enthalten.');
    return;
  }

  // Auto-detect delimiter
  const delimiter = lines[0].includes(';') ? ';' : ',';
  const headers = lines[0].split(delimiter).map(h => h.trim().toLowerCase().replace(/['"]/g, ''));

  // Flexible header mapping
  const colMap = {};
  const mappings = {
    name: ['name', 'kandidat', 'candidate'],
    age: ['alter', 'age', 'edad'],
    gender: ['geschlecht', 'gender', 'sexo', 'sex'],
    maritalStatus: ['familienstand', 'marital', 'marital_status', 'estado_civil'],
    hospital: ['krankenhaus', 'krankenhaustyp', 'hospital', 'hospital_type'],
    category: ['fachrichtung', 'category', 'kategorie', 'especialidad', 'nursing_category'],
    icu: ['icu', 'icu_erfahrung', 'icu_experience', 'intensivstation'],
    yearsExperience: ['berufsjahre', 'berufserfahrung', 'years_exp', 'years_experience', 'years', 'experiencia'],
    region: ['region', 'herkunft', 'origen'],
  };

  for (const [field, aliases] of Object.entries(mappings)) {
    const idx = headers.findIndex(h => aliases.some(a => h.includes(a)));
    if (idx >= 0) colMap[field] = idx;
  }

  if (colMap.age === undefined) {
    alert('CSV-Fehler: Spalte "Alter" / "Age" nicht gefunden.');
    return;
  }

  let added = 0;
  for (let i = 1; i < lines.length; i++) {
    const cols = lines[i].split(delimiter).map(c => c.trim().replace(/^["']|["']$/g, ''));
    if (cols.length < 2) continue;

    const getValue = (field) => colMap[field] !== undefined ? cols[colMap[field]] : '';

    const age = parseFloat(getValue('age'));
    if (!age || age < 18 || age > 65) continue;

    const genderRaw = getValue('gender').toLowerCase();
    const gender = (genderRaw === 'm' || genderRaw === 'male' || genderRaw === 'maennlich' || genderRaw === 'männlich' || genderRaw === 'masculino') ? 'Male' : 'Female';

    const icuRaw = getValue('icu').toLowerCase();
    const icuExperience = ['ja', 'yes', '1', 'true', 'si', 'x'].includes(icuRaw);

    const candidate = {
      id: state.nextId++,
      name: getValue('name') || `CSV-${i}`,
      age,
      gender,
      maritalStatus: normalizeMaritalStatus(getValue('maritalStatus')),
      hospital: normalizeHospital(getValue('hospital'), state.country),
      category: normalizeCategory(getValue('category')),
      icuExperience,
      yearsExperience: parseFloat(getValue('yearsExperience')) || 0,
      region: normalizeRegion(getValue('region'), state.country),
    };

    const result = scoreCandidate(state.country, candidate);
    candidate.score = result.probability;
    candidate.tier = result.tier;
    candidate.contributions = result.contributions;

    state.candidates.push(candidate);
    added++;
  }

  if (added > 0) {
    renderResults();
    renderSummary();
    alert(`${added} Kandidaten importiert.`);
  } else {
    alert('Keine gultigen Kandidaten in der CSV gefunden.');
  }
}

function normalizeMaritalStatus(val) {
  if (!val) return '';
  const v = val.toLowerCase().trim();
  if (['single', 'ledig'].includes(v)) return 'Single';
  if (['married', 'verheiratet'].includes(v)) return 'Married';
  if (['divorced', 'geschieden'].includes(v)) return 'Divorced';
  if (['widowed', 'verwitwet'].includes(v)) return 'Widowed';
  return val;
}

function normalizeHospital(val, country) {
  if (!val) return '';
  const v = val.toLowerCase().trim();
  const hospitals = MODELS[country]?.hospitals || [];
  // Exact match (case-insensitive)
  for (const h of hospitals) {
    if (h.toLowerCase() === v) return h;
  }
  // Partial match
  if (v.includes('tertiary') || v.includes('tertiaer')) return 'Tertiary';
  if (v.includes('primary') || v.includes('primaer')) return 'Primary';
  if (v.includes('secondary') || v.includes('sekundaer')) return 'Secondary';
  if (v.includes('regional')) return 'Regional';
  if (v.includes('republican') || v.includes('republik')) return 'Republican';
  if (v.includes('city') || v.includes('district') || v.includes('bezirk') || v.includes('stadt')) return 'City/District';
  return val;
}

function normalizeCategory(val) {
  if (!val) return '';
  const v = val.toLowerCase().trim();
  if (v.includes('general') || v.includes('allgemein')) return 'General Ward';
  if (v.includes('icu') || v.includes('intensiv')) return 'ICU';
  if (v.includes('emergency') || v.includes('notauf') || v.includes('er')) return 'ER';
  if (v.includes('operating') || v.includes('op-saal') || v.includes('op saal')) return 'OR';
  if (v.includes('surg') || v.includes('chirurg')) return 'Med-Surg';
  if (v.includes('geriatr')) return 'Geriatrics';
  if (v.includes('pediatr') || v.includes('paediatr') || v.includes('kinder')) return 'Pediatrics';
  if (v.includes('ob-gyn') || v.includes('obstetric') || v.includes('gynaek') || v.includes('geburt')) return 'OB-GYN';
  if (v.includes('dialys') || v.includes('nephro')) return 'Dialysis';
  if (v.includes('psych') || v.includes('mental')) return 'Psych';
  return 'Other';
}

function normalizeRegion(val, country) {
  if (!val || !MODELS[country]?.hasRegion) return '';
  const regions = Object.keys(REGIONS[country] || {});
  // Exact match
  const exact = regions.find(r => r.toLowerCase() === val.toLowerCase().trim());
  if (exact) return exact;
  // Partial match
  const partial = regions.find(r => val.toLowerCase().includes(r.toLowerCase()));
  if (partial) return partial;
  return 'Other';
}

// ═══════════════════════════════════════════════════════════
//  CSV EXPORT
// ═══════════════════════════════════════════════════════════

function exportCsv() {
  if (state.candidates.length === 0) {
    alert('Keine Kandidaten zum Exportieren.');
    return;
  }

  const sep = ';';
  const headers = ['Name', 'Alter', 'Geschlecht', 'Familienstand', 'Krankenhaus', 'Fachrichtung',
    'ICU', 'Berufsjahre', 'Region', 'Score (%)', 'Tier', 'Empfehlung'];

  const rows = state.candidates.map(c => {
    const tierCfg = TIERS[c.tier];
    let empfehlung = '';
    if (c.tier === 'HIGH') empfehlung = 'Hohes Risiko - genau pruefen';
    else if (c.tier === 'ELEVATED') empfehlung = 'Erhoehtes Risiko - beobachten';
    else if (c.tier === 'MEDIUM') empfehlung = 'Mittleres Risiko';
    else empfehlung = 'Niedriges Risiko';

    return [
      c.name, c.age, c.gender === 'Male' ? 'M' : 'W', c.maritalStatus || '',
      c.hospital || '', c.category || '', c.icuExperience ? 'Ja' : 'Nein',
      c.yearsExperience || '', c.region || '',
      (c.score * 100).toFixed(1).replace('.', ','), tierCfg.label, empfehlung,
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
//  INIT
// ═══════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', init);
