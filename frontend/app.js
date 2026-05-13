'use strict';

// ── State ─────────────────────────────────────────────────────────────────────
let srcLang = 'en';
let tgtLang = 'fr';
let selectedModel = 'transformer';

// ── DOM references ────────────────────────────────────────────────────────────
const srcTextarea   = document.getElementById('src-text');
const tgtOutput     = document.getElementById('tgt-text');
const charCount     = document.getElementById('char-count');
const translateBtn  = document.getElementById('translate-btn');
const btnText       = document.getElementById('btn-text');
const btnSpinner    = document.getElementById('btn-spinner');
const errorMsg      = document.getElementById('error-msg');
const clearBtn      = document.getElementById('clear-btn');
const copyBtn       = document.getElementById('copy-btn');
const swapBtn       = document.getElementById('swap-btn');
const modelBadge    = document.getElementById('model-badge');

const modelBtns     = document.querySelectorAll('.model-btn');
const langBtns      = document.querySelectorAll('.lang-btn');
const explainerRnn  = document.getElementById('explainer-rnn');
const explainerTfm  = document.getElementById('explainer-transformer');

// ── Model selector ────────────────────────────────────────────────────────────
modelBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    selectedModel = btn.dataset.model;
    modelBtns.forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    modelBadge.textContent = selectedModel === 'rnn' ? 'GRU RNN' : 'Transformer';
    // Highlight the matching explainer card
    explainerRnn.classList.toggle('active', selectedModel === 'rnn');
    explainerTfm.classList.toggle('active', selectedModel === 'transformer');
  });
});

// ── Language selector ─────────────────────────────────────────────────────────
langBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    const side = btn.dataset.side;
    const lang = btn.dataset.lang;
    const otherLang = lang === 'en' ? 'fr' : 'en';

    if (side === 'src') {
      if (lang === tgtLang) swapLanguages(); // avoid same lang on both sides
      else setSrcLang(lang);
    } else {
      if (lang === srcLang) swapLanguages();
      else setTgtLang(lang);
    }
  });
});

function setSrcLang(lang) {
  srcLang = lang;
  updateLangButtons();
}

function setTgtLang(lang) {
  tgtLang = lang;
  updateLangButtons();
}

function updateLangButtons() {
  document.querySelectorAll('[data-side="src"]').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.lang === srcLang);
  });
  document.querySelectorAll('[data-side="tgt"]').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.lang === tgtLang);
  });
}

// ── Swap ──────────────────────────────────────────────────────────────────────
swapBtn.addEventListener('click', swapLanguages);

function swapLanguages() {
  // Swap lang codes
  [srcLang, tgtLang] = [tgtLang, srcLang];
  updateLangButtons();

  // Swap text: put current translation into source textarea
  const currentTranslation = tgtOutput.dataset.translation || '';
  if (currentTranslation) {
    srcTextarea.value = currentTranslation;
    updateCharCount();
    clearOutput();
  }
}

// ── Char counter ──────────────────────────────────────────────────────────────
srcTextarea.addEventListener('input', updateCharCount);

function updateCharCount() {
  const len = srcTextarea.value.length;
  charCount.textContent = `${len} / 500`;
  charCount.classList.toggle('warn', len > 450);
}

// ── Clear ─────────────────────────────────────────────────────────────────────
clearBtn.addEventListener('click', () => {
  srcTextarea.value = '';
  updateCharCount();
  clearOutput();
  hideError();
  srcTextarea.focus();
});

function clearOutput() {
  tgtOutput.innerHTML = '<span class="placeholder">Translation will appear here</span>';
  tgtOutput.dataset.translation = '';
}

// ── Copy ──────────────────────────────────────────────────────────────────────
copyBtn.addEventListener('click', async () => {
  const text = tgtOutput.dataset.translation || '';
  if (!text) return;
  try {
    await navigator.clipboard.writeText(text);
    copyBtn.textContent = '✓';
    setTimeout(() => { copyBtn.textContent = '⎘'; }, 1500);
  } catch {
    // Fallback for browsers without clipboard API
    const ta = document.createElement('textarea');
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand('copy');
    document.body.removeChild(ta);
    copyBtn.textContent = '✓';
    setTimeout(() => { copyBtn.textContent = '⎘'; }, 1500);
  }
});

// ── Translate ─────────────────────────────────────────────────────────────────
translateBtn.addEventListener('click', doTranslate);

srcTextarea.addEventListener('keydown', e => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) doTranslate();
});

async function doTranslate() {
  const text = srcTextarea.value.trim();
  if (!text) return;

  setLoading(true);
  hideError();

  try {
    const res = await fetch('/api/translate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text,
        source_lang: srcLang,
        target_lang: tgtLang,
        model: selectedModel,
      }),
    });

    const data = await res.json();

    if (!res.ok) {
      showError(data.detail || `Server error ${res.status}`);
      return;
    }

    showTranslation(data.translation);
  } catch (err) {
    showError('Could not reach the server. Make sure the backend is running.');
  } finally {
    setLoading(false);
  }
}

// ── UI helpers ────────────────────────────────────────────────────────────────
function setLoading(isLoading) {
  translateBtn.disabled = isLoading;
  btnText.textContent = isLoading ? 'Translating…' : 'Translate';
  btnSpinner.classList.toggle('hidden', !isLoading);
}

function showTranslation(text) {
  tgtOutput.textContent = text;
  tgtOutput.dataset.translation = text;
}

function showError(msg) {
  errorMsg.textContent = msg;
  errorMsg.classList.remove('hidden');
}

function hideError() {
  errorMsg.classList.add('hidden');
}
