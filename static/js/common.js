'use strict';
/* Shared helpers for the vanilla-JS pages (today.html, journal.html).
   Auth/fetch, top-nav, formatting, and the universal ticker-detail modal. */

const token = () => localStorage.getItem('st_token');
const clearToken = () => localStorage.removeItem('st_token');
const authHdr = () => ({ Authorization: 'Bearer ' + token() });

async function api(url, opts = {}) {
  if (!token()) { window.location.href = '/login'; throw new Error('Not authenticated'); }
  const res = await fetch(url, { ...opts, headers: { ...authHdr(), ...(opts.headers || {}) } });
  if (res.status === 401) { clearToken(); window.location.href = '/login'; throw new Error('Session expired'); }
  if (!res.ok) throw new Error(await res.text());
  if (res.status === 204) return null;
  return res.json();
}

function esc(value) {
  return String(value ?? '').replace(/[&<>"']/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]));
}
function fmtPct(v, d = 1) {
  if (v == null) return '<span class="muted">--</span>';
  return `<span class="${v >= 0 ? 'up' : 'down'}">${v >= 0 ? '+' : ''}${Number(v).toFixed(d)}%</span>`;
}
function fmtNum(v, d = 2) { return v == null ? '--' : Number(v).toFixed(d); }
function fmtMoney(v, d = 2) { return v == null ? '--' : '$' + Number(v).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d }); }
function safeTicker(t) { return String(t || '').toUpperCase().replace(/[^A-Z0-9.-]/g, '').slice(0, 15); }

/* Canonical clickable-ticker markup (CLAUDE.md "Ticker Symbols — Universal Rule").
   Returns a `.ticker-link` span wired to openTickerDetail(). The ticker is
   sanitized by safeTicker() before it reaches the inline handler, so it can
   never carry a quote or angle bracket. An unusable ticker renders "--" rather
   than a dead, unclickable string. `fontSize` is any CSS length (e.g. '15px').
   Additive helper: existing pages that build the span by hand are unaffected. */
function tickerLink(ticker, fontSize) {
  const t = safeTicker(ticker);
  if (!t) return '<span class="muted">--</span>';
  const style = fontSize ? ` style="font-size:${esc(fontSize)};"` : '';
  return `<span class="ticker-link"${style} onclick="openTickerDetail('${t}')">${esc(t)}</span>`;
}

/* ── MA trend helpers (quote schema_v 2) ─────────────────────────────────────
   `ma_state` is the STATE ("bull" = MA50 above MA200 right now, "bear" = below).
   `gc`/`dc` (aliased as `gc_event`/`dc_event`) are CROSSOVER EVENTS that fired
   within the last 5 bars. Older cached rows predate these keys — fall back to
   "--" rather than rendering undefined. */
function maState(stock) {
  return stock && (stock.ma_state === 'bull' || stock.ma_state === 'bear') ? stock.ma_state : null;
}
function gcEvent(stock) { return !!(stock && (stock.gc_event ?? stock.gc)); }
function dcEvent(stock) { return !!(stock && (stock.dc_event ?? stock.dc)); }
function _maStateLabel(stock) {
  const s = maState(stock);
  if (s === 'bull') return 'Bullish (MA50 &gt; MA200)';
  if (s === 'bear') return 'Bearish (MA50 &lt; MA200)';
  return '--';
}
function _maCrossLabel(stock) {
  if (gcEvent(stock)) return 'Golden cross — last 5 days';
  if (dcEvent(stock)) return 'Death cross — last 5 days';
  return 'No cross';
}

/* ── Top navigation ──────────────────────────────────────────────────────── */
const NAV_LINKS = [
  // Order is the intended WORKFLOW, not alphabetical: the Playbook tells you
  // which strategy the regime picked, and Strategy Lab sits right next to it
  // because that is where you go to disagree with that pick and test your own.
  // Playbook -> Weekly Plan -> ... -> Trade is the weekly loop: see the market,
  // walk through what to do and why, then (last tab) send the ticked orders.
  ['/', 'Playbook', 'Your daily workflow: market regime, position alerts, and setups grouped by strategy with their tested edge.'],
  ['/plan', 'Weekly Plan', 'Step-by-step weekly check: the market, the strategies in play and why, what to do with each holding, and new trades with the reasoning for each.'],
  ['/backtest', 'Strategy Lab', 'Edge matrix by regime, walk-forward backtests (rotation vs trade-plan mode), and robustness checks. Go here to test your OWN forecast instead of the regime’s pick.'],
  ['/screener', 'Screener', 'Filter the full S&P 500 + ETF universe by fundamentals, technicals, and momentum.'],
  ['/charts', 'Charts', 'Market-wide dashboards: VIX, sectors, ETFs, breadth, and macro context.'],
  ['/scorecard', 'Scorecard', 'Realized results vs what the backtest expected, plus execution quality ranked by what it cost.'],
  ['/report', 'Report', 'AI-assisted daily market report and saved commentary.'],
  ['/journal', 'Journal', 'Closed-trade log with realized P&L, R-multiple, and win-rate stats.'],
  ['/trade', 'Trade', 'Send the trades you picked in the Weekly Plan to Robinhood — paper mode first, then live with explicit confirmation.'],
];

function buildTopbar(active, leftExtraHtml = '') {
  const links = NAV_LINKS.map(([href, label, tip]) => {
    const cls = href === active ? 'btn hint btn-active' : 'btn hint';
    return `<a class="${cls}" data-tip="${esc(tip)}" href="${href}">${esc(label)}</a>`;
  }).join('');
  // Page-specific controls (leftExtraHtml) sit next to the brand on the left;
  // the standard nav + logout always live on the right.
  return `
    <a href="/" class="brand">▲ SWING TRADER</a>
    <div class="topbar-sep"></div>
    <div class="universe-status"><span id="universeText" class="universe-text">Loading…</span><span id="dataTime" class="data-time"></span></div>
    ${leftExtraHtml ? `<div class="topbar-left">${leftExtraHtml}</div>` : ''}
    <div class="topbar-actions">
      ${links}
      <span id="username" class="username"></span>
      <button id="logoutBtn" class="btn hint" data-tip="End this browser session." type="button">Logout</button>
    </div>`;
}

async function initHeader(active, leftExtraHtml = '') {
  const bar = document.getElementById('topbar');
  if (bar) bar.innerHTML = buildTopbar(active, leftExtraHtml);
  const logout = document.getElementById('logoutBtn');
  if (logout) logout.addEventListener('click', () => { clearToken(); window.location.href = '/login'; });
  try {
    const user = await api('/auth/me');
    const u = document.getElementById('username');
    if (u) u.textContent = user.username || '';
    // Admin-only Users link, inserted before the username on the right.
    if (user.is_admin && u && !document.getElementById('adminLink')) {
      const a = document.createElement('a');
      a.id = 'adminLink';
      a.className = 'btn hint' + (active === '/admin' ? ' btn-active' : '');
      a.href = '/admin';
      a.textContent = 'Users';
      a.setAttribute('data-tip', 'Manage users, passwords, and per-user LiteLLM keys.');
      u.parentNode.insertBefore(a, u);
    }
  } catch (e) { return; }
  try {
    const status = await api('/api/screener/universe/status');
    const loaded = status.loaded || 0, total = status.total || 0;
    const isLoading = total > 0 && loaded < total;
    const text = document.getElementById('universeText');
    if (text) { text.classList.toggle('loading', isLoading); text.textContent = isLoading ? `Loading universe… ${loaded}/${total}` : `${loaded} stocks loaded`; }
    const dt = document.getElementById('dataTime');
    if (dt) dt.textContent = !isLoading && status.last_updated_time ? `Data as of ${status.last_updated_time}` : '';
  } catch (e) { /* non-fatal */ }
}

/* ── Universal ticker-detail modal (CLAUDE.md hard rule) ─────────────────── */
let _tdChart = null;

function _ensureModal() {
  if (document.getElementById('tdModal')) return;
  const div = document.createElement('div');
  div.id = 'tdModal';
  div.className = 'modal';
  div.innerHTML = `
    <div class="modal-card">
      <div class="modal-head">
        <span id="tdTicker" class="mono" style="font-weight:700;color:#f0a500;font-size:16px;"></span>
        <span id="tdName" class="muted" style="font-size:12px;"></span>
        <span id="tdPrice" class="mono" style="margin-left:8px;"></span>
        <button class="icon-btn" onclick="closeTickerModal()">✕ Close</button>
      </div>
      <div class="modal-body">
        <div class="chart-wrap"><canvas id="tdChart"></canvas></div>
        <div id="tdMetrics"></div>
      </div>
    </div>`;
  div.addEventListener('click', (e) => { if (e.target === div) closeTickerModal(); });
  document.body.appendChild(div);
}

async function openTickerDetail(ticker) {
  ticker = safeTicker(ticker);
  _ensureModal();
  document.getElementById('tdModal').style.display = 'flex';
  document.getElementById('tdTicker').textContent = ticker;
  document.getElementById('tdName').textContent = 'Loading…';
  document.getElementById('tdPrice').textContent = '';
  document.getElementById('tdMetrics').innerHTML = '';
  try {
    const [stock, hist] = await Promise.all([
      api('/api/stocks/' + encodeURIComponent(ticker)),
      api('/api/stocks/' + encodeURIComponent(ticker) + '/history').catch(() => ({ bars: [] })),
    ]);
    document.getElementById('tdName').textContent = [stock.name, stock.sector].filter(Boolean).join(' · ');
    document.getElementById('tdPrice').innerHTML = stock.price != null ? `$${Number(stock.price).toFixed(2)} ${fmtPct(stock.chg_pct)}` : '--';
    const rows = [
      ['RSI', fmtNum(stock.rsi, 1)], ['MACD', esc(stock.macd_sig || '--')], ['vs MA50', fmtPct(stock.vs_ma50)],
      ['vs MA200', fmtPct(stock.vs_ma200)],
      ['Vol Ratio (today vs 20D avg)', fmtNum(stock.vol_r, 2)],
      ['Vol Ratio (5D vs 20D avg)', stock.vol_r_5d == null ? '--' : fmtNum(stock.vol_r_5d, 2)],
      ['Sharpe', fmtNum(stock.sharpe, 2)],
      ['Sortino', fmtNum(stock.sortino, 2)],
      ['Ann Ret (full)', fmtPct(stock.ann_ret)],
      ['Ann Ret (1M)', stock.ann_ret_1m == null ? '--' : fmtPct(stock.ann_ret_1m)],
      ['Max DD 1M', stock.max_dd_1m == null ? '--' : `${fmtNum(stock.max_dd_1m, 1)}%`],
      ['P/E', fmtNum(stock.pe, 1)], ['Beta', fmtNum(stock.beta, 2)], ['52W Pos', stock.p52w == null ? '--' : `${fmtNum(stock.p52w, 0)}%`],
      ['MA Trend', _maStateLabel(stock)],
      ['MA Cross (last 5 bars)', _maCrossLabel(stock)],
      ['Swing Score', stock.swing_score == null ? '--' : fmtNum(stock.swing_score, 0)],
    ];
    document.getElementById('tdMetrics').innerHTML = `<table><tbody>${rows.map(([k, v]) => `<tr><td class="muted">${k}</td><td class="mono">${v}</td></tr>`).join('')}</tbody></table>`;
    _tdChart?.destroy();
    _tdChart = new Chart(document.getElementById('tdChart'), {
      type: 'line',
      data: {
        labels: (hist.bars || []).map(b => b.date),
        datasets: [{ label: ticker, data: (hist.bars || []).map(b => b.close), borderColor: '#f0a500', tension: .15, pointRadius: 0 }],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { labels: { color: '#aaa' } } },
        scales: { x: { ticks: { color: '#777', maxTicksLimit: 8 }, grid: { color: '#1a1a1a' } }, y: { ticks: { color: '#777' }, grid: { color: '#1a1a1a' } } },
      },
    });
  } catch (e) {
    document.getElementById('tdName').innerHTML = `<span class="error">${esc(e.message)}</span>`;
  }
}

function closeTickerModal() {
  const m = document.getElementById('tdModal');
  if (m) m.style.display = 'none';
  _tdChart?.destroy();
  _tdChart = null;
}

/* ── Account circuit breaker (services/circuit_breaker.py, ML4T gap 7) ──
   Loud banner for the Playbook, Weekly Plan and Trade pages. Returns '' when
   the breaker is ok, so pages can call it unconditionally. */
const _CB_TITLES = {
  warn: 'CIRCUIT BREAKER — WARNING',
  reduce: 'CIRCUIT BREAKER — SIZE DOWN',
  halt: 'CIRCUIT BREAKER — NEW ENTRIES PAUSED',
  unknown: 'CIRCUIT BREAKER — NOT EVALUATED',
};
function breakerBannerHtml(cb) {
  if (!cb || !cb.level || cb.level === 'ok') return '';
  const lvl = _CB_TITLES[cb.level] ? cb.level : 'unknown';
  const m = cb.metrics || {};
  const reasons = (cb.reasons || []).filter(r => r && r.message)
    .map(r => `<li>${esc(r.message)}</li>`).join('');
  const bits = [];
  if (m.equity != null) bits.push('equity ' + fmtMoney(m.equity, 0));
  if (m.peak_equity != null) bits.push('peak ' + fmtMoney(m.peak_equity, 0));
  if (m.drawdown_pct != null) bits.push('drawdown ' + Number(m.drawdown_pct).toFixed(1) + '%');
  if (m.daily_change_pct != null) bits.push('today ' + (m.daily_change_pct >= 0 ? '+' : '') + Number(m.daily_change_pct).toFixed(1) + '%');
  if (m.loss_streak) bits.push(m.loss_streak + ' losses in a row');
  const ack = cb.can_acknowledge
    ? `<div class="cb-actions"><button class="btn btn-primary btn-sm" type="button" onclick="acknowledgeBreaker(this)">I understand — resume at reduced size</button>
       <span style="font-size:11px;opacity:.8;">Recorded with your name and the time. It halts again if the drawdown deepens a further ${esc((cb.thresholds || {}).rehalt_step_pct ?? 3)} points.</span></div>`
    : '';
  return `<div class="cb-banner cb-${lvl}" role="alert"><div class="cb-title">${_CB_TITLES[lvl]}</div>`
    + (reasons ? `<ul>${reasons}</ul>` : '')
    + (bits.length ? `<div class="cb-metrics hint" data-tip="${esc(cb.equity_definition || '')}">${esc(bits.join(' · '))}</div>` : '')
    + ack + '</div>';
}
async function acknowledgeBreaker(btn) {
  if (btn) { btn.disabled = true; btn.textContent = 'Recording…'; }
  try {
    await api('/api/risk/breaker/acknowledge', { method: 'POST' });
    window.location.reload();
  } catch (e) {
    if (btn) { btn.disabled = false; btn.textContent = 'Acknowledge failed — retry'; }
  }
}
