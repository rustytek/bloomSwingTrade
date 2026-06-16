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

/* ── Top navigation ──────────────────────────────────────────────────────── */
const NAV_LINKS = [
  ['/', 'Today', 'Your daily guided workflow: market regime, position alerts, and top setups with trade plans.'],
  ['/screener', 'Screener', 'Filter the full S&P 500 + ETF universe by fundamentals, technicals, and momentum.'],
  ['/charts', 'Charts', 'Market-wide dashboards: VIX, sectors, ETFs, breadth, and macro context.'],
  ['/backtest', 'Backtest', 'Test the swing strategies over cached history and compare them to SPY.'],
  ['/report', 'Report', 'AI-assisted daily market report and saved commentary.'],
  ['/journal', 'Journal', 'Closed-trade log with realized P&L, R-multiple, and win-rate stats.'],
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
      ['vs MA200', fmtPct(stock.vs_ma200)], ['Vol Ratio', fmtNum(stock.vol_r, 2)], ['Sharpe', fmtNum(stock.sharpe, 2)],
      ['Sortino', fmtNum(stock.sortino, 2)], ['Ann Ret', fmtPct(stock.ann_ret)],
      ['Max DD 1M', stock.max_dd_1m == null ? '--' : `${fmtNum(stock.max_dd_1m, 1)}%`],
      ['P/E', fmtNum(stock.pe, 1)], ['Beta', fmtNum(stock.beta, 2)], ['52W Pos', stock.p52w == null ? '--' : `${fmtNum(stock.p52w, 0)}%`],
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
