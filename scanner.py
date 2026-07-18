"""
Daily Stock + Options Buy/Exit Scanner
=======================================
Each 15-min cycle:
  1. Full S&P 500 + watchlist scan (buy signals, sector sympathy)
  2. Fast loop: watchlist + open positions re-scanned every 2 minutes
     for the rest of the cycle (near-real-time alerts on your names)

BUY alerts include a suggested options contract:
  21-45 DTE calls, delta ~0.55-0.70, liquidity-filtered
  (open interest >= 200, bid-ask spread <= 12% of mid),
  with mid price and breakeven.

EXIT alerts fire on your POSITIONS when:
  - stop % or target % from entry is hit
  - price breaks below the 20 SMA
  - heavy upper-wick rejection while RSI > 70

Signals scored (max 100): trend 25, wicks 20, prior close/gap 15,
relative volume 15, RSI 10, sector sympathy 15.
"""

import json
import os
import smtplib
import subprocess
import time
from datetime import datetime, date
from email.mime.text import MIMEText
from math import log, sqrt, erf
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

# ----------------------------------------------------------------------
# CONFIG — edit these
# ----------------------------------------------------------------------

# Your personal watchlist (fast-loop scanned, lower alert threshold, * in texts)
WATCHLIST = [
    "AAPL", "AMD", "AMZN", "AXP", "AXSM", "COIN", "CVNA", "EA",
    "GOOGL", "HOOD", "JPM", "META", "MRVL", "MSFT", "MU", "NET",
    "NVDA", "PLTR", "QQQ", "RBRK", "SMCI", "SOFI", "SPY", "TSLA",
    "UNH", "VOO", "VTI", "WYNN",
    # added — very liquid options chains, delete any you don't want:
    "AVGO", "TSM", "NFLX", "ORCL",
]

# Open positions you want EXIT alerts on. Examples:
#   {"ticker": "NVDA", "entry": 180.00, "stop_pct": 6, "target_pct": 15}
#   entry = underlying price when you opened the trade (works for shares
#   or as the underlying reference for an options position)
POSITIONS: list[dict] = [
    # {"ticker": "NVDA", "entry": 180.00, "stop_pct": 6, "target_pct": 15},
]

SCORE_THRESHOLD = 60          # S&P 500 tickers must score >= this
WATCHLIST_THRESHOLD = 50      # your watchlist alerts a bit earlier
MAX_ALERTS_PER_RUN = 5        # don't blow up your phone
SUGGEST_OPTIONS = True        # attach a contract suggestion to buy alerts

FAST_INTERVAL_SEC = 120       # watchlist/positions re-scan cadence
RUN_LOOP_MINUTES = 12         # how long each job keeps fast-looping

# Options contract filters
OPT_MIN_DTE, OPT_MAX_DTE = 21, 45
OPT_MIN_DELTA, OPT_MAX_DELTA = 0.50, 0.75
OPT_TARGET_DELTA = 0.62
OPT_MIN_OI = 200
OPT_MAX_SPREAD_PCT = 0.12
RISK_FREE = 0.045

# Credentials come from environment variables (GitHub Secrets):
GMAIL_ADDRESS = os.environ.get("GMAIL_ADDRESS", "")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "")
SMS_ADDRESS = os.environ.get("SMS_ADDRESS", "")

STATE_FILE = "state.json"
ET = ZoneInfo("America/New_York")
DASH_ROWS = 20

# ----------------------------------------------------------------------
# Market hours guard
# ----------------------------------------------------------------------

def market_is_open() -> bool:
    now = datetime.now(ET)
    if now.weekday() >= 5:
        return False
    open_t = now.replace(hour=9, minute=30, second=0, microsecond=0)
    close_t = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return open_t <= now <= close_t


# ----------------------------------------------------------------------
# Universe: S&P 500 constituents + sectors (from Wikipedia)
# ----------------------------------------------------------------------

def get_sp500() -> pd.DataFrame:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    import requests; from io import StringIO
    html = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}, timeout=30).text
    table = pd.read_html(StringIO(html))[0]

    df = table[["Symbol", "GICS Sector"]].copy()
    df.columns = ["ticker", "sector"]
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
    return df


# ----------------------------------------------------------------------
# Data download (batched)
# ----------------------------------------------------------------------

def download_daily(tickers: list[str]) -> pd.DataFrame:
    return yf.download(tickers, period="10mo", interval="1d",
                       group_by="ticker", auto_adjust=True,
                       threads=True, progress=False)


def download_intraday(tickers: list[str], interval: str = "15m") -> pd.DataFrame:
    return yf.download(tickers, period="1d", interval=interval,
                       group_by="ticker", auto_adjust=True,
                       threads=True, progress=False)


def split_by_ticker(bulk: pd.DataFrame, tickers: list[str]) -> dict:
    out = {}
    for t in tickers:
        try:
            d = bulk[t].dropna(how="all")
            if len(d):
                out[t] = d
        except (KeyError, IndexError):
            continue
    return out


# ----------------------------------------------------------------------
# Indicators
# ----------------------------------------------------------------------

def rsi(series: pd.Series, length: int = 14) -> float:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / length, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / length, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return float((100 - 100 / (1 + rs)).iloc[-1])


def wick_score(intraday: pd.DataFrame) -> tuple[int, str]:
    """Long lower wicks near lows = absorption (bullish).
    Long upper wicks = rejection (bearish)."""
    if intraday is None or len(intraday) < 3:
        return 0, ""
    recent = intraday.tail(4)
    pts, note = 0, ""
    for _, c in recent.iterrows():
        rng = c["High"] - c["Low"]
        if rng <= 0 or np.isnan(rng):
            continue
        body = abs(c["Close"] - c["Open"])
        lower_wick = min(c["Open"], c["Close"]) - c["Low"]
        upper_wick = c["High"] - max(c["Open"], c["Close"])
        if lower_wick > 2 * body and lower_wick / rng > 0.5:
            pts += 8
            note = "lower-wick absorption"
        if upper_wick > 2 * body and upper_wick / rng > 0.55:
            pts -= 6
            note = "upper-wick rejection"
    return max(min(pts, 20), -10), note


def score_ticker(daily: pd.DataFrame, intraday: pd.DataFrame,
                 sector_chg: float, stock_chg: float,
                 live_price: float | None = None) -> tuple[int, list[str]]:
    if daily is None or len(daily) < 200:
        return 0, []
    close = daily["Close"].dropna()
    vol = daily["Volume"].dropna()
    price = live_price if live_price else float(close.iloc[-1])

    sma20 = float(close.rolling(20).mean().iloc[-1])
    sma50 = float(close.rolling(50).mean().iloc[-1])
    sma200 = float(close.rolling(200).mean().iloc[-1])

    score, reasons = 0, []

    # Trend structure (25)
    if price > sma200:
        score += 8
    if price > sma50:
        score += 7
    if sma20 > sma50 > sma200:
        score += 10
        reasons.append("SMAs stacked bullish (20>50>200)")
    elif price > sma200:
        reasons.append("above 200 SMA")

    # Wicks (20, can go negative)
    w_pts, w_note = wick_score(intraday)
    score += w_pts
    if w_note and w_pts > 0:
        reasons.append(w_note)

    # Prior close / gap behavior (15)
    prev_close = float(close.iloc[-2])
    prev_high = float(daily["High"].iloc[-2])
    prev_low = float(daily["Low"].iloc[-2])
    today_open = float(daily["Open"].iloc[-1])
    prev_rng = prev_high - prev_low
    if prev_rng > 0 and (prev_close - prev_low) / prev_rng > 0.66:
        score += 7
        reasons.append("closed strong yesterday")
    if today_open < prev_close and price > prev_close:
        score += 8
        reasons.append("gap-down reclaimed")
    elif today_open > prev_high and price > prev_high:
        score += 5
        reasons.append("gap-up holding")

    # Relative volume (15)
    avg_vol = float(vol.tail(30).mean())
    today_vol = float(vol.iloc[-1])
    now = datetime.now(ET)
    mins_open = max((now.hour - 9) * 60 + now.minute - 30, 15)
    est_full = today_vol * (390 / min(mins_open, 390))
    rvol = est_full / avg_vol if avg_vol > 0 else 0
    if rvol > 2.0:
        score += 15
        reasons.append(f"RVOL {rvol:.1f}x")
    elif rvol > 1.4:
        score += 8
        reasons.append(f"RVOL {rvol:.1f}x")

    # RSI (10)
    r = rsi(close)
    if not np.isnan(r):
        if 35 <= r <= 50 and price > sma200:
            score += 10
            reasons.append(f"RSI {r:.0f} pullback in uptrend")
        elif 50 < r <= 65:
            score += 5

    # Sector sympathy (15)
    if sector_chg > 0.8:
        if stock_chg < sector_chg:
            score += 15
            reasons.append(f"sector +{sector_chg:.1f}%, stock lagging (catch-up)")
        else:
            score += 7
            reasons.append(f"sector momentum +{sector_chg:.1f}%")

    return score, reasons


# ----------------------------------------------------------------------
# Options: suggest a contract on buy signals
# ----------------------------------------------------------------------

def bs_call_delta(S: float, K: float, T: float, iv: float,
                  r: float = RISK_FREE) -> float:
    if T <= 0 or iv <= 0 or S <= 0 or K <= 0:
        return float("nan")
    d1 = (log(S / K) + (r + iv * iv / 2) * T) / (iv * sqrt(T))
    return 0.5 * (1 + erf(d1 / sqrt(2)))


def suggest_contract(ticker: str, S: float) -> dict | None:
    """Best liquid call, 21-45 DTE, delta ~0.62. Returns None if nothing
    passes the liquidity filters (that itself is a signal: skip it)."""
    try:
        tk = yf.Ticker(ticker)
        expirations = tk.options or []
    except Exception:
        return None
    today = date.today()
    best, best_score = None, -1e9
    for exp in expirations:
        try:
            dte = (datetime.strptime(exp, "%Y-%m-%d").date() - today).days
        except ValueError:
            continue
        if not (OPT_MIN_DTE <= dte <= OPT_MAX_DTE):
            continue
        try:
            calls = tk.option_chain(exp).calls
        except Exception:
            continue
        window = calls[(calls["strike"] >= S * 0.90) & (calls["strike"] <= S * 1.08)]
        for _, row in window.iterrows():
            bid = float(row.get("bid", 0) or 0)
            ask = float(row.get("ask", 0) or 0)
            oi = float(row.get("openInterest", 0) or 0)
            iv = float(row.get("impliedVolatility", 0) or 0)
            mid = (bid + ask) / 2
            if mid <= 0.05 or bid <= 0 or oi < OPT_MIN_OI:
                continue
            spread_pct = (ask - bid) / mid
            if spread_pct > OPT_MAX_SPREAD_PCT:
                continue
            delta = bs_call_delta(S, float(row["strike"]), dte / 365, iv)
            if np.isnan(delta) or not (OPT_MIN_DELTA <= delta <= OPT_MAX_DELTA):
                continue
            sc = (-abs(delta - OPT_TARGET_DELTA) * 10
                  - spread_pct * 5
                  + min(oi, 5000) / 5000)
            if sc > best_score:
                d = datetime.strptime(exp, "%Y-%m-%d")
                best_score = sc
                best = {
                    "label": f"{d.month}/{d.day} ${row['strike']:.0f}C",
                    "exp": f"{d.month}/{d.day}",
                    "strike": float(row["strike"]),
                    "entry_lo": round(S * 0.99, 2 if S < 50 else 0),
                    "entry_hi": round(S * 1.01, 2 if S < 50 else 0),
                    "mid": mid, "delta": delta, "oi": int(oi),
                    "iv": iv, "dte": dte,
                    "breakeven": float(row["strike"]) + mid,
                    "spread_pct": spread_pct,
                }
    return best


def fmt_option(o: dict | None) -> str:
    if not o:
        return ""
    dec = 2 if o["strike"] < 50 else 0
    return (f"Entry ${o['entry_lo']:.{dec}f}-{o['entry_hi']:.{dec}f}, "
            f"exp {o['exp']}, strike ${o['strike']:.0f}C, "
            f"prem ~${o['mid']:.2f}, BE ${o['breakeven']:.2f}")


# ----------------------------------------------------------------------
# Exit engine
# ----------------------------------------------------------------------

def check_exits(per_daily: dict, live_prices: dict,
                per_intra: dict, state: dict) -> list[str]:
    alerts = []
    for p in POSITIONS:
        t = p["ticker"]
        if t in state["exit_alerted"] or t not in per_daily:
            continue
        close = per_daily[t]["Close"].dropna()
        price = live_prices.get(t, float(close.iloc[-1]))
        entry = float(p["entry"])
        chg = (price / entry - 1) * 100
        reasons = []

        stop = p.get("stop_pct")
        target = p.get("target_pct")
        if stop and chg <= -abs(stop):
            reasons.append(f"stop hit ({chg:+.1f}% from entry)")
        if target and chg >= abs(target):
            reasons.append(f"target hit ({chg:+.1f}% from entry)")

        sma20 = float(close.rolling(20).mean().iloc[-1])
        if price < sma20 and entry >= sma20:
            reasons.append("broke below 20 SMA")

        w_pts, w_note = wick_score(per_intra.get(t))
        r = rsi(close)
        if w_pts < 0 and not np.isnan(r) and r > 70:
            reasons.append(f"rejection wicks with RSI {r:.0f}")

        if reasons:
            alerts.append(f"EXIT {t} ${price:.2f} ({chg:+.1f}%) — "
                          + "; ".join(reasons[:2]))
            state["exit_alerted"].append(t)
    return alerts


# ----------------------------------------------------------------------
# Alerting + state
# ----------------------------------------------------------------------

def send_sms(body: str):
    if not (GMAIL_ADDRESS and GMAIL_APP_PASSWORD and SMS_ADDRESS):
        print("Missing email credentials — printing alert instead:\n" + body)
        return
    msg = MIMEText(body)
    msg["From"] = GMAIL_ADDRESS
    msg["To"] = SMS_ADDRESS
    msg["Subject"] = ""
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
            s.send_message(msg)
        print("Alert sent:\n" + body)
    except Exception as e:
        print(f"SMS send failed: {e}\n{body}")


def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                state = json.load(f)
            if state.get("date") == str(date.today()):
                state.setdefault("exit_alerted", [])
                return state
        except (json.JSONDecodeError, OSError):
            pass
    return {"date": str(date.today()), "alerted": [], "exit_alerted": []}


def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


# ----------------------------------------------------------------------
# Mobile dashboard
# ----------------------------------------------------------------------

def sparkline_svg(closes: list[float]) -> str:
    pts_in = [c for c in closes if c and not np.isnan(c)]
    if len(pts_in) < 2:
        return ""
    w, h = 92, 26
    mn, mx = min(pts_in), max(pts_in)
    rng = (mx - mn) or 1e-9
    n = len(pts_in)
    pts = " ".join(f"{i * (w / (n - 1)):.1f},{h - 2 - (c - mn) / rng * (h - 4):.1f}"
                   for i, c in enumerate(pts_in))
    color = "#6FBF8F" if pts_in[-1] >= pts_in[0] else "#D2707A"
    return (f'<svg class="spark" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
            f'<polyline points="{pts}" fill="none" stroke="{color}" '
            f'stroke-width="1.5" stroke-linejoin="round"/></svg>')


def write_dashboard(results: list[dict], state: dict):
    now = datetime.now(ET)
    stamp = now.strftime("%a %b %-d · %-I:%M %p ET")
    rows = results[:DASH_ROWS]

    # ---------- RESULTS.md ----------
    md = [f"# Buy Scan — {stamp}", ""]
    if not rows:
        md.append("_No scored setups this run._")
    else:
        md.append("| # | Ticker | Price | Day | Score | Contract idea | Why |")
        md.append("|---|--------|-------|-----|-------|---------------|-----|")
        for i, r in enumerate(rows, 1):
            name = f"**{r['ticker']}**" + (" S" if r["watchlist"] else "")
            if r["alerted"]:
                name += " !"
            opt = fmt_option(r.get("option")) or "—"
            md.append(f"| {i} | {name} | ${r['price']:.2f} | {r['chg']:+.1f}% "
                      f"| {r['score']} | {opt} | {'; '.join(r['reasons'][:3])} |")
        md += ["", "S = watchlist · ! = alerted today",
               "", f"Buy alerts today: {', '.join(state['alerted']) or '—'}",
               f"Exit alerts today: {', '.join(state['exit_alerted']) or '—'}"]
    with open("RESULTS.md", "w") as f:
        f.write("\n".join(md) + "\n")

    # ---------- docs/index.html ----------
    os.makedirs("docs", exist_ok=True)

    def card(i, r):
        chg_cls = "up" if r["chg"] >= 0 else "dn"
        badges = ""
        if r["watchlist"]:
            badges += '<span class="badge wl">WATCHLIST</span>'
        if r["alerted"]:
            badges += '<span class="badge al">ALERTED</span>'
        pct = min(max(r["score"], 0), 100)
        reasons = " · ".join(r["reasons"][:3])
        opt = r.get("option")
        opt_html = ""
        if opt:
            dec = 2 if opt["strike"] < 50 else 0
            opt_html = (f'<div class="opt"><span class="optlabel">CALL IDEA</span> '
                        f'Entry ${opt["entry_lo"]:.{dec}f}&ndash;{opt["entry_hi"]:.{dec}f} · '
                        f'Exp {opt["exp"]} · Strike ${opt["strike"]:.0f}C · '
                        f'Prem ~${opt["mid"]:.2f} · BE ${opt["breakeven"]:.2f} · '
                        f'&Delta;{opt["delta"]:.2f} · OI {opt["oi"]:,}</div>')
        spark = sparkline_svg(r.get("spark", []))
        return f"""
      <div class="card">
        <div class="rowtop">
          <span class="rank">{i:02d}</span>
          <span class="tick">{r['ticker']}</span>
          {badges}
          <span class="score">{r['score']}</span>
        </div>
        <div class="gauge"><div class="fill" style="width:{pct}%"></div></div>
        <div class="rowmid">
          <span class="price">${r['price']:.2f}</span>
          <span class="chg {chg_cls}">{r['chg']:+.1f}%</span>
          {spark}
          <span class="sector">{r['sector']}</span>
        </div>
        {opt_html}
        <div class="why">{reasons}</div>
      </div>"""

    cards = "".join(card(i, r) for i, r in enumerate(rows, 1)) or \
        '<div class="empty">No scored setups this run.<br>Next scan shortly.</div>'

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Buy Scan</title>
<style>
  :root {{
    --bg:#101214; --panel:#181B1E; --line:#26292D;
    --ink:#E8E6E1; --dim:#8B8F94; --gold:#C8A96A;
    --up:#6FBF8F; --dn:#D2707A;
  }}
  * {{ margin:0; box-sizing:border-box; }}
  body {{ background:var(--bg); color:var(--ink);
    font:15px/1.45 -apple-system,"SF Pro Text",Segoe UI,Roboto,sans-serif;
    padding:16px 14px 40px; max-width:520px; margin:0 auto; }}
  header {{ padding:6px 2px 16px; border-bottom:1px solid var(--line);
    margin-bottom:14px; display:flex; align-items:baseline; justify-content:space-between; }}
  h1 {{ font-family:Georgia,'Times New Roman',serif; font-weight:400;
    font-size:21px; letter-spacing:.04em; }}
  h1 em {{ color:var(--gold); font-style:normal; }}
  .stamp {{ color:var(--dim); font-size:12px; text-align:right; }}
  .card {{ background:var(--panel); border:1px solid var(--line);
    border-radius:12px; padding:13px 14px 12px; margin-bottom:10px; }}
  .rowtop {{ display:flex; align-items:center; gap:8px; }}
  .rank {{ color:var(--dim); font-size:12px; font-variant-numeric:tabular-nums; }}
  .tick {{ font-size:18px; font-weight:700; letter-spacing:.03em; }}
  .badge {{ font-size:9px; letter-spacing:.12em; padding:3px 6px 2px;
    border-radius:4px; font-weight:600; }}
  .badge.wl {{ color:var(--gold); border:1px solid var(--gold); }}
  .badge.al {{ color:var(--bg); background:var(--gold); }}
  .score {{ margin-left:auto; font-size:22px; font-weight:700;
    color:var(--gold); font-variant-numeric:tabular-nums; }}
  .gauge {{ height:3px; background:var(--line); border-radius:2px; margin:9px 0 10px; }}
  .fill {{ height:100%; background:var(--gold); border-radius:2px; }}
  .rowmid {{ display:flex; gap:10px; align-items:center; }}
  .price {{ font-weight:600; font-variant-numeric:tabular-nums; }}
  .chg {{ font-weight:600; font-variant-numeric:tabular-nums; }}
  .chg.up {{ color:var(--up); }} .chg.dn {{ color:var(--dn); }}
  .spark {{ opacity:.9; }}
  .sector {{ margin-left:auto; color:var(--dim); font-size:11px;
    letter-spacing:.06em; text-transform:uppercase; text-align:right; }}
  .opt {{ margin-top:9px; padding:7px 9px; border:1px solid var(--line);
    border-left:2px solid var(--gold); border-radius:6px;
    font-size:12.5px; color:var(--ink); font-variant-numeric:tabular-nums; }}
  .optlabel {{ color:var(--gold); font-size:9px; letter-spacing:.14em;
    font-weight:700; margin-right:6px; }}
  .why {{ color:var(--dim); font-size:13px; margin-top:7px; }}
  .empty {{ color:var(--dim); text-align:center; padding:60px 0; }}
  footer {{ color:var(--dim); font-size:11px; text-align:center; margin-top:22px; }}
</style>
</head>
<body>
  <header>
    <h1>Buy <em>Scan</em></h1>
    <div class="stamp">{stamp}<br>auto-refreshes every 5 min</div>
  </header>
  {cards}
  <footer>Screener output — not financial advice. Contract ideas are
  liquidity-filtered suggestions, not recommendations. Data via Yahoo
  Finance (may be delayed).</footer>
  <script>setTimeout(function() {{ location.reload(); }}, 300000);</script>
</body>
</html>"""
    with open("docs/index.html", "w") as f:
        f.write(html)


def publish():
    """Commit + push dashboard/state mid-run when running in Actions."""
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return
    try:
        subprocess.run(["git", "add", "-A", "RESULTS.md", "docs", STATE_FILE],
                       check=False, capture_output=True)
        r = subprocess.run(["git", "diff", "--cached", "--quiet"],
                           capture_output=True)
        if r.returncode != 0:
            subprocess.run(["git", "commit", "-m", "scan update"],
                           check=False, capture_output=True)
            subprocess.run(["git", "pull", "--rebase"], check=False,
                           capture_output=True)
            subprocess.run(["git", "push"], check=False, capture_output=True)
    except Exception as e:
        print(f"publish skipped: {e}")


# ----------------------------------------------------------------------
# Scan passes
# ----------------------------------------------------------------------

def build_results(tickers, per_daily, per_intra, sector_map, sector_chg,
                  day_chg, live_prices, state) -> list[dict]:
    results = []
    for t in tickers:
        if t not in per_daily:
            continue
        sec = sector_map.get(t, "")
        try:
            score, reasons = score_ticker(
                per_daily[t], per_intra.get(t),
                sector_chg.get(sec, 0.0), day_chg.get(t, 0.0),
                live_prices.get(t))
        except Exception as e:
            print(f"{t}: scoring error {e}")
            continue
        if score <= 0 or not reasons:
            continue
        intr = per_intra.get(t)
        spark = list(intr["Close"].dropna().values) if intr is not None else []
        results.append({
            "ticker": t,
            "price": live_prices.get(t, float(per_daily[t]["Close"].iloc[-1])),
            "chg": day_chg.get(t, 0.0),
            "score": score, "reasons": reasons,
            "watchlist": t in WATCHLIST,
            "sector": sec or ("Watchlist" if t in WATCHLIST else ""),
            "alerted": t in state["alerted"],
            "spark": spark, "option": None,
        })
    results.sort(key=lambda r: r["score"], reverse=True)
    return results


def alert_new_hits(results, state) -> bool:
    hits = []
    for r in results:
        if r["ticker"] in state["alerted"]:
            continue
        threshold = WATCHLIST_THRESHOLD if r["watchlist"] else SCORE_THRESHOLD
        if r["score"] >= threshold:
            hits.append(r)
    hits = hits[:MAX_ALERTS_PER_RUN]
    if not hits:
        return False
    for r in hits:
        if SUGGEST_OPTIONS:
            r["option"] = suggest_contract(r["ticker"], r["price"])
        star = "*" if r["watchlist"] else ""
        line = (f"BUY {r['ticker']}{star} ${r['price']:.2f} "
                f"({r['chg']:+.1f}%) [{r['score']}] "
                + "; ".join(r["reasons"][:2]))
        opt = fmt_option(r.get("option"))
        if opt:
            line += " | " + opt
        send_sms(line)
        state["alerted"].append(r["ticker"])
        r["alerted"] = True
    return True


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    force = os.environ.get("FORCE_RUN") == "1"
    if not force and not market_is_open():
        print("Market closed — exiting.")
        return

    state = load_state()
    pos_tickers = [p["ticker"] for p in POSITIONS]
    fast_set = sorted(set(WATCHLIST) | set(pos_tickers))

    # ---------- FULL SCAN ----------
    sp500 = get_sp500()
    sector_map = dict(zip(sp500["ticker"], sp500["sector"]))
    universe = sorted(set(sp500["ticker"]) | set(fast_set))
    print(f"Full scan: {len(universe)} tickers...")

    per_daily = split_by_ticker(download_daily(universe), universe)
    per_intra = split_by_ticker(download_intraday(universe), universe)

    day_chg, live = {}, {}
    for t, d in per_daily.items():
        if len(d) >= 2:
            p = float(d["Close"].iloc[-1])
            live[t] = p
            day_chg[t] = (p / float(d["Close"].iloc[-2]) - 1) * 100

    sector_chg = (pd.Series(day_chg).rename("chg").to_frame()
                  .join(pd.Series(sector_map).rename("sector"))
                  .groupby("sector")["chg"].mean().to_dict())

    results = build_results(universe, per_daily, per_intra, sector_map,
                            sector_chg, day_chg, live, state)
    alert_new_hits(results, state)
    for line in check_exits(per_daily, live, per_intra, state):
        send_sms(line)
    write_dashboard(results, state)
    save_state(state)
    publish()

    # ---------- FAST LOOP: watchlist + positions every 2 min ----------
    if force:
        return
    deadline = time.time() + RUN_LOOP_MINUTES * 60
    while time.time() + FAST_INTERVAL_SEC <= deadline and market_is_open():
        time.sleep(FAST_INTERVAL_SEC)
        try:
            fresh = split_by_ticker(download_intraday(fast_set, "5m"), fast_set)
        except Exception as e:
            print(f"fast download failed: {e}")
            continue
        for t, d in fresh.items():
            per_intra[t] = d
            if len(d) and t in per_daily and len(per_daily[t]) >= 2:
                p = float(d["Close"].iloc[-1])
                live[t] = p
                day_chg[t] = (p / float(per_daily[t]["Close"].iloc[-2]) - 1) * 100
        fast_results = build_results(fast_set, per_daily, per_intra,
                                     sector_map, sector_chg, day_chg,
                                     live, state)
        changed = alert_new_hits(fast_results, state)
        exit_lines = check_exits(per_daily, live, per_intra, state)
        for line in exit_lines:
            send_sms(line)
        if changed or exit_lines:
            # merge fast rows into the big board and republish
            by_t = {r["ticker"]: r for r in results}
            for r in fast_results:
                by_t[r["ticker"]] = r
            results = sorted(by_t.values(), key=lambda r: r["score"],
                             reverse=True)
            write_dashboard(results, state)
            save_state(state)
            publish()
        print(f"fast pass done {datetime.now(ET).strftime('%H:%M:%S')}")


if __name__ == "__main__":
      main()
