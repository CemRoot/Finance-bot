import os
import sys
import json
import time
import logging
import base64
from datetime import date
from zoneinfo import ZoneInfo
import requests
import yfinance as yf
import google.generativeai as genai


SYSTEM_PROMPT = """You are a professional Wall Street equity analyst and personal portfolio manager
with 20+ years of experience across NYSE, NASDAQ, and international markets.

CONTEXT
You will receive two data inputs at runtime:
  [1] PORTFOLIO_DATA — JSON array: ticker, quantity, averagePrice,
      currentPrice, unrealizedPnL, unrealizedPnLPct
  [2] MARKET_NEWS    — Today's top US equity headlines

YOUR TASK
1. Cross-reference each holding against today's news
2. Identify catalysts (positive/negative) for those tickers
3. Provide short-term price outlook with key levels
4. Issue clear signals: HOLD / WATCH / TRIM / AVOID

RULES
R1 NEWS-PORTFOLIO LINKAGE
   Map headlines to tickers. Skip headlines with no portfolio impact.

R2 PRICE IMPACT
   Per impacted ticker:
   - Direction: bullish / bearish / neutral
   - Magnitude: minor <2% / moderate 2-5% / major >5%
   - Confidence: Low / Medium / High + one-line rationale

R3 KEY LEVELS
   Support, resistance, optional trim/entry zone. Impacted tickers only.

R4 SATURATION SIGNAL
   - unrealizedPnLPct > 15% AND news neutral/negative → TRIM CANDIDATE
   - Price near 52w high AND volume declining         → DISTRIBUTION ZONE

R5 OPPORTUNITY SIGNAL
   - unrealizedPnLPct < -10% AND fundamentals intact  → ACCUMULATE ZONE
   - Price at known support                           → WATCH LEVEL

R6 STRICT OUTPUT FORMAT (Telegram Markdown)
Output EXACTLY this structure and nothing else:

🗓 *Daily Broker Brief — [DATE]*
━━━━━━━━━━━━━━━━━━━━━━
📰 *Market Pulse*
[2-3 sentence macro summary]

📊 *Portfolio Impact Analysis*

`[TICKER]` — [emoji] *[SIGNAL]*
• Catalyst: [one line]
• Outlook: [one line]
• Levels: S [x.xx] / R [x.xx]
• Confidence: [Low/Med/High] — [reason]

✅ *Clean Positions*
[comma-separated tickers with no news impact today]

⚠️ _AI-generated analysis. Not financial advice. Always DYOR._

R7 TONE & LENGTH
   Senior broker voice. Active tense. No filler. Max 3800 characters.
   If too long: prioritize impacted tickers, shorten clean list.

R8 HALLUCINATION GUARD
   Never fabricate analysis for tickers not mentioned in today's news.
   Mark them clean.

SIGNAL EMOJI LEGEND
🟢 HOLD  — healthy, no action needed
🟡 WATCH — monitor closely, move incoming
🔴 TRIM  — reduce position size
⛔ AVOID — strong negative catalyst, reassess thesis"""


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


REQUIRED_ENV = [
    "TELEGRAM_TOKEN",
    "TELEGRAM_CHAT_ID",
    "TRADING212_API_KEY",
    "TRADING212_API_SECRET",
    "GEMINI_API_KEY",
]

for env_name in REQUIRED_ENV:
    if not os.environ.get(env_name):
        log.error("Missing required environment variable: %s", env_name)
        sys.exit(1)

if not os.environ.get("FINNHUB_KEY"):
    log.warning("Optional environment variable FINNHUB_KEY is missing. Live news will be disabled.")

TOKEN = os.environ["TELEGRAM_TOKEN"]
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
ACTION_TYPE = os.environ.get("ACTION_TYPE", "analiz")
NY_TZ = ZoneInfo("America/New_York")


genai.configure(api_key=os.environ["GEMINI_API_KEY"])

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    system_instruction=SYSTEM_PROMPT,
)


def retry(fn, retries=3, backoff=2.0):
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_error = exc
            if attempt >= retries:
                log.error("Attempt %s/%s failed: %s. No retries left.", attempt, retries, exc)
                raise
            wait_seconds = backoff ** attempt
            log.warning(
                "Attempt %s/%s failed: %s. Retrying in %.2f seconds.",
                attempt,
                retries,
                exc,
                wait_seconds,
            )
            time.sleep(wait_seconds)
    raise last_error


def build_t212_auth_header() -> str:
    key = os.environ["TRADING212_API_KEY"]
    secret = os.environ["TRADING212_API_SECRET"]
    encoded = base64.b64encode(f"{key}:{secret}".encode("utf-8")).decode("utf-8")
    return f"Basic {encoded}"


def normalize_ticker(raw: str) -> str:
    return raw.split("_")[0] if "_" in raw else raw


def safe_float(value, fallback: float = 0.0) -> float:
    if value is None:
        return fallback
    try:
        return float(value)
    except Exception:
        return fallback


def safe_pnl_pct(current: float, average: float) -> float:
    if not average:
        return 0.0
    return round(((current - average) / average) * 100, 2)


def get_portfolio() -> list[dict]:
    url = "https://live.trading212.com/api/v0/equity/positions"
    headers = {"Authorization": build_t212_auth_header()}

    def _fetch():
        response = None
        try:
            response = requests.get(url, headers=headers, timeout=10)
            return response
        finally:
            time.sleep(1)

    response = retry(_fetch)

    if response.status_code == 401:
        log.error(
            "T212 auth failed (401). Check TRADING212_API_KEY and TRADING212_API_SECRET. Must be Basic base64(key:secret)"
        )
        raise RuntimeError("Trading212 authentication failed (401)")

    if response.status_code == 429:
        log.warning("T212 rate limit hit (429). Exceeded 1 req/s. Backing off...")
        raise RuntimeError("Trading212 rate limit hit (429)")

    if response.status_code != 200:
        log.error("Trading212 positions request failed with status %s", response.status_code)
        raise RuntimeError(f"Trading212 request failed with status {response.status_code}")

    payload = response.json()
    positions = []

    for item in payload:
        instrument = item.get("instrument") or {}
        raw_ticker = instrument.get("ticker", "UNKNOWN")
        avg = safe_float(item.get("averagePricePaid"))
        current = safe_float(item.get("currentPrice"))
        qty = safe_float(item.get("quantity"))
        wallet = item.get("walletImpact") or {}
        pnl = safe_float(wallet.get("unrealizedProfitLoss"))

        positions.append(
            {
                "ticker": normalize_ticker(raw_ticker),
                "quantity": qty,
                "averagePrice": avg,
                "currentPrice": current,
                "unrealizedPnL": pnl,
                "unrealizedPnLPct": safe_pnl_pct(current, avg),
            }
        )

    log.info("Loaded %s positions from Trading212.", len(positions))
    return positions


def get_market_news() -> str:
    finnhub_key = os.environ.get("FINNHUB_KEY")
    if not finnhub_key:
        return "No live news available. Perform general market analysis only."

    url = "https://finnhub.io/api/v1/news"

    def _fetch():
        return requests.get(
            url,
            params={"category": "general", "token": finnhub_key},
            timeout=10,
        )

    response = retry(_fetch)
    response.raise_for_status()

    headlines = []
    for item in response.json():
        headline = item.get("headline", "").strip()
        if not headline:
            continue
        headlines.append(f"- {headline}")
        if len(headlines) == 15:
            break

    log.info("Collected %s market headlines from Finnhub.", len(headlines))
    return "\n".join(headlines)


def get_technicals(ticker: str) -> dict:
    try:
        hist = yf.Ticker(ticker).history(period="3mo")
        if hist.empty:
            log.warning("No yfinance history returned for %s", ticker)
            return {}

        support_series = hist["Low"].rolling(20).min().dropna()
        resistance_series = hist["High"].rolling(20).max().dropna()

        support = round(float(support_series.iloc[-1]), 2) if not support_series.empty else None
        resistance = round(float(resistance_series.iloc[-1]), 2) if not resistance_series.empty else None

        volume_series = hist["Volume"].dropna()
        if volume_series.empty:
            vol_trend = "normal"
        else:
            avg_vol = volume_series.mean()
            last_vol = volume_series.iloc[-1]
            vol_trend = "declining" if last_vol < avg_vol * 0.7 else "normal"

        return {"support": support, "resistance": resistance, "volume_trend": vol_trend}
    except Exception as exc:
        log.warning("Failed to compute technicals for %s: %s", ticker, exc)
        return {}


def build_user_prompt(portfolio: list, news: str, technicals: dict) -> str:
    return f"""DATE: {date.today().strftime("%B %d, %Y")}

PORTFOLIO_DATA:
{json.dumps(portfolio, indent=2)}

TECHNICALS:
{json.dumps(technicals, indent=2)}

MARKET_NEWS:
{news}

Generate the daily broker brief now."""


def get_analysis(portfolio: list, news: str, technicals: dict) -> str:
    def _generate():
        return model.generate_content(build_user_prompt(portfolio, news, technicals))

    response = retry(_generate)
    text = (getattr(response, "text", "") or "").strip()
    if not text:
        return "⚠️ Gemini returned an empty response. Markets may be unusually quiet today."

    log.info("Gemini analysis generated (%s chars).", len(text))
    return text


def send_telegram(text: str, parse_mode: str = "Markdown") -> None:
    suffix = "\n\n_[truncated]_"
    max_len = 4000
    if len(text) > max_len:
        text = text[: max_len - len(suffix)] + suffix

    body = {"chat_id": CHAT_ID, "text": text, "parse_mode": parse_mode}
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

    def _post(current_body):
        def _send():
            return requests.post(url, json=current_body, timeout=10)

        return retry(_send)

    response = _post(body)
    lower_body = (response.text or "").lower()

    if response.status_code == 400 and (
        "parse" in lower_body or "can't parse entities" in lower_body
    ):
        log.warning("Telegram parse_mode error, retrying without parse_mode: %s", response.text)
        body_no_parse = {"chat_id": CHAT_ID, "text": text}
        response = requests.post(url, json=body_no_parse, timeout=10)
        if response.status_code >= 400:
            log.error("Telegram fallback send failed with status %s: %s", response.status_code, response.text)

    response.raise_for_status()
    log.info("Telegram message sent (%s chars).", len(text))


def send_telegram_chunked_json(data: list) -> None:
    json_str = json.dumps(data, indent=2, ensure_ascii=False)
    max_chunk = 3700
    chunks = []
    current_chunk = ""
    for line in json_str.splitlines(keepends=True):
        if len(current_chunk) + len(line) <= max_chunk:
            current_chunk += line
            continue
        if current_chunk:
            chunks.append(current_chunk)
        if len(line) <= max_chunk:
            current_chunk = line
            continue
        for i in range(0, len(line), max_chunk):
            chunks.append(line[i : i + max_chunk])
        current_chunk = ""
    if current_chunk:
        chunks.append(current_chunk)
    if not chunks:
        chunks = [""]

    for chunk in chunks:
        send_telegram(f"```json\n{chunk}\n```")
        time.sleep(1)

    log.info("Sent %s JSON chunk(s) to Telegram.", len(chunks))


def main():
    log.info("Starting AI Broker Bot — action=%s", ACTION_TYPE)

    try:
        portfolio = get_portfolio()
    except Exception as e:
        send_telegram(f"⚠️ *Trading212 fetch failed*\n`{e}`")
        return

    if not portfolio:
        send_telegram("⚠️ Portfolio is empty or no open positions found.")
        return

    if ACTION_TYPE == "portfoy_json":
        send_telegram_chunked_json(portfolio)
        return

    try:
        news = get_market_news()
    except Exception as e:
        log.warning("News fetch failed: %s", e)
        news = "News unavailable today."

    technicals = {}
    for pos in portfolio:
        result = get_technicals(pos["ticker"])
        if result:
            technicals[pos["ticker"]] = result

    try:
        analysis = get_analysis(portfolio, news, technicals)
    except Exception as e:
        send_telegram(f"⚠️ *Gemini analysis failed*\n`{e}`")
        return

    send_telegram(analysis)
    log.info("Bot run complete.")


if __name__ == "__main__":
    main()
