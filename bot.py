import os
import sys
import json
import time
import logging
import base64
from datetime import date
from zoneinfo import ZoneInfo
from typing import Tuple, List, Dict, Any, Optional

import requests
import yfinance as yf
from google import genai
from google.genai import types

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


class Config:
    """Handles environment variables and configuration."""

    def __init__(self):
        self.telegram_token = os.environ.get("TELEGRAM_TOKEN")
        self.telegram_chat_id = os.environ.get("TELEGRAM_CHAT_ID")
        self.t212_key = os.environ.get("TRADING212_API_KEY")
        self.t212_secret = os.environ.get("TRADING212_API_SECRET")
        self.gemini_key = os.environ.get("GEMINI_API_KEY")
        self.finnhub_key = os.environ.get("FINNHUB_KEY")
        self.action_type = os.environ.get("ACTION_TYPE", "analiz")
        self.timezone = ZoneInfo("America/New_York")

    def validate(self):
        required = {
            "TELEGRAM_TOKEN": self.telegram_token,
            "TELEGRAM_CHAT_ID": self.telegram_chat_id,
            "TRADING212_API_KEY": self.t212_key,
            "TRADING212_API_SECRET": self.t212_secret,
            "GEMINI_API_KEY": self.gemini_key,
        }
        for name, val in required.items():
            if not val:
                log.error(f"Missing required environment variable: {name}")
                sys.exit(1)
        if not self.finnhub_key:
            log.warning("Optional environment variable FINNHUB_KEY is missing. Live news will be disabled.")


def retry(fn, retries=3, backoff=2.0):
    """Utility to retry API calls with exponential backoff."""
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_error = exc
            if attempt >= retries:
                log.error(f"Attempt {attempt}/{retries} failed. No retries left.")
                raise
            wait_seconds = backoff ** attempt
            log.warning(f"Attempt {attempt}/{retries} failed: {exc}. Retrying in {wait_seconds:.2f}s.")
            time.sleep(wait_seconds)
    raise last_error


class Utils:
    """Common parsing and formatting utilities."""
    @staticmethod
    def normalize_ticker(raw: str) -> str:
        return raw.split("_")[0] if "_" in raw else raw

    @staticmethod
    def safe_float(value: Any, fallback: float = 0.0) -> float:
        if value is None:
            return fallback
        try:
            return float(value)
        except (ValueError, TypeError):
            return fallback

    @staticmethod
    def safe_pnl_pct(current: float, average: float) -> float:
        if not average:
            return 0.0
        return round(((current - average) / average) * 100, 2)


class Trading212Client:
    """Client for Trading212 unofficial API integrations."""
    BASE_URL = "https://live.trading212.com/api/v0"

    def __init__(self, key: str, secret: str):
        self.key = key
        self.secret = secret

    def _get_auth_header(self) -> str:
        encoded = base64.b64encode(f"{self.key}:{self.secret}".encode("utf-8")).decode("utf-8")
        return f"Basic {encoded}"

    def _request(self, endpoint: str) -> requests.Response:
        url = f"{self.BASE_URL}/{endpoint}"
        headers = {"Authorization": self._get_auth_header()}

        def _fetch():
            res = requests.get(url, headers=headers, timeout=10)
            if res.status_code == 401:
                log.error(f"T212 auth failed for {endpoint}. Check API keys.")
                raise RuntimeError("Trading212 authentication failed (401)")
            if res.status_code == 429:
                log.warning("T212 rate limit hit (429). Exceeded 1 req/s.")
                raise RuntimeError("Trading212 rate limit hit (429)")
            return res

        response = retry(_fetch)
        time.sleep(1)  # Rate limit protection
        return response

    def get_portfolio(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Returns structured positions, and raw API positions for inference."""
        res = self._request("equity/positions")
        if res.status_code != 200:
            log.error(f"T212 positions request failed: {res.status_code}")
            raise RuntimeError(f"Positions failed: {res.status_code}")
        
        raw_positions = res.json()
        positions = []
        for item in raw_positions:
            instrument = item.get("instrument", {})
            raw_ticker = instrument.get("ticker", "UNKNOWN")
            avg = Utils.safe_float(item.get("averagePricePaid"))
            current = Utils.safe_float(item.get("currentPrice"))
            qty = Utils.safe_float(item.get("quantity"))
            wallet = item.get("walletImpact", {})
            pnl = Utils.safe_float(wallet.get("unrealizedProfitLoss"))

            positions.append({
                "ticker": Utils.normalize_ticker(raw_ticker),
                "quantity": qty,
                "averagePrice": avg,
                "currentPrice": current,
                "unrealizedPnL": pnl,
                "unrealizedPnLPct": Utils.safe_pnl_pct(current, avg),
            })
        
        log.info(f"Loaded {len(positions)} positions from Trading212.")
        return positions, raw_positions

    def get_account_cash(self) -> Dict[str, Any]:
        """Fetch account cash summary."""
        res = self._request("equity/account/cash")
        if res.status_code == 200:
            return res.json()
        log.warning(f"T212 account/cash request failed: {res.status_code}")
        return {}

    def get_orders(self, raw_positions: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """Fetch pending orders, falling back to inference if 403."""
        try:
            res = self._request("equity/orders")
            if res.status_code == 200:
                orders = []
                for item in res.json():
                    if item.get("status") in ["FILLED", "CANCELLED"]:
                        continue
                    orders.append({
                        "ticker": Utils.normalize_ticker(item.get("ticker", "UNKNOWN")),
                        "type": item.get("type"),
                        "action": item.get("side", item.get("action")),
                        "quantity": Utils.safe_float(item.get("quantity")),
                        "limitPrice": Utils.safe_float(item.get("limitPrice")),
                        "stopPrice": Utils.safe_float(item.get("stopPrice")),
                        "status": item.get("status"),
                    })
                log.info(f"Loaded {len(orders)} pending orders directly.")
                return orders
            log.warning(f"/equity/orders returned {res.status_code}. Using inference fallback.")
        except RuntimeError as e:
            if "401" in str(e):
                raise  # Hard fail on auth
            log.warning("Orders request errored. Falling back to inference.")

        # --- Inference Fallback ---
        orders = []
        if raw_positions:
            for item in raw_positions:
                instrument = item.get("instrument", {})
                raw_ticker = instrument.get("ticker", "UNKNOWN")
                qty = Utils.safe_float(item.get("quantity"))
                available = Utils.safe_float(item.get("quantityAvailableForTrading"))
                locked = round(qty - available, 8)

                if locked > 0.0001:
                    ticker = Utils.normalize_ticker(raw_ticker)
                    orders.append({
                        "ticker": ticker,
                        "type": "LIMIT (inferred)",
                        "action": "SELL",
                        "quantity": locked,
                        "limitPrice": None,
                        "stopPrice": None,
                        "status": "PENDING (locked)",
                        "note": f"Full qty={qty}, available={available}, locked={locked}",
                    })
                    log.info(f"Inferred pending sell: {ticker} (locked {locked})")

        cash = self.get_account_cash()
        blocked = Utils.safe_float(cash.get("blocked"))
        if blocked > 0:
            orders.append({
                "ticker": "CASH_RESERVED",
                "type": "BUY LIMIT (inferred)",
                "action": "BUY",
                "quantity": None,
                "limitPrice": None,
                "stopPrice": None,
                "status": "PENDING",
                "blockedCash": blocked,
                "note": f"€{blocked:.2f} reserved for pending buy limit orders",
            })
            log.info(f"Detected €{blocked:.2f} reserved cash for buy orders.")

        log.info(f"Inferred {len(orders)} pending orders total.")
        return orders


class MarketDataClient:
    """Handles external news and technicals integration."""
    def __init__(self, finnhub_key: Optional[str]):
        self.finnhub_key = finnhub_key

    def get_news(self) -> str:
        if not self.finnhub_key:
            return "No live news available. Perform general market analysis only."
        
        def _fetch():
            res = requests.get(
                "https://finnhub.io/api/v1/news",
                params={"category": "general", "token": self.finnhub_key},
                timeout=10,
            )
            res.raise_for_status()
            return res
        
        try:
            res = retry(_fetch)
            headlines = [item.get("headline", "").strip() for item in res.json() if item.get("headline", "").strip()]
            headlines = headlines[:15]
            log.info(f"Collected {len(headlines)} market headlines from Finnhub.")
            return "\n".join(f"- {h}" for h in headlines)
        except Exception as e:
            log.warning(f"Finnhub news fetch failed: {e}")
            return "News unavailable today."

    def get_technicals(self, ticker: str) -> Dict[str, Any]:
        if not ticker or ticker == "CASH_RESERVED":
            return {}

        try:
            # Silence massive yfinance logger spam
            tkr = yf.Ticker(ticker)
            hist = tkr.history(period="3mo", auto_adjust=False)
            if hist.empty:
                log.info(f"{ticker}: No moving average data (history is empty).")
                return {}

            support_series = hist["Low"].rolling(20).min().dropna()
            resistance_series = hist["High"].rolling(20).max().dropna()

            support = round(float(support_series.iloc[-1]), 2) if not support_series.empty else None
            resistance = round(float(resistance_series.iloc[-1]), 2) if not resistance_series.empty else None

            vol_trend = "normal"
            vol_series = hist["Volume"].dropna()
            if not vol_series.empty:
                avg_vol = vol_series.mean()
                if vol_series.iloc[-1] < avg_vol * 0.7:
                    vol_trend = "declining"

            return {"support": support, "resistance": resistance, "volume_trend": vol_trend}
        except Exception as e:
            # yfinance often throws json.decoder.JSONDecodeError inside pandas
            log.debug(f"Failed to compute technicals for {ticker}. Error: {e}")
            return {}


class AIAnalyst:
    """Google Gemini orchestrator to generate broker briefs."""
    SYSTEM_INSTRUCTION = """You are a professional Wall Street equity analyst and personal portfolio manager
with 20+ years of experience across NYSE, NASDAQ, and international markets.

CONTEXT
You will receive three data inputs at runtime:
  [1] PORTFOLIO_DATA   — JSON array: ticker, quantity, averagePrice,
      currentPrice, unrealizedPnL, unrealizedPnLPct
  [2] MARKET_NEWS      — Today's top US equity headlines
  [3] PENDING_ORDERS   — JSON array of detected pending orders:
      - Sell orders: inferred from locked positions (quantity locked
        for trading). Fields: ticker, action=SELL, quantity (locked amount),
        note with full/available/locked breakdown.
      - Buy orders: detected via blocked cash in account. Fields:
        ticker=CASH_RESERVED, action=BUY, blockedCash (EUR amount reserved).
      - If the direct API is available: ticker, type, action, quantity,
        limitPrice, stopPrice, status.

YOUR TASK
1. Cross-reference each holding against today's news
2. Identify catalysts (positive/negative) for those tickers
3. Provide short-term price outlook with key levels
4. Issue clear signals: HOLD / WATCH / TRIM / AVOID
5. Review PENDING_ORDERS and comment on their validity based on key technical levels

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

⏳ *Pending Orders*
[For each locked sell order:]
`[TICKER]` — SELL [qty] shares locked
• Assessment: [one line on whether the limit is well-placed vs. key levels]

[If blocked cash exists:]
💰 €[amount] reserved for pending BUY limit orders

⚠️ _AI-generated analysis. Not financial advice. Always DYOR._

R7 TONE & LENGTH
   Senior broker voice. Active tense. No filler. Max 3800 characters.
   If too long: prioritize impacted tickers, shorten clean list.

R8 HALLUCINATION GUARD
   Never fabricate analysis for tickers not mentioned in today's news.
   Mark them clean.

SIGNAL EMOJI LEGEND
🟢 HOLD    — healthy, no action needed
🟡 WATCH   — monitor closely, move incoming
🔴 TRIM    — reduce position size
⛔ AVOID   — strong negative catalyst, reassess thesis
⏳ PENDING — open limit/stop order; assess vs. key levels"""

    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        # Using 2.5-flash as the older 1.5 model is deprecated and returns 404.
        self.model_name = "gemini-2.5-flash"

    def analyze(self, portfolio: List, orders: List, news: str, technicals: Dict) -> str:
        prompt = f"""DATE: {date.today().strftime("%B %d, %Y")}

PORTFOLIO_DATA:
{json.dumps(portfolio, indent=2)}

PENDING_ORDERS:
{json.dumps(orders, indent=2)}

TECHNICALS:
{json.dumps(technicals, indent=2)}

MARKET_NEWS:
{news}

Generate the daily broker brief now."""

        def _generate():
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=self.SYSTEM_INSTRUCTION,
                )
            )
            return response

        try:
            res = retry(_generate)
            text = getattr(res, "text", "").strip()
            if not text:
                return "⚠️ Gemini returned an empty response. Markets may be unusually quiet."
            log.info(f"AI analysis generated ({len(text)} chars).")
            return text
        except Exception as e:
            log.error(f"AI generation failed: {e}")
            raise


class TelegramNotifier:
    """Sends broadcast and JSON payloads via Telegram Bot API."""
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{self.token}/sendMessage"

    def send_text(self, text: str, parse_mode: str = "Markdown") -> None:
        max_len = 4000
        suffix = "\n\n_[truncated]_"
        if len(text) > max_len:
            text = text[:max_len - len(suffix)] + suffix

        def _post(body):
            def _send():
                return requests.post(self.base_url, json=body, timeout=10)
            return retry(_send)

        body = {"chat_id": self.chat_id, "text": text, "parse_mode": parse_mode}
        res = _post(body)
        
        # Fallback if markdown parsing fails
        if res.status_code == 400 and ("parse" in res.text.lower() or "entities" in res.text.lower()):
            log.warning("Markdown parse error. Sending plain text.")
            del body["parse_mode"]
            res = requests.post(self.base_url, json=body, timeout=10)
        
        res.raise_for_status()
        log.info(f"Telegram text sent ({len(text)} chars).")

    def send_json(self, data: Any) -> None:
        json_str = json.dumps(data, indent=2, ensure_ascii=False)
        max_chunk = 3700
        chunks = []
        current_chunk = ""
        for line in json_str.splitlines(keepends=True):
            if len(current_chunk) + len(line) <= max_chunk:
                current_chunk += line
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = line
        if current_chunk:
            chunks.append(current_chunk)
        if not chunks:
            chunks = ["{}"]

        for c in chunks:
            self.send_text(f"```json\n{c}\n```")
            time.sleep(1)
        log.info(f"Sent JSON in {len(chunks)} chunks.")


class AIBrokerBot:
    """Main orchestration application."""
    def __init__(self):
        self.cfg = Config()
        self.cfg.validate()
        
        self.t212 = Trading212Client(self.cfg.t212_key, self.cfg.t212_secret)
        self.market = MarketDataClient(self.cfg.finnhub_key)
        self.ai = AIAnalyst(self.cfg.gemini_key)
        self.notifier = TelegramNotifier(self.cfg.telegram_token, self.cfg.telegram_chat_id)

    def run(self):
        log.info(f"Starting AI Broker v2 — mode: {self.cfg.action_type}")
        try:
            portfolio, raw = self.t212.get_portfolio()
            orders = self.t212.get_orders(raw)
        except Exception as e:
            self.notifier.send_text(f"⚠️ *Trading212 fetch failed*\n`{e}`")
            return

        if not portfolio and not orders:
            self.notifier.send_text("⚠️ No open positions or pending orders found.")
            return

        if self.cfg.action_type == "portfoy_json":
            self.notifier.send_json({"positions": portfolio, "orders": orders})
            return

        # Gather context
        news = self.market.get_news()
        technicals = {}
        
        # Consolidate tickers for technicals
        tickers = {p["ticker"] for p in portfolio} | {o["ticker"] for o in orders}
        for ticker in tickers:
            technicals[ticker] = self.market.get_technicals(ticker)

        # Analyze and broadcast
        try:
            analysis = self.ai.analyze(portfolio, orders, news, technicals)
            self.notifier.send_text(analysis)
        except Exception as e:
            self.notifier.send_text(f"⚠️ *Gemini analysis failed*\n`{e}`")

if __name__ == "__main__":
    AIBrokerBot().run()
