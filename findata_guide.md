# findata API — usage guide

Base URL: `https://lum.id/findata`  ·  Data routes are PAT-authenticated (an invalid token returns 401; anonymous requests ride an anon-read tier).

## Start here
New to the API? Open these in a browser (no auth needed):
- **`https://lum.id/findata/`** — landing page: quickstart, auth, and the endpoint map.
- **`https://lum.id/findata/reference`** — browsable API reference (ReDoc, generated from the live spec).
- **`https://lum.id/findata/openapi.json`** — OpenAPI 3.1 spec (import into Postman / generate a client).
- **`https://lum.id/findata/llm`** — LLM proxy quickstart.
- **`https://lum.id/findata/status`** / **`/usage`** — health board + global usage dashboard.

Then get a token (a Lumid PAT) and call any data route with `Authorization: Bearer <token>`.

## Auth
Send a bearer token on every data route:
```
Authorization: Bearer <token>
```
- A **Lumid PAT** (`lm_pat_live_…`) or an internal **local key** (`LUMID_API_KEYS`).
- Invalid → `401`. Over rate limit → `429` with `Retry-After`.
- Rate-limit headers on every response: `x-ratelimit-limit`, `x-ratelimit-remaining`, `x-ratelimit-reset`.
- **Public (no auth):** `/`, `/reference`, `/openapi.json`, `/llm`, `/usage.md`, `/skill.md`, `/status`, `/usage`, `/freshness`, `/health`, `/docs`, `/redoc`.

## Conventions
Date/time filters (`start`/`end`, `from`/`to`, `since`/`until`) accept **RFC3339** (`2026-05-16T00:00:00Z`) or bare **`YYYY-MM-DD`** (→ `00:00:00 UTC`). Comma-separated list params (`symbols=`, `ticker=`) take multiple values in one call. Symbols are upper-cased server-side. Responses carry `ETag` + `Cache-Control`; send `If-None-Match` for `304`.

```bash
H='Authorization: Bearer <token>'
```

---

## Prices & OHLC

### Live quotes
```bash
curl -H "$H" "https://lum.id/findata/quotes?symbols=AAPL,MSFT,BTCUSD"
```
Returns the live streaming tick when available (`source` starts with `tier_a:`, `stale:false`). Falls back to the last stored bar when the feed is down or market is closed (`stale:true`, `source:"prev_close"` or `source:"last_bar"`). Only symbols with no tick *and* no bar return `price:null, source:"no_cache"`.

```bash
curl -H "$H" "https://lum.id/findata/quotes/stream?symbols=AAPL,BTCUSD"   # SSE: subscribed, tick, heartbeat
curl -H "$H" "https://lum.id/findata/quote-stats/AAPL"                    # 52w high/low, day range, SMAs
```

### OHLC bars
`/ohlc/:symbol?interval=<1min|5min|15min|30min|1hour|4hour|1d>&from=&to=`

Valid intervals: `1min 5min 15min 30min 1hour 4hour 1d`. `daily`/`day`/`1d` variants other than exactly `1d` are **invalid**.
```bash
curl -H "$H" "https://lum.id/findata/ohlc/AAPL?interval=1d&from=2025-12-01&to=2026-02-01"
curl -H "$H" "https://lum.id/findata/ohlc/BTCUSD?interval=1h"     # non-equity rolled up from 1-min on the fly
curl -H "$H" "https://lum.id/findata/ohlc/EURUSD?interval=5min"   # forex
```
Returns `{symbol, interval, count, bars:[{ts,o,h,l,c,v},...]}`. `from`/`to` and `start`/`end` are interchangeable; omit for a default trailing window.

US equity OHLC bars (all intervals) are sourced from two independent feeds. The server merges them automatically; coverage continues if one feed has a gap (e.g. bandwidth exhaustion on the primary). Non-equity bars (forex, crypto, indices) use the primary feed only.

### Technical indicators
```bash
curl -H "$H" "https://lum.id/findata/technical/AAPL"           # all indicators, latest values
curl -H "$H" "https://lum.id/findata/technical/AAPL/latest"    # same, single-row snapshot
```
Indicators (SMA, EMA, RSI, MACD) are drawn from two independent data sources. Multi-source coverage lets you cross-validate and provides continuity when one source has a data gap.

### Market movers & sector/industry snapshots
```bash
curl -H "$H" "https://lum.id/findata/market-movers"                  # gainers/losers/most-active
curl -H "$H" "https://lum.id/findata/market-cap/AAPL/history"        # historical market-cap series
curl -H "$H" "https://lum.id/findata/sectors/pe"                     # sector P/E snapshot
curl -H "$H" "https://lum.id/findata/sectors/performance"            # sector % performance
curl -H "$H" "https://lum.id/findata/industries/pe"
curl -H "$H" "https://lum.id/findata/industries/performance"
```

---

## Symbols & universe

```bash
curl -H "$H" "https://lum.id/findata/symbols?ticker=AAPL,MSFT,SPY,BTCUSD"  # batch metadata, 1 call (max 500)
curl -H "$H" "https://lum.id/findata/symbols/AAPL"                          # single symbol metadata
curl -H "$H" "https://lum.id/findata/symbols/search?q=apple"
curl -H "$H" "https://lum.id/findata/universe"                              # full 7,851-symbol roster
curl -H "$H" "https://lum.id/findata/universe/actively-trading"             # currently active subset
curl -H "$H" "https://lum.id/findata/screener?sector=Technology&marketCapMoreThan=1e12&limit=50"
```
Screener params: `sector`, `industry`, `exchange`, `country`, `marketCapMoreThan`/`marketCapLessThan`, `isEtf`, `isFund`, `limit`, `offset`.

---

## Fundamentals

```bash
# Latest wide-format row (all statements merged)
curl -H "$H" "https://lum.id/findata/fundamentals/AAPL/latest"

# Historical statements — statement ∈ income|balance|cashflow
curl -H "$H" "https://lum.id/findata/fundamentals/AAPL/history?statement=income&period=quarter&limit=40"
curl -H "$H" "https://lum.id/findata/fundamentals/AAPL/history?statement=balance&period=annual"
```

---

## Analysis

```bash
curl -H "$H" "https://lum.id/findata/ratios/AAPL"                   # P/E, P/B, ROE, margins, …
curl -H "$H" "https://lum.id/findata/key-metrics/AAPL"              # EV/EBITDA, FCF yield, ROIC, …
curl -H "$H" "https://lum.id/findata/metrics-snapshot/AAPL"         # point-in-time: 52w stats, beta, returns
curl -H "$H" "https://lum.id/findata/financial-scores/AAPL"         # Altman Z, Piotroski F, custom scores
curl -H "$H" "https://lum.id/findata/earnings-quality/AAPL"         # accruals, cash conversion
curl -H "$H" "https://lum.id/findata/enterprise-value/AAPL"         # EV series
curl -H "$H" "https://lum.id/findata/owner-earnings/AAPL"           # Buffett-style owner earnings
curl -H "$H" "https://lum.id/findata/dcf/AAPL"                      # discounted cash-flow model
curl -H "$H" "https://lum.id/findata/financial-growth/AAPL"         # revenue/EPS/FCF growth rates
curl -H "$H" "https://lum.id/findata/income-statement-growth/AAPL"
curl -H "$H" "https://lum.id/findata/balance-sheet-growth/AAPL"
curl -H "$H" "https://lum.id/findata/cash-flow-growth/AAPL"
```

---

## Estimates

```bash
curl -H "$H" "https://lum.id/findata/grades/AAPL"                         # analyst buy/hold/sell grades
curl -H "$H" "https://lum.id/findata/estimates/AAPL/price-target"         # price target history
curl -H "$H" "https://lum.id/findata/analyst-estimates/AAPL"              # consensus EPS/revenue estimates
curl -H "$H" "https://lum.id/findata/recommendation/AAPL"                 # upgrade/downgrade history
```

---

## Ownership & investors

```bash
curl -H "$H" "https://lum.id/findata/holders/AAPL/top"                     # top 13F holders
curl -H "$H" "https://lum.id/findata/insider/AAPL/transactions"            # insider buys/sells
curl -H "$H" "https://lum.id/findata/insider/AAPL/sentiment"               # net insider sentiment
curl -H "$H" "https://lum.id/findata/insider/AAPL/statistics"              # insider stats summary
curl -H "$H" "https://lum.id/findata/fund-ownership/AAPL"                  # mutual/ETF fund holders
curl -H "$H" "https://lum.id/findata/funds-disclosure/AAPL"                # Form-D / fund disclosures
curl -H "$H" "https://lum.id/findata/gov-trades/AAPL"                      # congressional trades
curl -H "$H" "https://lum.id/findata/acquisitions/AAPL"                    # beneficial ownership (13D/G)

# Institutional analytics (by CIK)
curl -H "$H" "https://lum.id/findata/institutional/AAPL/holders/analytics"
curl -H "$H" "https://lum.id/findata/institutional/holder/<cik>/dates"
curl -H "$H" "https://lum.id/findata/institutional/holder/<cik>/performance"
curl -H "$H" "https://lum.id/findata/institutional/holder/<cik>/industries"
curl -H "$H" "https://lum.id/findata/institutional/industries"
```

---

## Company metadata

```bash
curl -H "$H" "https://lum.id/findata/executives/AAPL"                  # C-suite + board
curl -H "$H" "https://lum.id/findata/governance/AAPL/compensation"      # exec compensation
curl -H "$H" "https://lum.id/findata/exec-comp-benchmark/<industry>"    # industry avg comp benchmarks
curl -H "$H" "https://lum.id/findata/peers/AAPL"                        # peer companies
curl -H "$H" "https://lum.id/findata/supply-chain/AAPL"                 # supply-chain relationships
curl -H "$H" "https://lum.id/findata/shares-float/AAPL"                 # float, short interest
curl -H "$H" "https://lum.id/findata/employee-count/AAPL"               # historical headcount
curl -H "$H" "https://lum.id/findata/symbol-changes"                    # ticker rename history
curl -H "$H" "https://lum.id/findata/symbol/SPY/etf-exposure"           # ETF exposure for a symbol
```

---

## Events & calendars

```bash
curl -H "$H" "https://lum.id/findata/earnings"                          # earnings calendar (upcoming)
curl -H "$H" "https://lum.id/findata/earnings/AAPL/history"             # per-symbol earnings history
curl -H "$H" "https://lum.id/findata/transcripts/AAPL"                  # earnings call transcripts (list)
curl -H "$H" "https://lum.id/findata/transcripts/AAPL/2025/4"           # specific quarter transcript
curl -H "$H" "https://lum.id/findata/ipos"                              # IPO calendar
curl -H "$H" "https://lum.id/findata/dividends/AAPL"                    # dividend history
curl -H "$H" "https://lum.id/findata/dividends-calendar"                # upcoming dividend dates
curl -H "$H" "https://lum.id/findata/splits/AAPL"                       # split history
curl -H "$H" "https://lum.id/findata/splits-calendar"                   # upcoming splits
curl -H "$H" "https://lum.id/findata/fda-calendar"                      # FDA PDUFA / advisory dates
curl -H "$H" "https://lum.id/findata/mergers-acquisitions"              # M&A deal tracker
curl -H "$H" "https://lum.id/findata/exchange-market-hours"             # exchange open/close windows
curl -H "$H" "https://lum.id/findata/exchange/XNYS/holidays"            # exchange holidays
```

---

## ETF

```bash
curl -H "$H" "https://lum.id/findata/etf/SPY/info"                  # fund info, AUM, expense ratio
curl -H "$H" "https://lum.id/findata/etf/SPY/holdings"              # top holdings
curl -H "$H" "https://lum.id/findata/etf/SPY/sector-weightings"     # sector allocation
curl -H "$H" "https://lum.id/findata/etf/SPY/country-weightings"    # country allocation
curl -H "$H" "https://lum.id/findata/index/SPX/constituents"        # index constituent list
```

---

## News

~20.7 M articles across multiple wire and CSV feeds, deduplicated, full-text indexed. A sentiment-enhanced feed adds structured per-ticker sentiment (positive/negative/neutral + reasoning text) for each article; stored in the `raw` field and refreshed every 4 hours.

```bash
curl -H "$H" "https://lum.id/findata/news/AAPL?limit=50"               # per-symbol news
curl -H "$H" "https://lum.id/findata/news/latest?since=2026-06-01&limit=100"  # global firehose
curl -H "$H" "https://lum.id/findata/news/search?q=earnings+beat&limit=50"    # full-text search
curl -H "$H" "https://lum.id/findata/news/stats"                       # article counts / source breakdown
curl -H "$H" "https://lum.id/findata/news/social-sentiment/AAPL"       # social sentiment series
curl -H "$H" "https://lum.id/findata/news/symbol-sentiment/AAPL"       # symbol-level sentiment
```

---

## KOL tweets

35.1 M tweets from curated handles, 2009–2026. Full-text + cashtag indexed.

```bash
curl -H "$H" "https://lum.id/findata/kols"                                           # roster list
curl -H "$H" "https://lum.id/findata/kols/tweets?limit=50"                           # recent tweets, all handles
curl -H "$H" "https://lum.id/findata/kols/tweets/search?q=earnings+beat&limit=50"   # full-text search
curl -H "$H" "https://lum.id/findata/kols/tweets/search?q=NVDA&cashtag=NVDA"        # cashtag filter
curl -H "$H" "https://lum.id/findata/kols/tweets/by-symbol/AAPL"                    # tweets mentioning $AAPL
curl -H "$H" "https://lum.id/findata/kols/tweets/by-symbol/AAPL/history?limit=200"  # historical
curl -H "$H" "https://lum.id/findata/kols/elonmusk/tweets"                          # per-handle recent
curl -H "$H" "https://lum.id/findata/kols/elonmusk/tweets/history?since=2025-01-01" # per-handle archive
curl -H "$H" "https://lum.id/findata/kols/archive/stats"                            # archive row counts
```

### KOL media (11.5 M mirrored images)
```bash
curl -H "$H" "https://lum.id/findata/kols/media"                           # media index
curl -H "$H" -L "https://lum.id/findata/kols/media/by-url?u=<twimg-url>"  # serve by original CDN URL; falls through if not cached
curl -H "$H" "https://lum.id/findata/kols/media/<rel>"                     # serve by internal path
```

---

## Macro

```bash
curl -H "$H" "https://lum.id/findata/macro/treasury-rates"                          # yield curve history
curl -H "$H" "https://lum.id/findata/macro/economic-indicators"                     # CPI, unemployment, GDP, …
curl -H "$H" "https://lum.id/findata/macro/economic-calendar?from=2026-01-01&to=2026-12-31"
curl -H "$H" "https://lum.id/findata/macro/cot/ES"                                  # CFTC commitment of traders
```

---

## Short interest & short volume

FINRA bi-weekly short interest and daily off-exchange short volume per symbol. Updated weekly.

```bash
curl -H "$H" "https://lum.id/findata/short-interest/AAPL"           # FINRA short interest history (settlement_date, short_interest, avg_daily_volume, days_to_cover)
curl -H "$H" "https://lum.id/findata/short-interest/AAPL?limit=10"  # last 10 bi-weekly prints
curl -H "$H" "https://lum.id/findata/short-volume/AAPL"             # daily off-exchange short volume (ADF + Nasdaq Carteret + NYSE breakdowns)
```

---

## SEC filings — text

Machine-readable SEC risk-factor disclosures and plain-text 10-K sections, updated weekly.

```bash
# Risk factors — standardized taxonomy (primary/secondary/tertiary category + supporting text)
curl -H "$H" "https://lum.id/findata/risk-factors/AAPL"
curl -H "$H" "https://lum.id/findata/risk-factors/AAPL?limit=50"

# 10-K annual filing sections (business, risk_factors, mda, legal_proceedings, …)
curl -H "$H" "https://lum.id/findata/10k-sections/AAPL"
curl -H "$H" "https://lum.id/findata/10k-sections/AAPL?section=mda"           # management's discussion
curl -H "$H" "https://lum.id/findata/10k-sections/AAPL?section=risk_factors"  # risk factor text
```

---

## Regulatory

```bash
curl -H "$H" "https://lum.id/findata/esg/AAPL/disclosures"          # ESG disclosure filings
curl -H "$H" "https://lum.id/findata/esg/AAPL/ratings"              # third-party ESG ratings
curl -H "$H" "https://lum.id/findata/esg/AAPL/historical"           # historical ESG time-series
curl -H "$H" "https://lum.id/findata/filings/AAPL"                  # SEC filings (10-K, 10-Q, 8-K, …)
curl -H "$H" "https://lum.id/findata/xbrl/AAPL/filings"             # XBRL filing index
curl -H "$H" "https://lum.id/findata/xbrl/AAPL/filing/<accession>"  # XBRL filing detail
curl -H "$H" "https://lum.id/findata/lobbying/AAPL"                 # lobbying spend history
curl -H "$H" "https://lum.id/findata/usa-spending/AAPL"             # federal contract awards
curl -H "$H" "https://lum.id/findata/visa-applications/AAPL"        # H-1B / LCA visa applications
curl -H "$H" "https://lum.id/findata/uspto-patents/AAPL"            # patent grants
```

---

## Prediction markets

`venue` ∈ `polymarket | kalshi | polymarket_us`  ·  `market_id` = polymarket **condition_id** (`0x…`) or kalshi **ticker**

```bash
# Discovery
curl -H "$H" "https://lum.id/findata/prediction-markets/markets/search?q=bitcoin"
curl -H "$H" "https://lum.id/findata/prediction-markets/events?status=open&limit=100"
curl -H "$H" "https://lum.id/findata/prediction-markets/markets/polymarket/<condition_id>"
curl -H "$H" "https://lum.id/findata/prediction-markets/markets/kalshi/<ticker>"

# Polymarket US (US-regulated exchange universe — 10.6k markets; discovery/detail only, no candles/trades)
curl -H "$H" "https://lum.id/findata/prediction-markets/markets/polymarket_us?q=election&category=Politics&closed=false&limit=100"
curl -H "$H" "https://lum.id/findata/prediction-markets/markets/polymarket_us/<slug>"

# Price history (OHLC — UNION of executed trades + orderbook midprice)
# interval = minutes (1|5|15|60|1440) or a string alias (1m|5m|15m|30m|1h|4h|1d); default 1d
# from/to accept `start`/`end` as aliases
# Polymarket coverage: interval=1440 (daily) back to 2024-01-02; intervals ≤60 only from ~Dec 2025.
# For any market older than ~7 months, use interval=1440. Kalshi: all intervals back to 2021-06.
curl -H "$H" "https://lum.id/findata/prediction-markets/candles/polymarket/<cid>?interval=60"
curl -H "$H" "https://lum.id/findata/prediction-markets/candles/kalshi/<ticker>?interval=60"

# Trade history
curl -H "$H" "https://lum.id/findata/prediction-markets/trades/polymarket/<condition_id>"
curl -H "$H" "https://lum.id/findata/prediction-markets/trades/kalshi/<ticker>"

# Orderbook (latest L2 snapshot)
curl -H "$H" "https://lum.id/findata/prediction-markets/orderbook/polymarket/<asset_id>"
curl -H "$H" "https://lum.id/findata/prediction-markets/orderbook/kalshi/<ticker>"

# Market analytics
curl -H "$H" "https://lum.id/findata/prediction-markets/open-interest/polymarket/<cid>"
curl -H "$H" "https://lum.id/findata/prediction-markets/top-holders/polymarket/<cid>"
curl -H "$H" "https://lum.id/findata/prediction-markets/matched-pairs/polymarket/<cid>"  # cross-venue equivalents

# Leaderboard — window ∈ 7d|24h|30d|week|all
curl -H "$H" "https://lum.id/findata/prediction-markets/leaderboard?window=7d&venue=polymarket&limit=50"

# Wallet (Polymarket)
curl -H "$H" "https://lum.id/findata/prediction-markets/wallet/<address>"
curl -H "$H" "https://lum.id/findata/prediction-markets/wallet/<address>/pnl"
curl -H "$H" "https://lum.id/findata/prediction-markets/wallet/<address>/positions"
curl -H "$H" "https://lum.id/findata/prediction-markets/wallet/<address>/activity"
```
Note: `event_id` is a **string** (`"284199"`), not an integer. Kalshi is accepted on most endpoints but wallet/leaderboard data is Polymarket-only.

---

## LQT integration (trading pipeline)

Endpoints that back the Lumid QuantTrading (LQT) pipeline — the monitored prediction-market universe and the mailbox command-plane. These are a specialized surface (not general-purpose market data); the mailbox write path requires the LQT fleet scope.

```bash
# Monitored universe — venue ∈ polymarket|polymarket_us|kalshi (required)
curl -H "$H" "https://lum.id/findata/api/v1/lqt/universe?venue=polymarket"                 # add include_inactive=1 for inactive markets
curl -H "$H" "https://lum.id/findata/api/v1/lqt/markets?venue=polymarket&min_similarity=70" # LQT-tagged markets; min_similarity 0–100 (default 90 = exact-only, 70 = include related), optional tag=
```

```bash
# Mailbox — strategies in, results/telemetry out
curl -H "$H" "https://lum.id/findata/xpio/strategies?status=deployed&limit=100"   # list deployed strategies (status=sent|deployed, kind=<payload kind>)
curl -H "$H" "https://lum.id/findata/xpio/results?topic=result.fill&since=2026-07-01T00:00:00Z&limit=100"  # executor outbox (result.* topics)
curl -H "$H" "https://lum.id/findata/xpio/telemetries?topic=telemetry.heartbeat&limit=100"                # executor outbox (telemetry.* topics)
curl -H "$H" "https://lum.id/findata/xpio/stats"                                  # harvest-cursor position + strategy/result/telemetry counts

# Deploy a strategy into the mailbox (executor picks it up) — returns {strategy_id, msg_id}. Requires the LQT fleet scope.
curl -H "$H" -H 'Content-Type: application/json' -X POST https://lum.id/findata/xpio/strategies -d '{ ... }'
```

---

## Realtime (SSE / WebSocket)

```bash
# SSE streams (no reconnect needed — server sends heartbeat every 30 s)
curl -N -H "$H" "https://lum.id/findata/quotes/stream?symbols=AAPL,BTCUSD"   # tick: subscribed|tick|heartbeat
curl -N -H "$H" "https://lum.id/findata/prediction-markets/stream"            # PM: trade + orderbook deltas
```

**WebSocket** — auth via `Authorization` header, `?token=`, or `Sec-WebSocket-Protocol: bearer.<token>`:
- `wss://lum.id/findata/ws/quotes` — market ticks (subscribe `{symbols:[...]}` after connect)
- `wss://lum.id/findata/ws/news` — news article stream
- `wss://lum.id/findata/ws/prediction-markets` — PM events (same as the PM SSE)

US equity quotes are delivered from the primary tick stream with a secondary quote feed as a live backup — the server auto-covers from the secondary within seconds if the primary goes quiet, with no manual intervention required.

PM event shape:
```jsonc
// Polymarket trade
{"channel":"trade","venue":"polymarket","asset_id":"…","condition_id":"0x…","price":"0.62","size":"40","side":"BUY"}
// Polymarket orderbook delta
{"channel":"orderbook","venue":"polymarket","asset_id":"…","condition_id":"0x…", …}
// Kalshi trade
{"channel":"trade","venue":"kalshi","market_ticker":"KXBTC-…","yes_price_dollars":"0.59","count":"164","taker_side":"yes"}
```
SSE optional filters: `?asset_ids=…&condition_ids=…` (Polymarket keys — Kalshi passes through unfiltered).

---

## OpenAI-compatible LLM (`/v1/*`)

The service proxies to a reasoning model. Point any OpenAI or Anthropic SDK at this host and use your Lumid PAT as the API key.

```python
from openai import OpenAI
client = OpenAI(base_url="https://lum.id/findata/v1", api_key="<token>")
resp = client.chat.completions.create(
    model="<model>",                        # or omit for the default
    messages=[{"role":"user","content":"Summarize AAPL's latest quarter."}],
    max_tokens=1024,
)
print(resp.choices[0].message.content)
```
```bash
curl https://lum.id/findata/v1/models -H "$H"                             # list available models
curl https://lum.id/findata/v1/chat/completions -H "$H" \
  -H 'Content-Type: application/json' \
  -d '{"model":"<model>","messages":[{"role":"user","content":"hi"}]}'
```
Endpoints: `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/models`, `/v1/messages`, `/v1/messages/count_tokens`.

**Reasoning models**: thinking lands in `reasoning_content` (OpenAI shape) or a `thinking` block (Anthropic shape). Set `max_tokens` ≥ a few hundred or `content` may be empty. Returns `503` if no LLM backend is configured.

---

## MCP (Model Context Protocol)

`POST /mcp` — JSON-RPC 2.0, Streamable-HTTP transport (MCP 2025-03-26). 90 auto-generated tools, one per read endpoint.

```bash
curl -H "$H" -H 'Content-Type: application/json' -X POST https://lum.id/findata/mcp \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
curl -H "$H" -H 'Content-Type: application/json' -X POST https://lum.id/findata/mcp \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"fundamentals_latest","arguments":{"symbol":"AAPL"}}}'
```

---

## Write / ingest

```bash
# Write rows to an existing table (validated, provenance-stamped, upserted)
curl -H "$H" -H 'Content-Type: application/json' \
  -X POST https://lum.id/findata/ingest/<schema>/<table> \
  -d '{"records":[{"symbol":"AAPL","date":"2026-06-01", ...}]}'
```
Provenance fields (`source`, `source_endpoint`, `source_run_id`, `ingest_ts`) are stamped server-side — don't supply them. Upsert is newest-wins on the natural key; re-POSTing identical rows is a no-op.

Other ingest modes: `/ingest/<schema>/<table>/stream` (NDJSON chunked), `/ingest/<schema>/<table>/file` (multipart JSON/CSV/Parquet), `/ingest/blob` (binary / images with sha256 dedup).

### Keyed blob store
`/ingest/blob` content-addresses by sha256; `/blobs/<key>` addresses by a caller-chosen key.
```bash
curl -H "$H" --data-binary @report.pdf -X PUT https://lum.id/findata/blobs/<key>   # write/overwrite a blob at <key>
curl -H "$H" -X DELETE https://lum.id/findata/blobs/<key>                          # delete it
```

### Schema negotiation (new table)
POST to an unknown table → platform suggests a schema and stages a **proposal**:
```bash
curl -H "$H" -H 'Content-Type: application/json' -X POST https://lum.id/findata/ingest/sandbox/widgets \
  -d '{"records":[{"widget_id":7,"name":"alpha","price":9.99,"ts":"2026-06-01T00:00:00Z"}]}'
# returns proposal_id; then negotiate:
curl -H "$H" https://lum.id/findata/catalog/ingress/proposals/<id>
curl -H "$H" -X POST https://lum.id/findata/ingress/proposals/<id>/approve
```

---

## Catalog & lineage

```bash
curl -H "$H" "https://lum.id/findata/catalog/schemas"
curl -H "$H" "https://lum.id/findata/catalog/schemas/<schema>/tables"
curl -H "$H" "https://lum.id/findata/catalog/tables/<schema>/<table>"
curl -H "$H" "https://lum.id/findata/catalog/tables/<schema>/<table>/schema.json"
curl -H "$H" "https://lum.id/findata/catalog/lineage/run/<run_id>"
curl -H "$H" "https://lum.id/findata/catalog/lineage/row?schema=market&table=ohlc_daily&symbol=AAPL"
curl -H "$H" "https://lum.id/findata/catalog/lineage/runs"
curl -H "$H" "https://lum.id/findata/catalog/sources"
curl -H "$H" "https://lum.id/findata/catalog/submitters"
```

**Query cost profile** — `POST /profile` returns a cost/plan estimate for a query before you run it. See `/reference` for the request-body schema.
```bash
curl -H "$H" -H 'Content-Type: application/json' -X POST https://lum.id/findata/profile -d '{ ... }'
```

---

## Query it from your own tools

**Two response shapes.** Most list routes (`/news/*`, `/kols/*`,
`/prediction-markets/trades/*`) return a **bare JSON array**. A few wrap the rows:
`/ohlc/*` returns `{…, "bars":[…]}` and `/screener` returns `{…, "hits":[…]}`.
Unwrap accordingly — the examples below show both.

### DuckDB — SQL straight over the API
Nothing to download; `httpfs` fetches and DuckDB does the SQL.
```sql
INSTALL httpfs; LOAD httpfs;
CREATE SECRET findata (
  TYPE http,
  EXTRA_HTTP_HEADERS MAP {'Authorization': 'Bearer ' || getenv('LUMID_TOKEN')}
);

-- envelope route: unnest the rows out of the wrapper
SELECT b.* FROM (
  SELECT unnest(bars) AS b
  FROM read_json_auto('https://lum.id/findata/ohlc/AAPL?interval=1d&from=2026-01-01')
);

-- bare-array route: read it directly
SELECT ts, title, source
FROM read_json_auto('https://lum.id/findata/news/search?q=nvidia&limit=200')
ORDER BY ts DESC;
```

### Python — requests + pandas
```python
import os, requests, pandas as pd

S = requests.Session()
S.headers["Authorization"] = f"Bearer {os.environ['LUMID_TOKEN']}"
BASE = "https://lum.id/findata"

def frame(path, key=None, **params):
    r = S.get(f"{BASE}{path}", params=params, timeout=60)
    r.raise_for_status()
    body = r.json()
    return pd.DataFrame(body[key] if key else body)   # key= only for envelope routes

bars = frame("/ohlc/AAPL", key="bars", interval="1d", **{"from": "2026-01-01"})
news = frame("/news/search", q="nvidia", limit=200)
```

### Shell — curl + jq → CSV
```bash
curl -sH "$H" "https://lum.id/findata/prediction-markets/trades/kalshi/<ticker>?limit=1000" \
  | jq -r '.[] | [.ts,.price,.size] | @csv' > trades.csv
```
Anything that speaks HTTP + JSON works the same way: `csvkit`, R's `jsonlite`,
Observable, a notebook, your own client generated from `/findata/openapi.json`.

### Postgres wire — psql, DBeaver, Metabase, anything with a Postgres driver

The warehouse is directly reachable over the Postgres protocol, read-only. This is
the right surface for exploratory work, joins across schemas, and extracts too
large or too irregular for the REST routes.

```bash
curl -O https://lum.id/findata/sql-ca.pem
psql "host=sql.lum.id port=5432 dbname=findata user=lumid_reader \
      sslmode=verify-full sslrootcert=sql-ca.pem"
```

```
postgresql://lumid_reader:<password>@sql.lum.id:5432/findata?sslmode=verify-full&sslrootcert=sql-ca.pem
```

**Use `verify-full`, not `require`.** `require` encrypts but accepts *any*
certificate, so it does not protect against an active MITM. `sql-ca.pem` above is
the CA that signs this endpoint, served over lum.id's own TLS — that is what
anchors the trust. Credentials are issued per requester; ask for the password.

What you get, and the limits that come with it:

| | |
|---|---|
| 22 schemas, ~9990 tables | `\dn` to list; `timescaledb_information.hypertables` for the time-series ones |
| read-only | the session is `default_transaction_read_only`; writes are refused, not ignored |
| `statement_timeout` 2 min | narrow the window rather than raising it — you cannot |
| 40 concurrent connections | shared across users; a pool of 2-3 is neighbourly |
| TLS required | plaintext connections are refused outright |

Where things live: `market` holds the OHLC hypertables (`ohlc_1min`, `ohlc_5min`,
`ohlc_daily`, columns `symbol, date, open, high, low, close, adj_close, volume,
vwap`); `prediction_markets` holds 34 tables of Polymarket/Kalshi markets, trades,
per-interval candles, orderbook snapshots and wallet analytics; then
`fundamentals`, `news`, `macro`, `estimates`, `ownership`, `reference`,
`instrument`, `events`, `regulatory`, `md`.

```sql
select symbol, date, close, volume
from market.ohlc_daily
where symbol = 'AAPL'
order by date desc
limit 5;
```

DuckDB can attach it directly, which is usually nicer than the httpfs route above
because the filtering then happens server-side:

```sql
INSTALL postgres; LOAD postgres;
ATTACH 'host=sql.lum.id port=5432 dbname=findata user=lumid_reader
        password=... sslmode=verify-full sslrootcert=sql-ca.pem' AS fd (TYPE postgres, READ_ONLY);
SELECT * FROM fd.market.ohlc_daily WHERE symbol='NVDA' ORDER BY date DESC LIMIT 10;
```

The HTTP API remains the primary surface — it is rate-limited, accounted, cached
and federated, and SQL bypasses all four. `usage/me` does not see SQL sessions.

### Paging, limits, backpressure
`limit` caps at **1000** on list routes; page with the time filters (`from`/`to`,
`since`/`until`) rather than an offset. `/ohlc` ignores `limit` and applies a
server-side row cap — narrow the window instead. Watch `x-ratelimit-remaining` and
honour `Retry-After` on a `429`. For anything the REST shape fights you on, use
the Postgres wire below. Bulk one-off extracts are still provisioned per
account — ask.

---

## Your usage & status

```bash
curl -H "$H" "https://lum.id/findata/usage/me"   # your calls, bytes, hourly breakdown
```
Returns `{sub, total_calls, bytes_out, calls_last_24h, hourly_last_24h:[24 hourly buckets]}`.

`/status` (HTML health board), `/freshness` (per-endpoint SLA counts), `/health` (liveness probe). All public, no auth.