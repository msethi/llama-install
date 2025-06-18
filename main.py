import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from dotenv import load_dotenv

import pandas as pd
import requests
import yfinance as yf
from jinja2 import Environment, select_autoescape
from jinja2 import Template
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from pushover_complete import PushoverAPI
from bs4 import BeautifulSoup
from py_vollib.black_scholes.greeks.analytical import delta, theta
from py_vollib.black_scholes.implied_volatility import implied_volatility
import feedparser, datetime as _dt, pytz
from datetime import datetime, timedelta, timezone
# Load environment variables from .env file
load_dotenv()


# --- Config and Constants ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FMP_API_KEY = os.getenv("FMP_API_KEY")
PUSHOVER_USER_KEY = os.getenv("PUSHOVER_USER_KEY")
PUSHOVER_APP_TOKEN = os.getenv("PUSHOVER_APP_TOKEN")

REPORT_DIR = "reports"
DETAILED_DIR = "detailed_option_chains"
MAX_REPORTS_TO_KEEP = 3
MAX_THREADS = 8

TICKERS_CSV_PATH = "sp100_tickers_sectors.csv"
ADDITIONAL_TICKERS_CSV_PATH = "additional_tickers.csv"

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY  # for langchain/OpenAI SDK

# --- Jinja2 HTML Template for report ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>S&P 100 Options Report {{ run_time }}</title>
<style>
body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
h1, h2 { color: #333; }
table { border-collapse: collapse; width: 100%; background: white; }
th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }
th { background-color: #4CAF50; color: white; }
tr:nth-child(even) { background-color: #f2f2f2; }
.highlight { background-color: #ffdddd; }
</style>
</head>
<body>
<h1>S&P 100 Options Report</h1>
<p>Run Time: {{ run_time }}</p>

<h2>Major US Economic Events (Next 7 Days)</h2>
<ul>
{% for event in economic_events %}
  <li>{{ event }}</li>
{% else %}
  <li>No major US economic events in next 7 days.</li>
{% endfor %}
</ul>

<h2>Upcoming Earnings</h2>
<ul>
{% for ticker, edate in earnings_summary.items() %}
  <li>{{ ticker }}: {{ edate if edate else 'No upcoming earnings' }}</li>
{% endfor %}
</ul>

<h2>Top 50 Cash-Secured Put Options (Sorted by Score)</h2>
<table>
<thead>
<tr>
<th>Ticker</th><th>Sector</th><th>Strike</th><th>Expiration</th><th>Bid</th><th>Ask</th><th>Last Price</th><th>IV</th><th>Delta</th><th>Theta</th><th>Score</th><th>Annualized Yield</th><th>Earnings Date</th>
</tr>
</thead>
<tbody>
{% for row in options_data %}
<tr {% if row['annualized_yield'] > 0.07 and row['score'] > 90 %}class="highlight"{% endif %}>
<td>{{ row['ticker'] }}</td><td>{{ row['sector'] }}</td><td>{{ row['strike'] }}</td><td>{{ row['expirationDate'] }}</td>
<td>{{ "%.2f"|format(row['bid']) }}</td><td>{{ "%.2f"|format(row['ask']) }}</td><td>{{ "%.2f"|format(row['lastPrice']) }}</td><td>{{ "%.2f"|format(row['impliedVolatility']*100) if row['impliedVolatility'] else 'N/A' }}%</td>
<td>{{ "%.2f"|format(row['delta']) }}</td><td>{{ "%.4f"|format(row['theta']) }}</td><td>{{ "%.2f"|format(row['score']) }}</td><td>{{ "%.2f"|format(row['annualized_yield']) }}</td><td>{{ row['earningsDate'] if row['earningsDate'] else 'N/A' }}</td>
</tr>
{% endfor %}
</tbody>
</table>

</body>
</html>
"""

env = Environment(autoescape=select_autoescape())
template = env.from_string(HTML_TEMPLATE)

# --- Option Strategist Prompt and Function ---

OPTION_STRATEGIST_SYSTEM_PROMPT = """
You are the Chief Options Strategist at a Wall Street hedge fund. You are a quantitative expert in options pricing, Greeks, and volatility risk management.
You are provided with an option chain (as a table), which contains columns for expiration, type (call/put), strike, last price, bid, ask, implied volatility, delta, gamma, theta, vega, and open interest.

**Your goals:**
- Analyze all positions using the full option chain.
- Summarize overall Greeks exposures and surface the highest absolute values for delta, gamma, theta, and vega.
- Identify positions or strikes with unusual implied volatility or rapid theta decay.
- Flag options where gamma is high (especially near expiry or at-the-money).
- Recommend strategies for exploiting current volatility conditions (e.g., high IV = sell premium; low IV = consider buying options).
- Detect and summarize risk clusters (e.g., large directional risk, large vega, or exposure to fast decay).
- Make tactical recommendations (trade ideas or risk alerts) based on your findings.

**Always:**
- Start with a concise summary of the current options landscape for the ticker.
- Quantify risks and exposures, citing the specific strikes and expirations involved.
- Explain your reasoning in terms of Greeks and volatility.
- Suggest any hedges or adjustments if significant risks are found.
- Only use the data from the provided option chain.

Always use technical language and show calculations as needed.
"""

def analyze_option_chain(csv_path, ticker="TICKER"):
    df = pd.read_csv(csv_path)
    df.columns = [c.lower().strip() for c in df.columns]

    display_cols = ['expiration', 'type', 'strike', 'last price', 'bid', 'ask',
                    'implied volatility', 'delta', 'gamma', 'theta', 'vega', 'open interest']
    df = df[display_cols]
    for col in ['implied volatility', 'delta', 'gamma', 'theta', 'vega']:
        df[col] = df[col].round(4)

    head_table = df.head(30).to_markdown(index=False)

    agent_prompt = (
        f"TICKER: {ticker}\n"
        f"Here is the latest option chain (showing the first 30 rows):\n\n"
        f"{head_table}\n\n"
        f"If you need more data, ask for it. Analyze this option chain per the strategy and instructions above."
    )

    messages = [
        SystemMessage(content=OPTION_STRATEGIST_SYSTEM_PROMPT),
        HumanMessage(content=agent_prompt)
    ]

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, max_tokens=4000)
    result = llm.invoke(messages)
    return result.content


# --- Utilities for ticker loading and blending ---

def load_tickers_and_sectors(csv_path):
    df = pd.read_csv(csv_path)
    return df.to_dict(orient="records")


def load_additional_tickers(csv_path):
    if not os.path.exists(csv_path):
        print(f"No additional tickers file found at {csv_path}")
        return []
    df = pd.read_csv(csv_path)
    # Convert the first column to a list of dictionaries with 'ticker' key
    ticker_column = df.columns[0]  # Get the name of the first column
    return [{'ticker': str(ticker).strip(), 'sector': 'Unknown'} for ticker in df[ticker_column]]


def blend_tickers(sp100_list, additional_list, tech_weight=0.5, total_tickers=50):
    # Convert additional_list items to dict format if they're not already
    formatted_additional = []
    for item in additional_list:
        if isinstance(item, str):
            formatted_additional.append({'ticker': item, 'sector': 'Unknown'})
        elif isinstance(item, dict):
            if 'ticker' not in item:
                continue
            if 'sector' not in item:
                item['sector'] = 'Unknown'
            formatted_additional.append(item)
    
    combined = sp100_list + formatted_additional
    seen = set()
    unique_combined = []
    for t in combined:
        if t['ticker'] not in seen:
            seen.add(t['ticker'])
            unique_combined.append(t)

    tech_stocks = [s for s in unique_combined if s['sector'].lower() in ['technology', 'information technology']]
    other_stocks = [s for s in unique_combined if s not in tech_stocks]

    n_tech = int(total_tickers * tech_weight)
    n_other = total_tickers - n_tech

    selected_tech = tech_stocks[:n_tech]
    selected_other = other_stocks[:n_other]

    return selected_tech + selected_other


# --- Economic Events and Earnings from FMP ---

import feedparser
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone

def get_economic_events_fmp_or_rss():
    """
    First try FMP calendar; on HTTP 4xx/5xx or empty result,
    fallback to Investing.com economic RSS feed.
    Returns up to 10 high-impact US events in next 7 days.
    """
    # --- Attempt FMP API first ---
    try:
        url = (
            f"https://financialmodelingprep.com/api/v3/economic_calendar"
            f"?from={datetime.today().strftime('%Y-%m-%d')}"
            f"&to={(datetime.today() + timedelta(days=7)).strftime('%Y-%m-%d')}"
            f"&apikey={FMP_API_KEY}"
        )
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        us_high = [
            f"{e['date']} - {e['event']} (Imp {e.get('importance', '?')})"
            for e in data
            if e.get('country') == "US" and e.get('importance', 0) >= 3
        ]
        if us_high:
            return us_high[:10]
    except Exception as e:
        # print(f"[WARN] FMP econ events failed: {e}")
        pass

    # --- Fallback: Investing.com RSS ---
    try:
        rss = feedparser.parse("https://www.investing.com/rss/news_285.rss")
        today = datetime.now(timezone.utc).date()
        cutoff = today + timedelta(days=7)
        events = []
        for entry in rss.entries:
            # parse published date
            pub = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc).date()
            title = entry.title or ""
            if today <= pub <= cutoff and "United States" in title:
                events.append(f"{pub} – {title}")
            if len(events) >= 10:
                break
        return events
    except Exception as rss_err:
        # print(f"[WARN] RSS fallback failed: {rss_err}")
        return []



def get_upcoming_earnings(ticker):
    """
    Try FMP earning_calendar first; if empty or error,
    fallback to yfinance's calendar for 'Earnings Date'.
    Returns a datetime.date or None.
    """
    # --- Try FMP API ---
    try:
        url = f"https://financialmodelingprep.com/api/v3/earning_calendar/{ticker}?apikey={FMP_API_KEY}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        arr = resp.json()
        for item in arr:
            date = datetime.strptime(item['date'], "%Y-%m-%d").date()
            if date >= datetime.today().date():
                return date
    except Exception as e:
        # print(f"[WARN] FMP earnings failed for {ticker}: {e}")
        pass

    # --- Fallback: yfinance calendar ---
    try:
        cal = yf.Ticker(ticker).calendar
        if isinstance(cal, pd.DataFrame) and 'Earnings Date' in cal.index:
            ed = cal.loc['Earnings Date'][0]
            if isinstance(ed, (datetime, pd.Timestamp)):
                d = ed.date() if isinstance(ed, datetime) else ed
                return d if d >= datetime.today().date() else None
    except Exception as yerr:
        # print(f"[WARN] yfinance earnings failed for {ticker}: {yerr}")
        pass

    return None



# --- Option Chain and Scoring ---

def get_puts_options(ticker):
    stock = yf.Ticker(ticker)
    today = datetime.today()
    expirations = stock.options
    options_list = []
    for date_str in expirations:
        try:
            exp_date = datetime.strptime(date_str, "%Y-%m-%d")
            days_to_exp = (exp_date - today).days
            if 14 <= days_to_exp <= 175:  # Changed from 105 (15 weeks) to 175 (25 weeks)
                opt_chain = stock.option_chain(date_str)
                puts = opt_chain.puts
                puts['expirationDate'] = date_str
                puts['ticker'] = ticker
                options_list.append(puts)
        except Exception:
            continue
    if options_list:
        df = pd.concat(options_list)
        if 'delta' not in df.columns:
            df['delta'] = 0
        if 'theta' not in df.columns:
            df['theta'] = 0
        return df
    return pd.DataFrame()

def enrich_greeks(row, underlying_price, r=0.05):
    row = row.copy()  # Create a copy of the row to avoid SettingWithCopyWarning
    S = underlying_price
    K = row['strike']
    T = (pd.to_datetime(row['expirationDate']) - pd.Timestamp.today()).days / 365
    if T <= 0:
        return row
    # Use mid price for IV seed
    mid = (row['bid'] + row['ask']) / 2 if row['ask'] > 0 else row['lastPrice']
    try:
        iv = implied_volatility(mid, S, K, T, r, flag='p')
        row['impliedVolatility'] = iv
        row['delta'] = delta('p', S, K, T, r, iv)
        row['theta'] = theta('p', S, K, T, r, iv) / 365  # per-day
    except Exception:
        pass
    return row

def fetch_data_for_ticker(ticker_info):
    ticker = ticker_info["ticker"]
    sector = ticker_info["sector"]
    fundamentals = {}
    options = pd.DataFrame()
    earnings_date = None

    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # ─── Get a reliable spot price ──────────────────────────────────────
        try:
            spot = info.get("regularMarketPrice")
            if spot is None:
                spot = stock.history(period="1d", auto_adjust=True)["Close"].iloc[-1]
        except Exception:
            spot = None
            

        fundamentals = {
            "marketCap": info.get("marketCap", 0),
            "sector": sector,
            "beta": info.get("beta", 1),
            "lastPrice": spot,            # ← needed for the ITM filter
        }

        # ─── earnings date ─────────────────────────────────────────────────
        earnings_date = get_upcoming_earnings(ticker)   

        # ─── raw option chain (puts only) ──────────────────────────────────
        options = get_puts_options(ticker)
        if not options.empty and spot is not None:
            # Enrich with IV & Greeks
            options = options.apply(lambda r: enrich_greeks(r, spot), axis=1)
            # Ensure gamma/vega exist
            for col in ["gamma", "vega"]:
                if col not in options.columns:
                    options[col] = 0

    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")

    return ticker, fundamentals, earnings_date, options


def improved_score_option(row, fundamentals, sector_beta, events, strike_cushion=0.9):
    """
    Heuristic score for cash-secured puts with slightly relaxed risk parameters:
      • Allows higher IV (up to 400%)
      • Allows higher theta decay (up to 0.2/day)
      • Reduces small-cap penalty to $8B and –20 points
      • strike_cushion: e.g. 0.9 for 10% OTM, 0.8 for 20% OTM
    """

    spot = fundamentals.get("lastPrice")
    if spot is None or row["strike"] > spot * strike_cushion:
        return -100
    


    # ─── 0. Reject extreme IV ─────────────────────────────────────────────
    iv = row.get("impliedVolatility", 0)
    if iv and iv > 4.0:                       # now tolerates up to 400% IV
        return -100

    # ─── 1. OTM filter (strike ≤ spot) ───────────────────────────────────
    spot = fundamentals.get("lastPrice")
    if spot is None or row["strike"] > spot:
        return -100

    # ─── 2. Liquidity ─────────────────────────────────────────────────────
    if row.get("bid", 0) == 0 or row.get("openInterest", 0) < 50:
        return -100

    # ─── 3. Compute annualized yield & filter ─────────────────────────
    strike = row["strike"]
    bid    = row.get("bid", 0)
    ask    = row.get("ask", bid)             # fallback to bid if ask missing
    mid    = (bid + ask) / 2
    # time to expiry in years
    T = (pd.to_datetime(row["expirationDate"]) - pd.Timestamp.today()).days / 365
    if T <= 0 or strike <= 0:
        return -100
    annualized_yield = mid / strike / T     # annualized
    # drop anything under 9% annualized return
    if annualized_yield < 0.09:
        return -100

    # ─── 4. Greeks filters ──────────────────────────────────────────────

    # Delta window
    delta_val = row.get("delta")
    if delta_val is None or abs(delta_val) < 1e-6:
        delta_val = -0.25
    if delta_val < -0.80 or delta_val > -0.05:
        return -100

    # Gamma cap unchanged
    gamma_val = row.get("gamma", 0)
    if gamma_val > 0.1:
        return -100

    # Vega cap unchanged
    vega_val = row.get("vega", 0)
    if vega_val > 0.5:
        return -100

    # Theta cap now relaxed
    theta_val = row.get("theta", 0)
    if theta_val > 0.2:                      # now allows decay up to 0.2/day
        return -100

    # ─── 5. Composite scoring ────────────────────────────────────────────
    score = 0
    score += annualized_yield * 100                            # 30% weight
    score += (1 - abs(delta_val + 0.30)) * 50                  # 20% weight
    score += (0.1 - min(gamma_val, 0.1)) * 100                 # 15% weight
    score += (0.5 - min(vega_val, 0.5)) * 100                  # 15% weight
    score += (0.2 - min(abs(theta_val), 0.2)) * 100            # 20% weight (relaxed θ)

    # Market-cap penalty adjusted
    if fundamentals.get("marketCap", 0) < 8e9:
        score -= 20                                           # –20 if < $8B

    # Sector beta adjustment
    beta_adj = sector_beta.get(fundamentals.get("sector", "").lower(), 1)
    score /= beta_adj

    # ─── 6. Earnings proximity penalty ─────────────────────────────────
    edate = events.get("earningsDate")
    if edate and (edate - datetime.today().date()).days < 7:
        score -= 40

    # ─── 7. Final Greeks balance check ─────────────────────────────────
    greeks_balance = abs(delta_val) + gamma_val + vega_val + abs(theta_val)
    if greeks_balance > 1.0:
        score *= 0.8

    return round(score, 2)



def analyze_options_sp100(tickers_sector_list, max_threads=MAX_THREADS, strike_cushion=0.9):
    sector_beta = {
        'technology': 1.2,
        'financials': 1.0,
        'consumer discretionary': 1.1,
        'health care': 0.9,
        'industrials': 1.0,
        'communication services': 1.1,
        'consumer staples': 0.8,
        'energy': 1.3,
        'utilities': 0.7,
        'real estate': 0.7,
        'materials': 1.1,
    }

    all_options = []
    earnings_summary = {}

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = {executor.submit(fetch_data_for_ticker, ti): ti['ticker'] for ti in tickers_sector_list}
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                ticker, fundamentals, earnings_date, options = future.result()
                earnings_summary[ticker] = earnings_date
                if options.empty:
                    continue

                options['score'] = options.apply(
                    lambda row: improved_score_option(row, fundamentals, sector_beta, {"earningsDate": earnings_date}, strike_cushion=strike_cushion),
                    axis=1,
                )
                options['earningsDate'] = earnings_date
                options['sector'] = fundamentals.get('sector', 'Unknown')
                all_options.append(options)
            except Exception as e:
                print(f"Exception processing {ticker}: {e}")

    if all_options:
        df = pd.concat(all_options)
        df_sorted = df.sort_values(by='score', ascending=False)
        
        # Calculate annualized yield for all options
        expiries = pd.to_datetime(df_sorted['expirationDate'])
        today = pd.Timestamp.today()
        days_left = (expiries - today).dt.days.clip(lower=1)  # at least 1 day
        mid = (df_sorted['bid'] + df_sorted['ask']) / 2
        df_sorted['annualized_yield'] = (mid / df_sorted['strike']) / (days_left / 365)
        
        output_cols = [
            'ticker', 'sector', 'strike', 'expirationDate', 'bid', 'ask',
            'lastPrice', 'impliedVolatility', 'delta', 'theta', 'annualized_yield', 'score', 'earningsDate'
        ]
        return df_sorted, earnings_summary
    else:
        return pd.DataFrame(), earnings_summary


# --- Report, Cleanup, Alerts and GPT Integration ---


def save_html_report(economic_events, earnings_summary, options_df, run_time):
    # Get only the top 50 options for the HTML report
    top_options = options_df.head(50)

    # 2) compute days to expiry
    expiries = pd.to_datetime(top_options['expirationDate'])
    today     = pd.Timestamp.today()
    days_left = (expiries - today).dt.days.clip(lower=1)  # at least 1 day

    # 3) compute mid‐premium
    mid = (top_options['bid'] + top_options['ask']) / 2
    
     # 4) annualized yield = (mid/strike) / (days_left/365)
    top_options.loc[:, 'annualized_yield'] = (mid / top_options['strike']) / (days_left / 365)

    html_content = template.render(
        economic_events=economic_events,
        earnings_summary={k: (v.strftime('%Y-%m-%d') if v else None) for k, v in earnings_summary.items()},
        options_data=top_options.to_dict(orient="records"),
        run_time=run_time.strftime('%Y-%m-%d %H:%M:%S')
    )
    if not os.path.exists(REPORT_DIR):
        os.makedirs(REPORT_DIR)
    filename = os.path.join(REPORT_DIR, f"options_report_{run_time.strftime('%Y%m%d_%H%M%S')}.html")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_content)
    return filename


def cleanup_old_reports():
    if not os.path.exists(REPORT_DIR):
        return
    files = sorted(
        [os.path.join(REPORT_DIR, f) for f in os.listdir(REPORT_DIR) if f.endswith(".html")],
        key=os.path.getmtime,
        reverse=True,
    )
    for old_file in files[MAX_REPORTS_TO_KEEP:]:
        os.remove(old_file)


def save_detailed_csv_per_ticker(df, output_dir=DETAILED_DIR):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    ticker_files = []
    for ticker in df['ticker'].unique():
        ticker_df = df[df['ticker'] == ticker].copy()

        ticker_df = ticker_df.rename(columns={
            'expirationDate': 'expiration',
            'impliedVolatility': 'implied volatility',
            'lastPrice': 'last price',
            'openInterest': 'open interest',
        })

        if 'type' not in ticker_df.columns:
            ticker_df['type'] = 'put'  # Assume puts here

        cols = ["expiration", "type", "strike", "last price", "bid", "ask",
                "implied volatility", "delta", "gamma", "theta", "vega", "open interest"]

        for col in cols:
            if col not in ticker_df.columns:
                ticker_df[col] = None

        ticker_df = ticker_df[cols]

        file_path = os.path.join(output_dir, f"{ticker}_option_chain.csv")
        ticker_df.to_csv(file_path, index=False)
        ticker_files.append((ticker, file_path))
    return ticker_files


def batch_analyze_option_chains(ticker_files, batch_size=5, sleep_sec=2):  # Increased batch size and reduced sleep time
    detailed_reports = {}
    for i in range(0, len(ticker_files), batch_size):
        batch = ticker_files[i:i + batch_size]
        for ticker, csv_path in batch:
            print(f"Analyzing detailed option chain for {ticker}...")
            try:
                report = analyze_option_chain(csv_path, ticker)
                detailed_reports[ticker] = report
                print(f"Completed detailed report for {ticker}")
            except Exception as e:
                detailed_reports[ticker] = f"Error generating report: {e}"
                print(f"Error on {ticker}: {e}")
        if i + batch_size < len(ticker_files):
            time.sleep(sleep_sec)
    return detailed_reports


def alert_with_detailed_reports(options_df, pushover_api, detailed_reports):
    alert_rows = options_df[(options_df['annualized_yield'] > 0.08) & (options_df['score'] > 90)]
    for _, row in alert_rows.iterrows():
        ticker = row['ticker']
        message = (f"🔥 Attractive Option Alert:\n"
                   f"{ticker} - Strike: {row['strike']} Exp: {row['expirationDate']}\n"
                   f"Bid: {row['bid']:.2f}, Ask: {row['ask']:.2f}, Score: {row['score']:.2f}, Annulaized Yield: {row['annualized_yield']:.2%}\n\n")
        detailed_report = detailed_reports.get(ticker, None)
        if detailed_report:
            max_len = 700
            detail_excerpt = detailed_report[:max_len] + ("..." if len(detailed_report) > max_len else "")
            message += f"Detailed Analysis:\n{detail_excerpt}"

        # Commented out pushover alert sending
        # try:
        #     pushover_api.send_message(
        #         message=message,
        #         title=f"Options Alert: {ticker}",
        #         user=PUSHOVER_USER_KEY
        #     )
        #     print(f"Sent Pushover alert with detailed report for {ticker}")
        # except Exception as e:
        #     print(f"Failed to send Pushover alert for {ticker}: {e}")
        #print(f"[Pushover alert would be sent here for {ticker}]\n{message}\n")


def run_full_agent():
    run_time = datetime.now()
    sp100_list = load_tickers_and_sectors(TICKERS_CSV_PATH)
    additional_list = load_additional_tickers(ADDITIONAL_TICKERS_CSV_PATH)
    
    # Process additional tickers first
    print("Processing additional tickers first...")
    additional_options_df, additional_earnings = analyze_options_sp100(additional_list, MAX_THREADS, strike_cushion=0.8)
    
    # Then process SP100 tickers
    print("\nProcessing SP100 tickers...")
    sp100_options_df, sp100_earnings = analyze_options_sp100(sp100_list, MAX_THREADS, strike_cushion=0.9)
    
    # Combine the results
    options_df = pd.concat([additional_options_df, sp100_options_df])

    earnings_summary = {**additional_earnings, **sp100_earnings}

    economic_events = get_economic_events_fmp_or_rss()
    html_report_path = save_html_report(economic_events, earnings_summary, options_df, run_time)

    cleanup_old_reports()

    ticker_files = save_detailed_csv_per_ticker(options_df)

    detailed_reports = batch_analyze_option_chains(ticker_files, batch_size=5, sleep_sec=2)

    pushover_api = None
    if PUSHOVER_APP_TOKEN and PUSHOVER_USER_KEY:
        print(f"Debug - PUSHOVER_APP_TOKEN: {PUSHOVER_APP_TOKEN[:5]}...")  # Only show first 5 chars for security
        print(f"Debug - PUSHOVER_USER_KEY: {PUSHOVER_USER_KEY[:5]}...")    # Only show first 5 chars for security
        pushover_api = PushoverAPI(token=PUSHOVER_APP_TOKEN)
        # alert_with_detailed_reports(options_df, pushover_api, detailed_reports)  # <--- Commented out pushover alerts
        print("[Pushover alerts are currently disabled]")
    else:
        print("Pushover keys not set, skipping alerts.")

    economic_events_str = "\n".join([f"• {e}" for e in economic_events]) or "No major US economic events."
    earnings_str = "\n".join(
        [f"• {ticker}: {edate.strftime('%Y-%m-%d') if edate else 'No upcoming earnings'}" for ticker, edate in earnings_summary.items()]
    )
    table_str = options_df.head(10).to_string(index=False) if not options_df.empty else "No suitable options found."

    prompt_template = """
You are a Chief Strategist at a hedge fund specializing in options trading. 
Given these major upcoming US economic events, upcoming earnings, and the top ranked cash-secured put options from a diversified portfolio (50% technology), provide a detailed strategic briefing.

Economic Events:
{economic_events}

Upcoming Earnings:
{earnings_summary}

Top Option Trades:
{options_table}

Explain the attractiveness and risks of the top options considering Greeks, IV, fundamentals, and macro events.
"""
    prompt = PromptTemplate(input_variables=["economic_events", "earnings_summary", "options_table"], template=prompt_template)
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    
    # Create a runnable sequence
    chain = (
        {"economic_events": RunnablePassthrough(), 
         "earnings_summary": RunnablePassthrough(), 
         "options_table": RunnablePassthrough()} 
        | prompt 
        | llm
    )
    
    # Run the chain with the inputs
    summary = chain.invoke({
        "economic_events": economic_events_str,
        "earnings_summary": earnings_str,
        "options_table": table_str
    })

    print(f"HTML report saved: {html_report_path}")
    print("\n=== GPT Strategic Briefing ===\n")
    print(summary.content)

    return summary.content, html_report_path


if __name__ == "__main__":
    if not OPENAI_API_KEY or not FMP_API_KEY:
        print("Please set OPENAI_API_KEY and FMP_API_KEY environment variables before running.")
        sys.exit(1)

    run_full_agent()
