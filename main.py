"""
Commodities Market Dashboard
Left chart: top 10 performers (cumulative % return).
Right chart: daily price change (derivative) for those same top 10.
Click any point on the left chart to fetch GDELT news headlines for the
5 days prior and get a Claude-powered summary of the likely driving event.
"""

import os
import anthropic
from duckduckgo_search import DDGS
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from dash import Dash, Input, Output, dcc, html, no_update
from datetime import datetime, timedelta

# --- Commodity futures tickers ---
COMMODITIES = {
    "Crude Oil (WTI)": "CL=F",
    "Brent Crude": "BZ=F",
    "Natural Gas": "NG=F",
    "RBOB Gasoline": "RB=F",
    "Heating Oil": "HO=F",
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Copper": "HG=F",
    "Platinum": "PL=F",
    "Palladium": "PA=F",
    "Corn": "ZC=F",
    "Wheat": "ZW=F",
    "Soybeans": "ZS=F",
    "Oats": "ZO=F",
    "Coffee": "KC=F",
    "Sugar": "SB=F",
    "Cotton": "CT=F",
    "Cocoa": "CC=F",
    "Live Cattle": "LE=F",
    "Lean Hogs": "HE=F",
    "Feeder Cattle": "GF=F",
    "Orange Juice": "OJ=F",
    "Lumber": "LBS=F",
}

TIME_WINDOWS = {
    "1 Week": 7,
    "1 Month": 30,
    "3 Months": 90,
    "6 Months": 180,
    "1 Year": 365,
}

# --- Data helpers ---

def fetch_data() -> pd.DataFrame:
    """Fetch 1 year of daily closing prices for all commodities."""
    end = datetime.today()
    start = end - timedelta(days=365)
    tickers = list(COMMODITIES.values())
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    ticker_to_name = {v: k for k, v in COMMODITIES.items()}
    raw = raw.rename(columns=ticker_to_name)
    raw = raw.dropna(axis=1, thresh=int(len(raw) * 0.5))
    return raw


def compute_returns(prices: pd.DataFrame, start_date: datetime) -> pd.DataFrame:
    """Normalize prices to % return from start_date."""
    subset = prices[prices.index >= pd.Timestamp(start_date)].copy()
    subset = subset.dropna(axis=1, how="all").ffill().bfill()
    return (subset / subset.iloc[0] - 1) * 100


def compute_daily_changes(prices: pd.DataFrame, start_date: datetime) -> pd.DataFrame:
    """Day-over-day price change (first difference) as % of previous close."""
    subset = prices[prices.index >= pd.Timestamp(start_date)].copy()
    subset = subset.dropna(axis=1, how="all").ffill().bfill()
    return subset.pct_change() * 100


# --- News & AI helpers ---

def fetch_headlines(commodity: str, end_date: datetime, lookback_days: int = 5) -> list[dict]:
    """
    Fetch top news headlines via DuckDuckGo for a commodity in the N days prior to end_date.
    Returns a list of dicts with 'title', 'domain', and 'date' keys.
    """
    start = end_date - timedelta(days=lookback_days)

    # Pick the broadest timelimit that still covers the clicked date
    days_ago = (datetime.today() - end_date).days
    if days_ago <= 7:
        timelimit = "w"
    elif days_ago <= 30:
        timelimit = "m"
    else:
        timelimit = "y"

    try:
        results = DDGS().news(
            keywords=f"{commodity} commodity price",
            timelimit=timelimit,
            max_results=50,
        )
        headlines = []
        for r in results:
            article_date = datetime.fromisoformat(r["date"][:10])
            if start <= article_date <= end_date:
                headlines.append({
                    "title": r.get("title", "").strip(),
                    "domain": r.get("source", ""),
                    "date": article_date.strftime("%Y-%m-%d"),
                })
        return headlines[:10]
    except Exception as e:
        return [{"title": f"Error fetching news: {e}", "domain": "", "date": ""}]


def summarize_with_claude(commodity: str, click_date: str, headlines: list[dict]) -> str:
    """Use Claude Haiku to identify the likely event driving a commodity price movement."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return "Set the ANTHROPIC_API_KEY environment variable to enable AI summaries."

    headlines_text = "\n".join(
        f"- [{h['date']}] {h['title']} ({h['domain']})" for h in headlines
    )
    if not headlines_text:
        return "No headlines found for this commodity and date range."

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        messages=[
            {
                "role": "user",
                "content": (
                    f"The commodity '{commodity}' had a notable price movement around {click_date}. "
                    f"Below are the top news headlines from the 5 days prior:\n\n{headlines_text}\n\n"
                    f"In 2-3 concise sentences, identify the most likely event or factor from these "
                    f"headlines that caused the price movement. Be specific about the event and its "
                    f"probable impact on supply or demand."
                ),
            }
        ],
    )
    return message.content[0].text


# --- Chart builder ---

def make_figure(data: pd.DataFrame, title: str, commodities: list[str], y_label: str = "Return (%)") -> go.Figure:
    fig = go.Figure()
    for name in commodities:
        if name not in data.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[name].round(2),
                mode="lines",
                name=name,
                customdata=[name] * len(data),
                hovertemplate="%{fullData.name}<br>%{x|%b %d, %Y}<br><b>%{y:+.2f}%</b><extra></extra>",
            )
        )
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=y_label,
        legend=dict(orientation="v", x=1.02, y=1),
        hovermode="x unified",
        template="plotly_dark",
        margin=dict(l=50, r=180, t=50, b=50),
    )
    return fig


# --- Fetch data once at startup ---
print("Fetching commodity data...")
ALL_PRICES = fetch_data()
print(f"Loaded {len(ALL_PRICES.columns)} commodities.")

# --- Dash app ---
app = Dash(__name__)

_panel_style = {
    "backgroundColor": "#16213e",
    "border": "1px solid #2a2a5a",
    "borderRadius": "8px",
    "padding": "20px",
    "marginTop": "20px",
    "color": "#e0e0e0",
}

app.layout = html.Div(
    style={"backgroundColor": "#1a1a2e", "minHeight": "100vh", "padding": "20px", "fontFamily": "sans-serif"},
    children=[
        html.H1(
            "Commodities Market Dashboard",
            style={"color": "#e0e0e0", "textAlign": "center", "marginBottom": "4px"},
        ),
        html.P(
            "Left: top 10 performers (cumulative return). Right: daily price change for those same commodities.",
            style={"color": "#888", "textAlign": "center", "marginBottom": "20px"},
        ),
        html.Div(
            style={"display": "flex", "justifyContent": "center", "marginBottom": "24px"},
            children=[
                html.Label("Time Window: ", style={"color": "#ccc", "marginRight": "12px", "alignSelf": "center"}),
                dcc.RadioItems(
                    id="time-window",
                    options=[{"label": k, "value": v} for k, v in TIME_WINDOWS.items()],
                    value=30,
                    inline=True,
                    style={"color": "#ccc"},
                    inputStyle={"marginRight": "4px", "marginLeft": "16px"},
                ),
            ],
        ),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"},
            children=[
                dcc.Graph(id="top-10", style={"height": "520px"}),
                dcc.Graph(id="top-10-derivative", style={"height": "520px"}),
            ],
        ),
        html.Div(
            id="last-updated",
            style={"color": "#555", "textAlign": "center", "marginTop": "12px", "fontSize": "12px"},
        ),

        # --- News Analysis Panel ---
        html.P(
            "Click any point on the left chart to analyze news headlines around that date.",
            style={"color": "#666", "textAlign": "center", "marginTop": "24px", "fontSize": "13px"},
        ),
        dcc.Loading(
            type="circle",
            color="#7b8cde",
            children=html.Div(id="news-panel"),
        ),
    ],
)


# --- Callbacks ---

@app.callback(
    Output("top-10", "figure"),
    Output("top-10-derivative", "figure"),
    Output("last-updated", "children"),
    Input("time-window", "value"),
)
def update_charts(days: int):
    start = datetime.today() - timedelta(days=days)
    returns = compute_returns(ALL_PRICES, start)

    if returns.empty:
        empty = go.Figure()
        return empty, empty, "No data available."

    top10 = returns.iloc[-1].sort_values(ascending=False).head(10).index.tolist()

    fig_top = make_figure(returns, "Top 10 Performers", top10)

    daily = compute_daily_changes(ALL_PRICES, start)
    fig_deriv = make_figure(daily, "Daily Price Change — Top 10 Performers", top10, y_label="Daily Change (%)")
    fig_deriv.update_traces(
        hovertemplate="%{fullData.name}<br>%{x|%b %d, %Y}<br><b>%{y:+.2f}%/day</b><extra></extra>"
    )

    updated = f"Data as of {datetime.today().strftime('%B %d, %Y')} | {len(returns.columns)} commodities tracked"
    return fig_top, fig_deriv, updated


@app.callback(
    Output("news-panel", "children"),
    Input("top-10", "clickData"),
    prevent_initial_call=True,
)
def analyze_click(click_data):
    if not click_data:
        return no_update

    point = click_data["points"][0]
    commodity = point.get("customdata") or point.get("data", {}).get("name", "Unknown")
    date_str = point["x"]  # ISO format: YYYY-MM-DD or YYYY-MM-DD HH:MM:SS
    click_date = datetime.fromisoformat(date_str[:10])
    display_date = click_date.strftime("%B %d, %Y")

    headlines = fetch_headlines(commodity, click_date)
    summary = summarize_with_claude(commodity, display_date, headlines)

    headline_items = [
        html.Li(
            [
                html.Span(f"[{h['date'][:4]}-{h['date'][4:6]}-{h['date'][6:]}] " if h["date"] else "",
                          style={"color": "#888", "fontSize": "12px"}),
                html.Span(h["title"], style={"color": "#c8d0e7"}),
                html.Span(f"  — {h['domain']}" if h["domain"] else "",
                          style={"color": "#555", "fontSize": "12px"}),
            ],
            style={"marginBottom": "6px"},
        )
        for h in headlines
    ] or [html.Li("No headlines found.", style={"color": "#888"})]

    return html.Div(
        style=_panel_style,
        children=[
            html.H3(
                f"News Analysis: {commodity} around {display_date}",
                style={"color": "#7b8cde", "marginTop": 0},
            ),
            html.H4("Claude's Summary", style={"color": "#aab4d4", "marginBottom": "8px"}),
            html.P(summary, style={"color": "#e0e0e0", "lineHeight": "1.6", "marginBottom": "20px"}),
            html.H4(f"Top Headlines (5 days prior)", style={"color": "#aab4d4", "marginBottom": "8px"}),
            html.Ul(headline_items, style={"paddingLeft": "20px", "lineHeight": "1.8"}),
        ],
    )


if __name__ == "__main__":
    app.run(debug=False)
