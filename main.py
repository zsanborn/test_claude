"""
Commodities Market Dashboard
Left: top 10 performers (cumulative % return).
Right: Claude-powered news analysis for any clicked point.
"""

import os
import anthropic
from duckduckgo_search import DDGS
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from dash import Dash, Input, Output, State, dcc, html, no_update
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

# --- Style constants ---
FONT = '"Helvetica Neue", Helvetica, Arial, sans-serif'
BG = "#F7F4EF"
CARD_BG = "#FFFFFF"
BORDER = "#DDD8CC"
TEXT = "#2C2C2C"
SUBTEXT = "#888880"
ACCENT = "#4A5568"


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


# --- News & AI helpers ---

def fetch_headlines(commodity: str, end_date: datetime, lookback_days: int = 5) -> list[dict]:
    """
    Fetch top news headlines via DuckDuckGo for a commodity in the N days prior to end_date.
    Returns a list of dicts with 'title', 'domain', and 'date' keys.
    """
    start = end_date - timedelta(days=lookback_days)
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

def make_figure(data: pd.DataFrame, title: str, commodities: list[str]) -> go.Figure:
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
    fig.add_hline(y=0, line_dash="dot", line_color="#BBBBBB", opacity=0.8)
    fig.update_layout(
        title=dict(text=title, font=dict(family=FONT, size=15, weight=300, color=TEXT)),
        xaxis_title="Date",
        yaxis_title="Return (%)",
        legend=dict(orientation="v", x=1.02, y=1, font=dict(family=FONT, size=11, color=TEXT)),
        hovermode="closest",
        template="plotly_white",
        font=dict(family=FONT, color=TEXT),
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        margin=dict(l=50, r=160, t=50, b=50),
        xaxis=dict(gridcolor="#EEEBE4", linecolor=BORDER),
        yaxis=dict(gridcolor="#EEEBE4", linecolor=BORDER),
    )
    return fig


# --- Fetch data once at startup ---
print("Fetching commodity data...")
ALL_PRICES = fetch_data()
print(f"Loaded {len(ALL_PRICES.columns)} commodities.")

# --- Dash app ---
app = Dash(__name__)

_card_style = {
    "backgroundColor": CARD_BG,
    "border": f"1px solid {BORDER}",
    "borderRadius": "6px",
    "padding": "24px",
    "fontFamily": FONT,
    "color": TEXT,
    "height": "100%",
    "boxSizing": "border-box",
}

_placeholder = html.Div(
    style={**_card_style, "display": "flex", "alignItems": "center", "justifyContent": "center", "minHeight": "520px"},
    children=html.P(
        "Click any point on the chart to analyze news headlines.",
        style={"color": SUBTEXT, "fontFamily": FONT, "fontWeight": 400, "fontSize": "14px", "textAlign": "center"},
    ),
)

app.layout = html.Div(
    style={"backgroundColor": BG, "minHeight": "100vh", "padding": "32px 40px", "fontFamily": FONT},
    children=[
        html.H1(
            "Commodities Market Dashboard",
            style={"color": TEXT, "textAlign": "center", "fontWeight": 300, "fontSize": "28px", "marginBottom": "6px", "letterSpacing": "-0.5px"},
        ),
        html.P(
            "Top 10 commodity futures performers. Click any point to surface related news and an AI-generated summary.",
            style={"color": SUBTEXT, "textAlign": "center", "fontWeight": 400, "fontSize": "14px", "marginBottom": "28px"},
        ),

        # Time window selector
        html.Div(
            style={"display": "flex", "justifyContent": "center", "alignItems": "center", "marginBottom": "24px", "gap": "12px"},
            children=[
                html.Label("Time window:", style={"color": SUBTEXT, "fontWeight": 400, "fontSize": "13px"}),
                dcc.RadioItems(
                    id="time-window",
                    options=[{"label": k, "value": v} for k, v in TIME_WINDOWS.items()],
                    value=30,
                    inline=True,
                    style={"color": TEXT, "fontSize": "13px"},
                    inputStyle={"marginRight": "4px", "marginLeft": "14px"},
                ),
            ],
        ),

        # Chart + News panel side by side
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px", "alignItems": "start"},
            children=[
                html.Div(
                    style={**_card_style, "padding": "8px"},
                    children=dcc.Graph(id="top-10", style={"height": "520px"}),
                ),
                dcc.Loading(
                    type="circle",
                    color=ACCENT,
                    children=html.Div(id="news-panel", children=_placeholder, style={"minHeight": "520px"}),
                ),
            ],
        ),

        html.Div(
            id="last-updated",
            style={"color": SUBTEXT, "textAlign": "center", "marginTop": "16px", "fontSize": "12px", "fontWeight": 400},
        ),
    ],
)


# --- Callbacks ---

@app.callback(
    Output("top-10", "figure"),
    Output("last-updated", "children"),
    Input("time-window", "value"),
)
def update_charts(days: int):
    start = datetime.today() - timedelta(days=days)
    returns = compute_returns(ALL_PRICES, start)

    if returns.empty:
        return go.Figure(), "No data available."

    top10 = returns.iloc[-1].sort_values(ascending=False).head(10).index.tolist()
    fig_top = make_figure(returns, "Top 10 Performers", top10)

    updated = f"Data as of {datetime.today().strftime('%B %d, %Y')}  ·  {len(returns.columns)} commodities tracked"
    return fig_top, updated


@app.callback(
    Output("news-panel", "children"),
    Output("top-10", "figure", allow_duplicate=True),
    Input("top-10", "clickData"),
    State("time-window", "value"),
    prevent_initial_call=True,
)
def analyze_click(click_data, days):
    if not click_data:
        return no_update, no_update

    point = click_data["points"][0]
    commodity = point.get("customdata") or point.get("data", {}).get("name", "Unknown")
    click_date = datetime.fromisoformat(point["x"][:10])
    display_date = click_date.strftime("%B %d, %Y")

    headlines = fetch_headlines(commodity, click_date)
    summary = summarize_with_claude(commodity, display_date, headlines)

    # Rebuild chart with selection circle
    start = datetime.today() - timedelta(days=days)
    returns = compute_returns(ALL_PRICES, start)
    top10 = returns.iloc[-1].sort_values(ascending=False).head(10).index.tolist()
    fig_top = make_figure(returns, "Top 10 Performers", top10)
    fig_top.add_trace(go.Scatter(
        x=[click_date],
        y=[point["y"]],
        mode="markers",
        marker=dict(size=14, color="rgba(0,0,0,0)", line=dict(color=TEXT, width=2)),
        showlegend=False,
        hoverinfo="skip",
    ))

    headline_items = [
        html.Li(
            [
                html.Span(f"{h['date']}  ", style={"color": SUBTEXT, "fontSize": "12px", "fontWeight": 400}),
                html.Span(h["title"], style={"color": TEXT, "fontWeight": 400}),
                html.Span(f"  — {h['domain']}" if h["domain"] else "", style={"color": SUBTEXT, "fontSize": "12px"}),
            ],
            style={"marginBottom": "8px", "lineHeight": "1.5"},
        )
        for h in headlines
    ] or [html.Li("No headlines found.", style={"color": SUBTEXT})]

    news_panel = html.Div(
        style=_card_style,
        children=[
            html.H3(
                f"{commodity}  ·  {display_date}",
                style={"color": TEXT, "fontWeight": 300, "fontSize": "17px", "marginTop": 0, "marginBottom": "20px", "letterSpacing": "-0.3px"},
            ),
            html.H4("AI Summary", style={"color": ACCENT, "fontWeight": 300, "fontSize": "13px", "textTransform": "uppercase", "letterSpacing": "1px", "marginBottom": "8px"}),
            html.P(summary, style={"color": TEXT, "fontWeight": 400, "lineHeight": "1.7", "marginBottom": "24px", "fontSize": "14px"}),
            html.H4("Headlines (5 days prior)", style={"color": ACCENT, "fontWeight": 300, "fontSize": "13px", "textTransform": "uppercase", "letterSpacing": "1px", "marginBottom": "10px"}),
            html.Ul(headline_items, style={"paddingLeft": "18px", "fontSize": "13px"}),
        ],
    )
    return news_panel, fig_top


if __name__ == "__main__":
    app.run(debug=False)
