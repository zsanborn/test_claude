"""
Commodities Market Dashboard
Displays top 10 and bottom 10 commodity futures performers over a user-selected time window.
"""

import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from dash import Dash, Input, Output, dcc, html
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


def fetch_data() -> pd.DataFrame:
    """Fetch 1 year of daily closing prices for all commodities."""
    end = datetime.today()
    start = end - timedelta(days=365)
    tickers = list(COMMODITIES.values())

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]

    # Rename columns from ticker to human-readable name
    ticker_to_name = {v: k for k, v in COMMODITIES.items()}
    raw = raw.rename(columns=ticker_to_name)

    # Drop commodities with insufficient data (>50% missing)
    raw = raw.dropna(axis=1, thresh=int(len(raw) * 0.5))
    return raw


def compute_returns(prices: pd.DataFrame, start_date: datetime) -> pd.DataFrame:
    """Normalize prices to % return from start_date."""
    subset = prices[prices.index >= pd.Timestamp(start_date)].copy()
    subset = subset.dropna(axis=1, how="all")

    # Forward-fill minor gaps, then normalize to first valid value
    subset = subset.ffill().bfill()
    normalized = (subset / subset.iloc[0] - 1) * 100
    return normalized


def make_figure(returns: pd.DataFrame, title: str, commodities: list[str]) -> go.Figure:
    fig = go.Figure()
    for name in commodities:
        if name not in returns.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=returns.index,
                y=returns[name].round(2),
                mode="lines",
                name=name,
                hovertemplate="%{fullData.name}<br>%{x|%b %d, %Y}<br><b>%{y:+.2f}%</b><extra></extra>",
            )
        )
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Return (%)",
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

app.layout = html.Div(
    style={"backgroundColor": "#1a1a2e", "minHeight": "100vh", "padding": "20px", "fontFamily": "sans-serif"},
    children=[
        html.H1(
            "Commodities Market Dashboard",
            style={"color": "#e0e0e0", "textAlign": "center", "marginBottom": "4px"},
        ),
        html.P(
            "Top 10 & Bottom 10 futures performers over the selected period.",
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
                dcc.Graph(id="bottom-10", style={"height": "520px"}),
            ],
        ),
        html.Div(
            id="last-updated",
            style={"color": "#555", "textAlign": "center", "marginTop": "12px", "fontSize": "12px"},
        ),
    ],
)


@app.callback(
    Output("top-10", "figure"),
    Output("bottom-10", "figure"),
    Output("last-updated", "children"),
    Input("time-window", "value"),
)
def update_charts(days: int):
    start = datetime.today() - timedelta(days=days)
    returns = compute_returns(ALL_PRICES, start)

    if returns.empty:
        empty = go.Figure()
        return empty, empty, "No data available."

    total_returns = returns.iloc[-1].sort_values(ascending=False)
    top10 = total_returns.head(10).index.tolist()
    bottom10 = total_returns.tail(10).sort_values().index.tolist()

    fig_top = make_figure(returns, "Top 10 Performers", top10)
    fig_bot = make_figure(returns, "Bottom 10 Performers", bottom10)

    updated = f"Data as of {datetime.today().strftime('%B %d, %Y')} | {len(returns.columns)} commodities tracked"
    return fig_top, fig_bot, updated


if __name__ == "__main__":
    app.run(debug=False)
