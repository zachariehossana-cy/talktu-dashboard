"""
talktü — Financial Cockpit
Built for: Zacharie Hossana Dayang, Finance Associate Intern, talktü
Background used to shape this app: Economics/Finance (2 Masters) + Data Science
(AIMS Cameroon / Univ. of Buea).

HOW THIS VERSION IS DIFFERENT (and why):

1. Safe Excel reading — finds rows by LABEL TEXT, not row number, so the app
   doesn't silently break if a row shifts in the workbook.

2. "Data Mode" switch (NEW) — a single toggle in the sidebar:
   "Assumptions (not yet confirmed)" vs "Real / confirmed data".
   This solves a real problem: right now the numbers in the workbook are
   placeholders, but they'll be replaced by real numbers soon. Instead of
   deleting features because the data is fake today, the app just shows a
   clear warning banner while in Assumptions mode, and removes it once you
   flip to Real mode — no code changes needed later.

3. Break-even calculator — "how many more schools do we need to stop
   losing money?"

4. Sensitivity tornado chart — shows which lever (B2B growth, B2C growth,
   or costs) moves year-end cash the most.

5. Trend forecast (RESTORED, but done more honestly) — a straight-line
   trend instead of a curvy polynomial, because a curved line fit to only
   12 months of data can shoot off in unrealistic directions. A shaded
   band shows the normal range of error, so it reads as "a rough guide",
   not "the future".

6. Monte Carlo risk simulation — uses a triangular distribution built from
   your own Conservative/Base/Optimistic bounds, not a made-up spread.
   In Assumptions mode it's clearly labeled "based on placeholder data,
   for structure/practice only". In Real mode that caption disappears.

7. Plain-English summary line + % of simulations ending cash-negative —
   turns numbers into one clear sentence, no interpretation needed.

To run: streamlit run app.py
Needs "talktu_financial_model(1).xlsx" in the same folder, with a
"P&L Summary" sheet.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ----------------------------------------------------------------------------
# 1. PAGE SETUP
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="talktü — Financial Cockpit",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY = "#1B2A4A"
GOLD = "#B8860B"

st.markdown(
    """
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e222d; padding: 15px; border-radius: 10px; border: 1px solid #2e364f; }
    </style>
    """,
    unsafe_allow_html=True,
)

FILE_PATH = "talktu_financial_model(1).xlsx"
SHEET_NAME = "P&L Summary"

# Keywords used to find each row by its label text (safer than row numbers).
ROW_LABELS = {
    "b2c_revenue": "Parent App",
    "b2b_revenue": "School",
    "assessment_revenue": "Assessment",
    "grant_revenue": "Grant",
    "salaries": "Salar",
    "tech": "Tech",
    "marketing": "Marketing",
    "admin": "Admin",
    "contingency": "Conting",
}


# ----------------------------------------------------------------------------
# 2. SAFE DATA LOADING (label-based, not row-number-based)
# ----------------------------------------------------------------------------
def find_row(raw: pd.DataFrame, keyword: str) -> int:
    """Return the index of the first row whose first column contains `keyword`."""
    col0 = raw.iloc[:, 0].astype(str)
    matches = col0[col0.str.contains(keyword, case=False, na=False)]
    if matches.empty:
        raise ValueError(f"Could not find a row containing '{keyword}' in the P&L sheet.")
    return matches.index[0]


def find_months_row(raw: pd.DataFrame) -> int:
    """The month header row (e.g. 'Aug-26', 'Sep-26', ...) has NO text label
    in column A in this workbook. So instead of searching column A, scan
    every row for cells matching a 'Mon-YY' pattern and return the first
    row where at least half the 12 monthly columns match."""
    import re
    pattern = re.compile(r"^[A-Za-z]{3}-\d{2}$")
    for i in range(len(raw)):
        row_vals = raw.iloc[i, 1:13]
        # pd.notna() check first: blank cells can stay as NaN even after a
        # string conversion in some pandas versions, which would otherwise
        # crash a plain string match here.
        hits = sum(1 for v in row_vals if pd.notna(v) and pattern.match(str(v).strip()))
        if hits >= 6:
            return i
    raise ValueError("Could not find the month header row (expected values like 'Aug-26').")


@st.cache_data
def load_financial_data(file_path: str) -> pd.DataFrame:
    raw = pd.read_excel(file_path, sheet_name=SHEET_NAME, header=None)

    months_row = find_months_row(raw)
    months = raw.iloc[months_row, 1:13].values

    def row_values(keyword: str) -> np.ndarray:
        r = find_row(raw, keyword)
        return raw.iloc[r, 1:13].values.astype(float)

    df = pd.DataFrame({
        "Month": months,
        "B2C App Revenue": row_values(ROW_LABELS["b2c_revenue"]),
        "B2B School Revenue": row_values(ROW_LABELS["b2b_revenue"]),
        "Assessment Revenue": row_values(ROW_LABELS["assessment_revenue"]),
        "Grants & Partnerships": row_values(ROW_LABELS["grant_revenue"]),
        "Salaries & Benefits": row_values(ROW_LABELS["salaries"]),
        "Tech & Hosting": row_values(ROW_LABELS["tech"]),
        "Marketing": row_values(ROW_LABELS["marketing"]),
        "Admin & Legal": row_values(ROW_LABELS["admin"]),
        "Contingency": row_values(ROW_LABELS["contingency"]),
    })
    return df


# ----------------------------------------------------------------------------
# 3. FINANCIAL HELPERS
# ----------------------------------------------------------------------------
OPEX_COLS = ["Salaries & Benefits", "Tech & Hosting", "Marketing", "Admin & Legal", "Contingency"]


def build_scenario(df_raw: pd.DataFrame, b2b_mult: float, b2c_mult: float, opex_mult: float,
                    starting_cash: float) -> pd.DataFrame:
    """Apply growth/cost multipliers and compute revenue, costs, and cash over time."""
    df = df_raw.copy()
    df["B2B School Revenue"] *= (1 + b2b_mult)
    df["B2C App Revenue"] *= (1 + b2c_mult)
    for col in OPEX_COLS:
        df[col] *= (1 + opex_mult)

    df["Total Revenue"] = (
        df["B2C App Revenue"] + df["B2B School Revenue"]
        + df["Assessment Revenue"] + df["Grants & Partnerships"]
    )
    df["Total Costs"] = df[OPEX_COLS].sum(axis=1)
    df["Net Result"] = df["Total Revenue"] - df["Total Costs"]
    df["Cash Balance"] = starting_cash + df["Net Result"].cumsum()
    return df


def plain_english_summary(runway_months: float, funding_gap: float) -> str:
    """Turn the numbers into one short, human sentence — no finance jargon."""
    if runway_months >= 12:
        cash_line = "Cash looks healthy for the full year at this pace."
    elif runway_months >= 6:
        cash_line = f"You have about {runway_months:.1f} months of cash left — worth watching."
    else:
        cash_line = f"Only about {runway_months:.1f} months of cash left — this needs attention soon."

    if funding_gap > 0:
        gap_line = f" You may need roughly ₦{funding_gap/1e6:,.1f}M in extra funding to stay safe all year."
    else:
        gap_line = " No extra funding looks needed to stay cash-positive for the year."

    return cash_line + gap_line


def break_even_schools(df: pd.DataFrame, price_per_school: float) -> int:
    """Rough estimate: how many extra schools (at current price) would close
    the average monthly gap between costs and revenue?"""
    avg_gap = (df["Total Costs"] - df["Total Revenue"]).mean()
    if avg_gap <= 0 or price_per_school <= 0:
        return 0
    return int(np.ceil(avg_gap / price_per_school))


def tornado_data(df_raw: pd.DataFrame, starting_cash: float, swing: float = 0.20):
    """For each lever, show what year-end cash looks like if that lever alone
    moves up or down by `swing` (default ±20%), holding others at 0%."""
    def end_cash(b2b, b2c, opex):
        return build_scenario(df_raw, b2b, b2c, opex, starting_cash)["Cash Balance"].iloc[-1]

    base_end_cash = end_cash(0, 0, 0)
    rows = [
        ("B2B School Growth", end_cash(-swing, 0, 0), end_cash(swing, 0, 0)),
        ("B2C App Growth", end_cash(0, -swing, 0), end_cash(0, swing, 0)),
        ("Operating Costs", end_cash(0, 0, -swing), end_cash(0, 0, swing)),
    ]
    out = pd.DataFrame(rows, columns=["Lever", "Low Case", "High Case"])
    out["Swing (₦M)"] = (out["High Case"] - out["Low Case"]).abs() / 1e6
    out = out.sort_values("Swing (₦M)", ascending=True)
    return out, base_end_cash


def monte_carlo(df_raw: pd.DataFrame, starting_cash: float,
                 b2b_low: float, b2b_high: float, b2c_low: float, b2c_high: float,
                 opex_low: float, opex_high: float, n: int = 1000) -> np.ndarray:
    """Simulate year-end cash `n` times using a triangular distribution built
    from your own Conservative/Base/Optimistic bounds (mode = 0%, i.e. Base
    case is treated as most likely)."""
    results = []
    for _ in range(n):
        b2b = np.random.triangular(b2b_low, 0, b2b_high)
        b2c = np.random.triangular(b2c_low, 0, b2c_high)
        opex = np.random.triangular(opex_low, 0, opex_high)
        end_cash = build_scenario(df_raw, b2b, b2c, opex, starting_cash)["Cash Balance"].iloc[-1]
        results.append(end_cash / 1e6)
    return np.array(results)


def linear_forecast(values: np.ndarray, n_future: int = 6):
    """Fit a straight line (not a curve) through the historical values and
    extrapolate `n_future` steps forward. A straight line is a more honest
    choice than a curved polynomial when there are only 12 data points —
    curves can swing wildly once extrapolated beyond the data they were
    fit on. Also returns a simple ± error band based on how far actual
    points sit from the fitted line historically."""
    x = np.arange(len(values))
    coeffs = np.polyfit(x, values, 1)
    trend = np.poly1d(coeffs)

    residuals = values - trend(x)
    error_band = residuals.std()

    future_x = np.arange(len(values), len(values) + n_future)
    forecast = trend(future_x)
    return forecast, error_band, coeffs[0]  # coeffs[0] = slope (₦/month trend)


# ----------------------------------------------------------------------------
# 4. MAIN APP
# ----------------------------------------------------------------------------
try:
    df_raw = load_financial_data(FILE_PATH)

    # --- Sidebar: Data Mode (NEW) ---
    st.sidebar.title("🎮 Control Panel")

    data_mode = st.sidebar.radio(
        "📁 Data Mode",
        ["Assumptions (not yet confirmed)", "Real / confirmed data"],
        help="Switch this once Sarah Andino gives you the real numbers. "
             "It turns off the 'placeholder data' warnings across the app — "
             "no need to edit the code.",
    )
    is_real = data_mode.startswith("Real")

    st.sidebar.markdown("---")
    st.sidebar.caption("Scenarios & model settings")

    scenario = st.sidebar.radio(
        "📌 Pick a scenario",
        ["Custom (move sliders)", "Conservative", "Base Case", "Optimistic"],
    )

    # Mirrors the Conservative/Base/Optimistic columns in your Assumptions tab.
    B2B_BOUNDS = (-0.30, 0.35)
    B2C_BOUNDS = (-0.20, 0.25)
    OPEX_BOUNDS = (-0.05, 0.15)

    if scenario == "Conservative":
        b2b_mult, b2c_mult, opex_mult = B2B_BOUNDS[0], B2C_BOUNDS[0], OPEX_BOUNDS[1]
    elif scenario == "Optimistic":
        b2b_mult, b2c_mult, opex_mult = B2B_BOUNDS[1], B2C_BOUNDS[1], OPEX_BOUNDS[0]
    elif scenario == "Base Case":
        b2b_mult, b2c_mult, opex_mult = 0.0, 0.0, 0.0
    else:
        st.sidebar.subheader("⚡ Sensitivity sliders")
        b2b_mult = st.sidebar.slider("B2B School Growth (%)", -50, 100, 0, 5) / 100.0
        b2c_mult = st.sidebar.slider("B2C App Growth (%)", -50, 100, 0, 5) / 100.0
        opex_mult = st.sidebar.slider("Cost Change (%)", -30, 50, 0, 5) / 100.0

    st.sidebar.markdown("---")
    starting_cash = st.sidebar.number_input("Starting Cash (₦)", value=136_214_034, step=5_000_000)
    price_per_school = st.sidebar.number_input(
        "Price per School (₦/month)", value=225_000, step=5_000,
        help="Used for the break-even calculator below.",
    )

    # --- Core numbers ---
    df = build_scenario(df_raw, b2b_mult, b2c_mult, opex_mult, starting_cash)

    total_revenue = df["Total Revenue"].sum()
    total_costs = df["Total Costs"].sum()
    net_result = df["Net Result"].sum()
    avg_burn = (df["Total Costs"] - df["Total Revenue"]).mean()
    runway_months = df["Cash Balance"].iloc[-1] / avg_burn if avg_burn > 0 else 12.0
    funding_gap = max(0, -df["Cash Balance"].min())

    # --- Header ---
    st.title("🚀 talktü — Financial Cockpit")
    st.caption("Aug 2026 – Jul 2027 | Full model, breakdowns, forecast & risk simulation")

    if is_real:
        st.success("✅ Data Mode: **Real / confirmed data** — figures below reflect talktü's actual numbers.")
    else:
        st.warning(
            "⚠️ Data Mode: **Assumptions (not yet confirmed)** — the figures below are placeholders "
            "waiting on real numbers from Sarah Andino. Treat forecasts and risk results as a "
            "practice run of the model's structure, not real predictions. Switch the Data Mode "
            "toggle in the sidebar once real numbers are in."
        )

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Yearly Revenue", f"₦{total_revenue/1e6:,.1f}M", delta=f"{b2b_mult*100:.0f}% B2B")
    k2.metric("Yearly Costs", f"₦{total_costs/1e6:,.1f}M", delta=f"{opex_mult*100:.0f}% Cost change", delta_color="inverse")
    k3.metric("Net Result", f"₦{net_result/1e6:,.1f}M")
    k4.metric("Cash Runway", f"{max(0, runway_months):.1f} months")
    k5.metric("Funding Gap", f"₦{funding_gap/1e6:,.1f}M", delta_color="inverse")

    st.info("💡 " + plain_english_summary(max(0, runway_months), funding_gap))

    st.markdown("---")

    t1, t2, t3, t4, t5, t6 = st.tabs([
        "🍕 Revenue & Cost Mix",
        "📈 Cash Trajectory",
        "🎯 What Matters Most",
        "🔮 Trend Forecast",
        "🎲 Risk Simulation",
        "📄 Data & Export",
    ])

    # --- TAB 1: Pie charts ---
    with t1:
        st.subheader("Where money comes from, where it goes")
        c1, c2 = st.columns(2)
        with c1:
            rev_data = {
                "B2C App": df["B2C App Revenue"].sum(),
                "B2B Schools": df["B2B School Revenue"].sum(),
                "Assessments": df["Assessment Revenue"].sum(),
                "Grants": df["Grants & Partnerships"].sum(),
            }
            fig = px.pie(names=list(rev_data.keys()), values=list(rev_data.values()),
                         title="<b>Revenue sources</b>", hole=0.4,
                         color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_traces(textinfo="percent+label")
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, width="stretch")
        with c2:
            opex_data = {col: df[col].sum() for col in OPEX_COLS}
            fig = px.pie(names=list(opex_data.keys()), values=list(opex_data.values()),
                         title="<b>Cost breakdown</b>", hole=0.4,
                         color_discrete_sequence=px.colors.qualitative.Set3)
            fig.update_traces(textinfo="percent+label")
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, width="stretch")

    # --- TAB 2: Trajectory + break-even ---
    with t2:
        st.subheader("Monthly performance & cash balance over time")
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=df["Month"], y=df["Total Revenue"], name="Revenue", marker_color="#2ecc71"), secondary_y=False)
        fig.add_trace(go.Bar(x=df["Month"], y=df["Total Costs"], name="Costs", marker_color="#e74c3c"), secondary_y=False)
        fig.add_trace(go.Scatter(x=df["Month"], y=df["Cash Balance"], name="Cash Balance",
                                  line=dict(color=GOLD, width=4)), secondary_y=True)
        fig.update_layout(barmode="group", template="plotly_dark", height=450)
        fig.update_yaxes(title_text="Monthly flows (₦)", secondary_y=False)
        fig.update_yaxes(title_text="Cash on hand (₦)", secondary_y=True)
        st.plotly_chart(fig, width="stretch")

        st.markdown("#### 🏫 Break-even calculator")
        extra_schools = break_even_schools(df, price_per_school)
        if extra_schools == 0:
            st.success("At this scenario, revenue already covers costs on average — no extra schools needed.")
        else:
            st.warning(
                f"On average, you'd need about **{extra_schools} more schools** "
                f"(at ₦{price_per_school:,.0f}/month each) to close the monthly gap between costs and revenue."
            )

    # --- TAB 3: Sensitivity tornado ---
    with t3:
        st.subheader("Which lever moves year-end cash the most?")
        st.caption("Each lever is tested alone, moved ±20%, while the others stay at 0%.")
        tornado_df, base_cash = tornado_data(df_raw, starting_cash)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=tornado_df["Lever"], x=tornado_df["Swing (₦M)"], orientation="h",
            marker_color=GOLD,
        ))
        fig.update_layout(
            template="plotly_dark", height=350,
            title=f"Impact on year-end cash (Base case ≈ ₦{base_cash/1e6:,.1f}M)",
            xaxis_title="Swing in year-end cash (₦ Millions)",
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            f"📌 Biggest lever: **{tornado_df.iloc[-1]['Lever']}** — focus your energy here first, "
            "since it swings the year-end cash balance the most."
        )

    # --- TAB 4: Trend forecast (restored, done more honestly) ---
    with t4:
        st.subheader("🔮 Where is revenue heading, 6 months past this model?")
        if not is_real:
            st.warning(
                "⚠️ This forecast is built on only 12 months of **placeholder** data. "
                "Treat the shape of the chart as a demo of how forecasting will work — "
                "not as a real prediction. It will become genuinely useful once the "
                "Assumptions tab holds real numbers, and even more so once a few months "
                "of real actuals exist."
            )
        else:
            st.caption(
                "Straight-line trend based on 12 months of real data, with a shaded band "
                "showing the normal month-to-month error. Refine as more months come in — "
                "12 points is still a short history for forecasting."
            )

        forecast, error_band, monthly_trend = linear_forecast(df["Total Revenue"].values, n_future=6)
        future_months = ["Aug-27", "Sep-27", "Oct-27", "Nov-27", "Dec-27", "Jan-28"]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(df["Month"]), y=list(df["Total Revenue"]),
                                  name="Actual / Modeled Revenue", line=dict(color="#2ecc71", width=3)))
        fig.add_trace(go.Scatter(x=future_months, y=forecast, name="Trend Forecast",
                                  line=dict(color="#00d2d3", width=3, dash="dash")))
        fig.add_trace(go.Scatter(
            x=future_months + future_months[::-1],
            y=list(forecast + error_band) + list(forecast - error_band)[::-1],
            fill="toself", fillcolor="rgba(0,210,211,0.15)", line=dict(width=0),
            name="Normal error range", showlegend=True,
        ))
        fig.update_layout(template="plotly_dark", height=420, xaxis_title="Month", yaxis_title="Revenue (₦)")
        st.plotly_chart(fig, width="stretch")

        trend_word = "growing" if monthly_trend > 0 else "shrinking"
        st.caption(f"📈 Revenue trend is roughly **{trend_word}** by about ₦{abs(monthly_trend)/1e6:,.2f}M per month, on this data.")

    # --- TAB 5: Monte Carlo ---
    with t5:
        st.subheader("🎲 Risk simulation (1,000 runs)")
        if not is_real:
            st.warning(
                "⚠️ Based on **placeholder** assumptions — treat this as a demo of how the "
                "risk simulation works, not a real risk read. It will be genuinely useful "
                "once the Conservative/Base/Optimistic bounds reflect real numbers."
            )
        else:
            st.caption(
                "Uses a triangular distribution built from your Conservative/Base/Optimistic "
                "assumptions — Base case is treated as the most likely outcome."
            )
        sim = monte_carlo(df_raw, starting_cash, *B2B_BOUNDS, *B2C_BOUNDS, *OPEX_BOUNDS)
        fig = px.histogram(sim, nbins=30, title="Distribution of year-end cash (₦ Millions)")
        fig.update_traces(marker_color="#9b59b6")
        fig.update_layout(template="plotly_dark", xaxis_title="Year-end cash (₦M)", yaxis_title="Number of simulations")
        st.plotly_chart(fig, width="stretch")

        pct_negative = (sim < 0).mean() * 100
        st.warning(f"⚠️ In this simulation, cash went negative in **{pct_negative:.0f}%** of the 1,000 runs.")

    # --- TAB 6: Data table & export ---
    with t6:
        st.subheader("📋 Full monthly table")
        st.dataframe(df.style.format("{:,.0f}", subset=df.columns[1:]), width="stretch")

        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download table as CSV",
            data=csv_data,
            file_name="talktu_financial_model_export.csv",
            mime="text/csv",
        )

except FileNotFoundError:
    st.error(f"File not found: '{FILE_PATH}'.")
    st.info("Put this Excel file in the same folder as app.py, then reload the page.")
except Exception as e:
    st.error(f"Something went wrong: {e}")
    st.info("Check that the 'P&L Summary' sheet still has the same row labels described at the top of this file.")
