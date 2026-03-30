"""
Stack Overflow Questions Over Time — with AI Milestone Markers
============================================================
Exercise 01: AI-Assisted Data Visualization

Data source: Stack Exchange Data Explorer (SEDE)
URL: https://data.stackexchange.com/stackoverflow/query/new

How to get the data:
  1. Go to https://data.stackexchange.com/stackoverflow/query/new
  2. Run the SQL query below (also saved as sede_query.sql)
  3. Download as CSV → save as "so_questions_per_month.csv"

Alternatively, the script can fetch via the SEDE API (see fetch_from_sede()).
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import requests
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


# ─── SEDE SQL Queryyy ──────────────────────────────────────────────────────────
# Paste this into https://data.stackexchange.com/stackoverflow/query/new
SEDE_SQL = """
SELECT
    FORMAT(CreationDate, 'yyyy-MM') AS [Month],
    COUNT(*) AS [Questions]
FROM Posts
WHERE PostTypeId = 1
  AND CreationDate >= '2008-07-01'
  AND CreationDate < DATEADD(MONTH, DATEDIFF(MONTH, 0, GETDATE()), 0)
GROUP BY FORMAT(CreationDate, 'yyyy-MM')
ORDER BY [Month]
"""


# ─── AI Milestones ────────────────────────────────────────────────────────────
AI_MILESTONES = [
    {
        "date": "2017-06",
        "label": "Transformer\npaper",
        "color": "#7c6cf0",
        "short": "Transformer",
    },
    {
        "date": "2022-11",
        "label": "ChatGPT\nlaunches",
        "color": "#e05252",
        "short": "ChatGPT",
    },
    {
        "date": "2023-03",
        "label": "GPT-4",
        "color": "#d4633f",
        "short": "GPT-4",
    },
    {
        "date": "2023-08",
        "label": "Cursor &\nAI IDEs",
        "color": "#2a9d8f",
        "short": "Cursor / AI IDEs",
    },
    {
        "date": "2025-02",
        "label": "Claude Code\n& Copilot Agent",
        "color": "#264653",
        "short": "Claude Code / Copilot Agent",
    },
]


# ─── Data fetching ────────────────────────────────────────────────────────────

def fetch_from_sede(query_id: int = None) -> pd.DataFrame:
    """
    Fetch results from a saved SEDE query via their undocumented JSON API.

    Steps to get a query_id:
      1. Visit https://data.stackexchange.com/stackoverflow/query/new
      2. Paste SEDE_SQL above, click Run, then Save
      3. Note the query ID from the URL  (e.g. /query/1234567)
      4. Pass that ID here.

    The SEDE API returns a JSON with a 'resultSets' key.
    """
    if query_id is None:
        raise ValueError(
            "Pass a saved SEDE query ID.\n"
            "  1. Go to https://data.stackexchange.com/stackoverflow/query/new\n"
            f"  2. Paste the SQL, Save, note the ID in the URL.\n"
            "  3. Call fetch_from_sede(query_id=YOUR_ID)"
        )

    url = f"https://data.stackexchange.com/stackoverflow/query/run/{query_id}"
    print(f"Fetching from SEDE query {query_id} …")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    data = resp.json()
    cols = [c["name"] for c in data["resultSets"][0]["columns"]]
    rows = data["resultSets"][0]["rows"]
    return pd.DataFrame(rows, columns=cols)


def load_from_csv(path: str = "so_questions_per_month.csv") -> pd.DataFrame:
    """Load from a CSV exported from SEDE (the Download button)."""
    print(f"Loading data from {path} …")
    df = pd.read_csv(path)
    return df


def load_demo_data() -> pd.DataFrame:
    """
    Synthetic demo data that approximates the real SO trend.
    Replace with real SEDE data for accurate results.
    """
    import numpy as np
    np.random.seed(42)

    months = pd.date_range("2008-09", "2025-01", freq="MS")
    n = len(months)

    # Parametric curve mimicking the real SO trajectory
    t = np.linspace(0, 1, n)
    peak_idx = int(0.40 * n)

    growth = np.exp(3.5 * t[:peak_idx]) * 5000 + 5000
    decay  = 205000 * np.exp(-2.2 * (t[peak_idx:] - t[peak_idx]))
    base   = np.concatenate([growth, decay])
    noise  = np.random.normal(0, base * 0.06)
    values = np.clip(base + noise, 5000, None).astype(int)

    df = pd.DataFrame({"Month": months.strftime("%Y-%m"), "Questions": values})
    print("⚠️  Using demo data — replace with real SEDE export for accuracy.")
    return df


# ─── Plotting ─────────────────────────────────────────────────────────────────

def make_chart(df: pd.DataFrame, save_path: str = "so_questions_ai_milestones.png"):
    """Reproduce the original chart and overlay AI milestone markers."""

    # Parse dates
    df["Date"] = pd.to_datetime(df["Month"])
    df = df.sort_values("Date").reset_index(drop=True)

    # 12-month rolling average
    df["Rolling12"] = df["Questions"].rolling(12, center=False).mean()

    # ── Figure setup ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 7))
    fig.patch.set_facecolor("#f8f7f4")
    ax.set_facecolor("#f8f7f4")

    # ── Base series ───────────────────────────────────────────────────────────
    ax.plot(df["Date"], df["Questions"],
            color="#4c8cbf", linewidth=0.9, alpha=0.75, label="Monthly questions")
    ax.plot(df["Date"], df["Rolling12"],
            color="#e07b39", linewidth=2.6, label="12-month rolling average")

    # ── Peak annotation ───────────────────────────────────────────────────────
    peak_row = df.loc[df["Questions"].idxmax()]
    ax.annotate(
        f"Peak\n{peak_row['Month']}\n{peak_row['Questions']:,}",
        xy=(peak_row["Date"], peak_row["Questions"]),
        xytext=(peak_row["Date"] - pd.DateOffset(months=15),
                peak_row["Questions"] + 8000),
        fontsize=8, ha="center",
        arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
    )

    # ── Latest full-month annotation ──────────────────────────────────────────
    last_row = df.iloc[-1]
    ax.annotate(
        f"Latest full month\n{last_row['Month']}\n{last_row['Questions']:,}",
        xy=(last_row["Date"], last_row["Questions"]),
        xytext=(last_row["Date"] - pd.DateOffset(months=10),
                last_row["Questions"] + 40000),
        fontsize=8, ha="center",
        arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
    )

    # ── AI Milestone markers ───────────────────────────────────────────────────
    y_max = df["Questions"].max()
    label_y_offsets = [0.93, 0.82, 0.70, 0.58, 0.46]  # stagger label heights

    for i, m in enumerate(AI_MILESTONES):
        mdate = pd.to_datetime(m["date"])
        color = m["color"]

        # Shaded region ±1 month around milestone
        region_start = mdate - pd.DateOffset(weeks=2)
        region_end   = mdate + pd.DateOffset(weeks=2)
        ax.axvspan(region_start, region_end, alpha=0.15, color=color, linewidth=0)

        # Vertical dashed line
        ax.axvline(mdate, color=color, linestyle="--", linewidth=1.2, alpha=0.8)

        # Label with offset to reduce overlaps
        label_y = y_max * label_y_offsets[i % len(label_y_offsets)]
        ax.text(
            mdate + pd.DateOffset(weeks=3), label_y,
            m["label"],
            fontsize=7.5, color=color, fontweight="bold",
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor=color, linewidth=0.8, alpha=0.85),
        )

    # ── Axes & grid ───────────────────────────────────────────────────────────
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x)))
    )
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.35, linewidth=0.5)
    ax.set_xlim(df["Date"].min(), df["Date"].max() + pd.DateOffset(months=2))
    ax.set_xlabel("Month", fontsize=11)
    ax.set_ylabel("Questions", fontsize=11)

    # ── Legend ────────────────────────────────────────────────────────────────
    series_handles = ax.get_legend_handles_labels()[0][:2]
    series_labels  = ["Monthly questions", "12-month rolling average"]

    milestone_handles = [
        Line2D([0], [0], color=m["color"], linestyle="--", linewidth=1.4,
               label=m["short"])
        for m in AI_MILESTONES
    ]

    ax.legend(
        series_handles + milestone_handles,
        series_labels + [m["short"] for m in AI_MILESTONES],
        loc="upper right", fontsize=8.5, framealpha=0.9,
        edgecolor="#cccccc", ncol=2,
    )

    ax.set_title("Stack Overflow questions per month\n(with AI milestone markers)",
                 fontsize=14, pad=10)

    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    print(f"Chart saved → {save_path}")
    plt.show()
    return fig


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Stack Overflow Questions Chart — AI Milestone Edition")
    print("=" * 60)

    # ── Choose your data source ───────────────────────────────────────────────
    #
    # Option A — SEDE (most accurate, requires a saved query ID):
    #   df = fetch_from_sede(query_id=XXXXXXX)
    #
    # Option B — CSV from SEDE (download from the website):
    #   df = load_from_csv("so_questions_per_month.csv")
    #
    # Option C — Demo/synthetic data (no account needed, approximate shape):
    #   df = load_demo_data()
    #
    # Uncomment the one you want:

    if os.path.exists("so_questions_per_month.csv"):
        df = load_from_csv("so_questions_per_month.csv")
    else:
        df = load_demo_data()

    # Rename columns if needed (SEDE exports "Month" and "Questions")
    df.columns = [c.strip() for c in df.columns]
    if "Month" not in df.columns:
        df.rename(columns={df.columns[0]: "Month", df.columns[1]: "Questions"}, inplace=True)

    df["Questions"] = pd.to_numeric(df["Questions"], errors="coerce")
    df = df.dropna(subset=["Questions"])

    print(f"Data loaded: {len(df)} months, "
          f"{df['Month'].min()} → {df['Month'].max()}")
    print(f"Peak: {df['Questions'].max():,.0f} questions in "
          f"{df.loc[df['Questions'].idxmax(), 'Month']}")

    make_chart(df)


if __name__ == "__main__":
    main()