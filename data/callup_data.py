# This file is going to get callup data from HoopRumors for 10-day contracts since Jan 2007 till date and save it to a CSV file
import os, re, time, requests
import pandas as pd
from bs4 import BeautifulSoup
from dateutil import parser
from unidecode import unidecode

URL = "https://www.hoopsrumors.com/hoops-apps/10_day_contract_tracker.php?name=&team=&type=&d1=2007-01-01&d2=2025-12-31"
OUT_DIR = "data"
RAW_CSV = os.path.join(OUT_DIR, "callups_10day_raw.csv")
TIDY_CSV = os.path.join(OUT_DIR, "callups_10day_tidy.csv")

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- Helpers ----------------
def to_date(x):
    if pd.isna(x): return pd.NaT
    try:
        # HoopsRumors uses mm/dd/yy
        return parser.parse(str(x), dayfirst=False, yearfirst=False)
    except Exception:
        return pd.NaT

def nba_season(dt):
    """
    Map a calendar date to NBA season label like '2024-25'.
    Season 'YYYY-YY' starts Aug 1 of YYYY and ends Jul 31 of YYYY+1 (safe for our use).
    """
    if pd.isna(dt): return None
    y = dt.year
    if dt.month >= 8:  # Aug..Dec -> season starts this year
        return f"{y}-{(y+1)%100:02d}"
    else:              # Jan..Jul -> season started previous year
        return f"{y-1}-{y%100:02d}"

def clean_name(name):
    # Normalize accents and spacing; keep original separately if you want
    if pd.isna(name): return None
    return unidecode(str(name)).strip()

# Optional: map team display names to a canonical form (extend as needed)
TEAM_CANON = {
    "LA Clippers": "Los Angeles Clippers",
    "L.A. Clippers": "Los Angeles Clippers",
    "LA Lakers": "Los Angeles Lakers",
    "L.A. Lakers": "Los Angeles Lakers",
    "Phoenix Suns": "Phoenix Suns",
    # add any oddities you spot here
}

def canon_team(t):
    if pd.isna(t): return None
    t = t.strip()
    return TEAM_CANON.get(t, t)

# -------------- Scrape -------------------
headers = {
    "User-Agent": "Mozilla/5.0 (DS3-NextUp class project; contact: example@ucsd.edu)"
}

def scrape_with_pandas(url):
    # Most reliable: server-rendered table; read_html grabs it straight away
    tables = pd.read_html(url)
    # Find the table that has the expected columns; usually the first
    for df in tables:
        cols = [c.lower() for c in df.columns.astype(str)]
        if {"date", "player", "team", "type"}.issubset(set(cols)):
            return df
    return None

def scrape_with_bs4(url):
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    table = soup.find("table")
    rows = []
    if table:
        th = [th.get_text(strip=True) for th in table.find_all("th")]
        for tr in table.find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) >= 4:
                rows.append([td.get_text(" ", strip=True) for td in tds[:4]])
        if rows:
            df = pd.DataFrame(rows, columns=th[:4])  # Date, Player, Active, Team, Type
            return df
    return None

# Try pandas first, fallback to bs4
df_raw = scrape_with_pandas(URL)
if df_raw is None:
    df_raw = scrape_with_bs4(URL)

if df_raw is None or df_raw.empty:
    raise RuntimeError("Failed to parse HoopsRumors table. Check the URL or site layout.")

# -------------- Clean --------------------
# Standardize column names
df_raw.columns = [c.strip().title() for c in df_raw.columns]
# Keep only relevant cols if extra present
keep = [c for c in ["Date","Player","Active","Team","Type"] if c in df_raw.columns]
df_raw = df_raw[keep].copy()

# Parse and enrich
df_raw["date"] = df_raw["Date"].apply(to_date)
df_raw["season"] = df_raw["date"].apply(nba_season)
df_raw["player_name"] = df_raw["Player"].apply(clean_name)
df_raw["nba_team"] = df_raw["Team"].apply(canon_team)
df_raw["contract_type"] = df_raw["Type"].astype(str).str.strip()

# Sort newest→oldest
df_raw = df_raw.sort_values("date", ascending=False).reset_index(drop=True)

# Save RAW scrape as-is
df_raw.to_csv(RAW_CSV, index=False)
print(f"Saved raw 10-day table: {RAW_CSV}  ({len(df_raw):,} rows)")

# -------------- Tidy / Deduplicate --------------------
# Many players sign multiple 10-days; for labeling a 'call-up event' we usually keep the FIRST date per (player, team, season)
df_tidy = (
    df_raw
    .dropna(subset=["player_name","nba_team","season","date"])
    .sort_values("date")  # oldest first, so first call-up is kept
    .drop_duplicates(subset=["player_name","nba_team","season"], keep="first")
    .sort_values(["season","nba_team","date"])
    .reset_index(drop=True)
)

# Select neat columns
df_tidy = df_tidy[["date","season","player_name","nba_team","contract_type"]]

df_tidy.to_csv(TIDY_CSV, index=False)
print(f"Saved tidy call-up events (first 10-day per player/team/season): {TIDY_CSV}  ({len(df_tidy):,} rows)")

