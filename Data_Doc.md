flowchart LR
  %% =========================
  %% Data Sources
  %% =========================
  A1[SportsRadar API\nG-League player + game stats\nmulti-season] 
  A2[Basketball Reference / G-League site\nrosters + team context]
  A3[HoopsRumors / NBA Transactions\n10-day call-ups + 2-way conversions]
  A4[Optional: Injury / Roster Need Signals\nSpotrac / news]

  %% =========================
  %% ETL + Raw Data
  %% =========================
  B0[(Next Up ETL Pipeline\nextract -> normalize -> load)]
  B1[pull_stats.py\nfetch season stats]
  B2[pull_rosters.py\nteam/roster context]
  B3[pull_callups.py\nscrape/label call-up events]
  B4[dedupe_join.py\nID matching + cleaning]

  %% =========================
  %% Storage
  %% =========================
  C1[(players_raw.csv)]
  C2[(callups_raw.csv)]
  C3[(rosters_raw.csv)]

  %% =========================
  %% Feature Engineering
  %% =========================
  D1[feature_build.py\nengineer player-season features]
  D2[(features_clean.csv\nmodel-ready table)]

  %% =========================
  %% Modeling + Validation
  %% =========================
  E1[train_model.ipynb\nbaseline + tuning]
  E2[Classifier\nLogReg / RF / XGBoost]
  E3[Temporal Validation\ntrain Y1-Y2, test Y3]
  E4[(model.pkl)]
  E5[Interpretability\nSHAP / Permutation Importance]

  %% =========================
  %% Analytics Engine
  %% =========================
  F0((Next Up Analytics Engine))
  F1[Prediction Service\nplayer/team probability]
  F2[Historical Validation Module\npred vs actual]
  F3[Insight Generator\nkey drivers per prediction]

  %% =========================
  %% Streamlit App
  %% =========================
  G0[Streamlit Dashboard]
  G1[Player Lookup + Simulator]
  G2[Team Filters + Comparisons]
  G3[Feature Importance View]
  G4[Season Playback\npredicted vs actual]
  G5[Export Snapshot\nCSV / PNG]

  %% =========================
  %% Flow
  %% =========================
  A1 --> B0
  A2 --> B0
  A3 --> B0
  A4 --> B0

  B0 --> B1 --> C1
  B0 --> B2 --> C3
  B0 --> B3 --> C2
  C1 --> B4
  C2 --> B4
  C3 --> B4

  B4 --> D1 --> D2
  D2 --> E1 --> E2 --> E4
  E1 --> E3
  E2 --> E5

  E4 --> F0
  D2 --> F0
  F0 --> F1 --> G1
  F0 --> F2 --> G4
  F0 --> F3 --> G3

  G0 --> G1
  G0 --> G2
  G0 --> G3
  G0 --> G4
  G0 --> G5
