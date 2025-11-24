# Next Up: Predicting NBA G-League Call-Ups

Next Up is a data science project that explores the most influential factors behind NBA call-ups from the G-League. Beyond performance metrics, it incorporates contextual variables such as agency representation, team needs, player history, and injury-related opportunities.

Students will build a predictive model using multi-season data and validate it by comparing the model’s predictions with actual call-up events from previous seasons.
The final deliverable is an interactive Streamlit web application that allows users to simulate call-up probabilities, visualize model insights, and explore historical outcomes.

# 📊 Deliverables

- Cleaned Dataset: Unified G-League player-season data with call-up labels and engineered features

- Predictive Model: Trained classification model (e.g., Random Forest, XGBoost, Logistic Regression)

- Streamlit Web Application featuring:

    - Player-level prediction simulator

    - Team filters and comparison tools

    - Feature importance visualizations (e.g., SHAP plots)

    - Historical validation and accuracy metrics

    - Interactive discussion and results summary section

- GitHub Repository: Documented codebase, organized data pipeline, and final cleaned scripts

# Dataset / Data Sources

- G-League stats & rosters: SportsRadar API

- Agent affiliations: HoopsHype, RealGM, LinkedIn scraping

- NBA call-up logs: NBA transactions, news aggregators

- Optional context data: Team injury reports, roster shortages (Spotrac, media reports)

# Learning Goals

- Clean, merge, and engineer features from heterogeneous data sources

- Build and evaluate classification models on real-world, temporally imbalanced data

- Interpret model behavior using feature importance (SHAP, permutation importance)

- Validate predictions against real-world events and assess model generalization

- Communicate results through an interactive, user-friendly web application

- Collaborate using GitHub for version control and project transparency

# Domain Background

The path from the NBA G-League to the NBA is highly competitive and often unpredictable.
This project uses data science to analyze how performance, context, and representation impact advancement.
The findings could provide insight for players, agents, analysts, and teams on which attributes correlate most strongly with call-up opportunities
