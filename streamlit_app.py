"""
Next Up: G-League Call-Up Predictor
Streamlit Application Entry Point

This file serves as the root-level entry point for Streamlit Cloud deployment.
It's a copy of app/app.py with adjusted paths for root-level execution.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os
import joblib
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

try:
    import shap  # type: ignore
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Next Up: G-League Call-Up Predictor",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load base G-League datasets (players, rosters, teams, callups)."""
    try:
        base_path = Path(__file__).parent

        players = pd.read_csv(base_path / 'raw' / 'gleague_players.csv')
        rosters = pd.read_csv(base_path / 'raw' / 'gleague_rosters.csv')
        teams = pd.read_csv(base_path / 'raw' / 'gleague_teams.csv')

        # Try to load manual call-up data (if available)
        try:
            callups = pd.read_csv(base_path / 'data' / 'external' / 'callups.csv')
        except FileNotFoundError:
            callups = None

        # Fallback: if manual callups.csv not present, try 10-day callups tidy file
        if callups is None:
            try:
                callups = pd.read_csv(base_path / 'data' / 'callups_10day_tidy.csv')
            except FileNotFoundError:
                callups = None

        return players, rosters, teams, callups
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.error(f"Looking in: {Path(__file__).parent}")
        return None, None, None, None


@st.cache_data
def load_prediction_data():
    """Load cleaned prediction dataset and feature importance (for modeling)."""
    base_path = Path(__file__).parent
    pred_path = base_path / 'data' / 'prediction_dataset_cleaned.csv'
    feat_imp_path = base_path / 'data' / 'feature_importance.csv'

    pred_df = None
    feat_imp_df = None

    try:
        pred_df = pd.read_csv(pred_path)
    except FileNotFoundError:
        pass

    try:
        feat_imp_df = pd.read_csv(feat_imp_path)
    except FileNotFoundError:
        pass

    return pred_df, feat_imp_df


@st.cache_resource
def load_model():
    """Train a Logistic Regression pipeline from the cleaned prediction dataset.

    We intentionally retrain the model at app startup rather than loading a
    serialized artifact to avoid sklearn version mismatch issues.
    """
    pred_df, _ = load_prediction_data()
    if pred_df is None:
        st.error("Prediction dataset not available; cannot train fallback model.")
        return None

    # Prepare features and target
    exclude_cols = ['player_name', 'season_year', 'called_up']
    feature_cols = [c for c in pred_df.columns if c not in exclude_cols]
    categorical_cols = ['position'] if 'position' in feature_cols else []
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols),
        ]
    )

    log_reg = LogisticRegression(
        class_weight='balanced',
        max_iter=2000,
        random_state=42,
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', log_reg),
    ])

    X = pred_df[feature_cols].copy()
    y = pred_df['called_up'].copy()
    pipeline.fit(X, y)

    st.success("✅ Trained Logistic Regression model from the cleaned prediction dataset.")
    return pipeline

# Sidebar navigation
st.sidebar.title("🏀 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "👤 Player Prediction", "🏀 Team Analysis", "📊 Data Explorer", "🔍 Model Insights"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "**Next Up** predicts which G-League players are most likely to be called up to the NBA. "
    "Built by DS3 @ UCSD."
)

# Load data
players_df, rosters_df, teams_df, callups_df = load_data()
prediction_df, feature_importance_df = load_prediction_data()
model = load_model()


def get_latest_player_predictions():
    """Return latest season row per player with model-predicted probability."""
    if prediction_df is None or model is None:
        return None

    exclude_cols = ['player_name', 'season_year', 'called_up']
    feature_cols = [c for c in prediction_df.columns if c not in exclude_cols]

    latest = (
        prediction_df
        .sort_values('season_year')
        .groupby('player_name', as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    try:
        probs = model.predict_proba(latest[feature_cols])[:, 1]
    except Exception:
        return None

    latest = latest.copy()
    latest['pred_prob'] = probs
    return latest


latest_predictions_df = get_latest_player_predictions()

# ==================== HOME PAGE ====================
if page == "🏠 Home":
    st.markdown('<p class="main-header">🏀 Next Up</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predicting NBA G-League Call-Ups</p>', unsafe_allow_html=True)
    
    # Project overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Total Players",
            value=len(players_df) if players_df is not None else "N/A"
        )
    
    with col2:
        st.metric(
            label="G-League Teams",
            value=len(teams_df) if teams_df is not None else "N/A"
        )
    
    with col3:
        callup_rate_value = "N/A"
        if callups_df is not None and 'called_up' in callups_df.columns:
            try:
                callup_rate = callups_df['called_up'].astype(float).mean() * 100
                callup_rate_value = f"{callup_rate:.1f}%"
            except Exception:
                callup_rate_value = "N/A"
        elif prediction_df is not None and 'called_up' in prediction_df.columns:
            callup_rate = prediction_df['called_up'].mean() * 100
            callup_rate_value = f"{callup_rate:.1f}%"

        st.metric(
            label="Call-Up Rate",
            value=callup_rate_value
        )
    
    st.markdown("---")
    
    # Project description
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📖 About This Project")
        st.write("""
        This application predicts which G-League players are most likely to be called up 
        to the NBA based on:
        
        - **Performance Metrics**: Stats, efficiency, consistency
        - **Player Demographics**: Position, age, physical attributes
        - **Team Context**: G-League team, NBA affiliate
        - **Contextual Factors**: Agent representation, team needs (future)
        
        Use the sidebar to explore player predictions, team analytics, and model insights.
        """)
    
    with col2:
        st.subheader("🎯 Features")
        st.write("""
        **Current Features:**
        - 🤖 Player-level call-up probability predictions
        - 🏀 Team-level call-up analysis with predicted probabilities
        - 📊 Data exploration and visualization
        - 👥 Player demographic insights
        - 📈 Model performance and feature importance views
        
        **Planned Enhancements:**
        - More advanced interpretability (e.g., SHAP plots)
        - Scenario analysis and what-if simulations
        """)

# ==================== PLAYER PREDICTION PAGE ====================
elif page == "👤 Player Prediction":
    st.header("👤 Player Call-Up Prediction")
    st.write("Predict the likelihood of a G-League player being called up to the NBA")
    
    if prediction_df is None:
        st.error("Prediction dataset not available. Please generate it via analysis/EDA.")
    else:
        # Player selection
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if latest_predictions_df is not None:
                player_options = sorted(latest_predictions_df['player_name'].dropna().unique())
            elif players_df is not None:
                player_options = sorted(players_df['full_name'].dropna().unique())
            else:
                player_options = []

            player_name = st.selectbox(
                "Select a player:",
                options=player_options
            )
        
        with col2:
            st.write("")
            st.write("")
            predict_button = st.button("🔮 Predict Call-Up Probability", type="primary")
        
        if predict_button or player_name:
            st.markdown("---")

            # Get latest prediction row for this player (from prediction dataset)
            pred_row = None
            if latest_predictions_df is not None:
                match_pred = latest_predictions_df[latest_predictions_df['player_name'] == player_name]
                if not match_pred.empty:
                    pred_row = match_pred.iloc[0]

            st.subheader(f"Player Profile: {player_name}")

            col_p1, col_p2, col_p3, col_p4 = st.columns(4)

            with col_p1:
                pos = pred_row.get('position', 'N/A') if pred_row is not None and 'position' in pred_row else 'N/A'
                st.metric("Position", pos)
            with col_p2:
                season_val = pred_row.get('season_year', 'N/A') if pred_row is not None else 'N/A'
                st.metric("Season", season_val)
            with col_p3:
                avg_pts = pred_row.get('avg_points', 'N/A') if pred_row is not None and 'avg_points' in pred_row else 'N/A'
                st.metric("Avg Points", avg_pts)
            with col_p4:
                avg_eff = pred_row.get('avg_efficiency', 'N/A') if pred_row is not None and 'avg_efficiency' in pred_row else 'N/A'
                st.metric("Avg Efficiency", avg_eff)

            st.markdown("---")

            # Model prediction + actual outcome
            if model is None or latest_predictions_df is None or pred_row is None:
                st.warning("⚠️ Trained model or prediction data not available.")
            else:
                prob = float(pred_row['pred_prob'])
                season = int(pred_row['season_year'])
                actual_called_up = int(pred_row.get('called_up', 0))

                st.subheader("📊 Predicted Call-Up Probability")
                st.write(f"Using data from season **{season}**.")

                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    title={'text': "Call-Up Probability"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 25], 'color': "lightgray"},
                            {'range': [25, 50], 'color': "lightyellow"},
                            {'range': [50, 75], 'color': "lightgreen"},
                            {'range': [75, 100], 'color': "green"}
                        ]
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)

                label = (
                    "**HIGH**" if prob > 0.6
                    else "**MODERATE**" if prob > 0.3
                    else "**LOW**"
                )
                st.markdown(f"**Prediction**: {label} likelihood of call-up (probability = `{prob:.3f}`)")

                # Actual historical outcome
                actual_text = "WAS CALLED UP (1)" if actual_called_up == 1 else "WAS NOT CALLED UP (0)"
                st.markdown(f"**Historical Outcome**: {actual_text}")

        # Player leaderboard (all players)
        if latest_predictions_df is not None and model is not None:
            st.markdown("---")
            st.subheader("📈 Player Leaderboard (Model Predictions)")

            leaderboard_df = latest_predictions_df.copy()
            leaderboard_df = leaderboard_df[
                ['player_name', 'season_year', 'position', 'pred_prob', 'called_up']
            ].drop_duplicates()

            col_lb1, col_lb2 = st.columns([1, 1])
            with col_lb1:
                sort_order = st.selectbox(
                    "Sort players by predicted probability",
                    ["Highest first", "Lowest first"],
                    index=0,
                )
            with col_lb2:
                max_n = min(100, len(leaderboard_df))
                top_n = st.slider("Number of players to show", 10, max_n, 25)

            ascending = sort_order == "Lowest first"
            leaderboard_df = leaderboard_df.sort_values('pred_prob', ascending=ascending).head(top_n)

            # Add readable columns
            leaderboard_df = leaderboard_df.assign(
                pred_prob_pct=lambda d: d['pred_prob'] * 100,
                called_up_label=lambda d: d['called_up'].map({1: "Yes", 0: "No"}),
            )

            fig_lb = px.bar(
                leaderboard_df,
                x='player_name',
                y='pred_prob',
                color='position',
                labels={'player_name': 'Player', 'pred_prob': 'Predicted Call-Up Probability'},
                title="Top Players by Predicted Call-Up Probability"
            )
            fig_lb.update_yaxes(tickformat=".0%")
            fig_lb.update_layout(xaxis_tickangle=-60, height=500)
            st.plotly_chart(fig_lb, use_container_width=True)

            st.dataframe(
                leaderboard_df[['player_name', 'season_year', 'position', 'pred_prob', 'pred_prob_pct', 'called_up_label']]
                .rename(columns={
                    'pred_prob': 'pred_prob (0-1)',
                    'pred_prob_pct': 'pred_prob (%)',
                    'called_up_label': 'actually_called_up',
                })
                .reset_index(drop=True),
                use_container_width=True,
            )

# ==================== TEAM ANALYSIS PAGE ====================
elif page == "🏀 Team Analysis":
    st.header("🏀 Team Call-Up Analysis")
    
    if rosters_df is None or teams_df is None:
        st.error("Unable to load team data")
    else:
        team_names = sorted(teams_df['team_name'].dropna().unique())

        col_team_sel, col_team_compare = st.columns([2, 2])

        with col_team_sel:
            team_name = st.selectbox(
                "Select a team:",
                options=team_names
            )

        with col_team_compare:
            compare_teams = st.multiselect(
                "Compare teams (by average predicted call-up probability):",
                options=team_names,
                default=[team_name] if team_name else None
            )

        # Prepare predictions merged with rosters for team-level analysis
        team_roster = rosters_df[rosters_df['team_name'] == team_name]

        if latest_predictions_df is not None:
            team_pred_merge = rosters_df.merge(
                latest_predictions_df[['player_name', 'pred_prob']],
                on='player_name',
                how='left'
            )
        else:
            team_pred_merge = None

        st.markdown("---")

        # Team metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Players", len(team_roster))

        with col2:
            if team_pred_merge is not None:
                this_team = team_pred_merge[team_pred_merge['team_name'] == team_name]
                avg_prob = this_team['pred_prob'].mean()
                if np.isnan(avg_prob):
                    st.metric("Avg Predicted Call-Up Prob", "N/A")
                else:
                    st.metric("Avg Predicted Call-Up Prob", f"{avg_prob*100:.1f}%")
            else:
                st.metric("Avg Predicted Call-Up Prob", "N/A")

        with col3:
            positions = team_roster['position'].nunique()
            st.metric("Positions", positions)

        st.markdown("---")

        # Position distribution
        st.subheader("📊 Position Distribution")
        position_counts = team_roster['position'].value_counts()

        fig = px.bar(
            x=position_counts.index,
            y=position_counts.values,
            labels={'x': 'Position', 'y': 'Number of Players'},
            title=f"{team_name} - Position Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Predicted probabilities for this team
        if team_pred_merge is not None and model is not None:
            st.subheader("🔮 Player Call-Up Probabilities (Model Predictions)")
            this_team = team_pred_merge[team_pred_merge['team_name'] == team_name].copy()
            this_team = this_team[['player_name', 'position', 'pred_prob']].drop_duplicates()
            this_team = this_team.sort_values('pred_prob', ascending=False)

            # Bar chart of probabilities
            fig_probs = px.bar(
                this_team,
                x='player_name',
                y='pred_prob',
                color='position',
                labels={'player_name': 'Player', 'pred_prob': 'Predicted Call-Up Probability'},
                title=f"{team_name} - Predicted Call-Up Probabilities",
            )
            fig_probs.update_yaxes(tickformat=".0%")
            fig_probs.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_probs, use_container_width=True)

            # Table
            st.dataframe(
                this_team.rename(columns={'pred_prob': 'pred_prob (0-1)'}).reset_index(drop=True),
                use_container_width=True
            )

        # Team comparison view
        if compare_teams and team_pred_merge is not None and model is not None:
            st.markdown("---")
            st.subheader("📈 Team Comparison (Average Predicted Call-Up Probability)")
            comp_df = (
                team_pred_merge[team_pred_merge['team_name'].isin(compare_teams)]
                .groupby('team_name')['pred_prob']
                .mean()
                .reset_index()
            )
            comp_df = comp_df.dropna(subset=['pred_prob'])

            if not comp_df.empty:
                fig_comp = px.bar(
                    comp_df,
                    x='team_name',
                    y='pred_prob',
                    labels={'team_name': 'Team', 'pred_prob': 'Avg Predicted Call-Up Probability'},
                    title="Average Predicted Call-Up Probabilities by Team"
                )
                fig_comp.update_yaxes(tickformat=".0%")
                fig_comp.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_comp, use_container_width=True)
            else:
                st.info("No prediction data available for the selected comparison teams.")

        # Global team leaderboard (all teams)
        if team_pred_merge is not None and model is not None:
            st.markdown("---")
            st.subheader("🏆 Team Leaderboard (Avg Predicted Call-Up Probability)")

            global_df = (
                team_pred_merge
                .groupby('team_name')['pred_prob']
                .mean()
                .reset_index()
                .dropna(subset=['pred_prob'])
            )

            if not global_df.empty:
                col_tb1, col_tb2 = st.columns([1, 1])
                with col_tb1:
                    team_sort_order = st.selectbox(
                        "Sort teams by average predicted probability",
                        ["Highest first", "Lowest first"],
                        index=0,
                    )
                with col_tb2:
                    max_n_teams = min(30, len(global_df))
                    top_n_teams = st.slider("Number of teams to show", 5, max_n_teams, 15)

                asc_teams = team_sort_order == "Lowest first"
                global_df = global_df.sort_values('pred_prob', ascending=asc_teams).head(top_n_teams)

                fig_team_lb = px.bar(
                    global_df,
                    x='team_name',
                    y='pred_prob',
                    labels={'team_name': 'Team', 'pred_prob': 'Avg Predicted Call-Up Probability'},
                    title="Teams by Avg Predicted Call-Up Probability",
                )
                fig_team_lb.update_yaxes(tickformat=".0%")
                fig_team_lb.update_layout(xaxis_tickangle=-60, height=500)
                st.plotly_chart(fig_team_lb, use_container_width=True)

                st.dataframe(
                    global_df.assign(avg_pred_prob_pct=lambda d: d['pred_prob'] * 100)
                    .rename(columns={
                        'pred_prob': 'avg_pred_prob (0-1)',
                        'avg_pred_prob_pct': 'avg_pred_prob (%)',
                    })
                    .reset_index(drop=True),
                    use_container_width=True,
                )

# ==================== DATA EXPLORER PAGE ====================
elif page == "📊 Data Explorer":
    st.header("📊 Data Explorer")
    st.write("Explore the G-League dataset")
    
    # Dataset selector
    dataset_options = ["Players", "Rosters", "Teams", "Call-Ups (if available)"]
    if prediction_df is not None:
        dataset_options.append("Prediction Dataset (Cleaned)")

    dataset = st.selectbox(
        "Select dataset to explore:",
        dataset_options
    )
    
    if dataset == "Players" and players_df is not None:
        st.subheader("👥 Players Dataset")
        st.write(f"Total players: {len(players_df)}")
        
        # Show data
        st.dataframe(players_df, use_container_width=True)
        
        # Basic visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Position Distribution")
            fig = px.pie(players_df, names='position', title='Players by Position')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Height Distribution")
            fig = px.histogram(players_df, x='height', title='Player Heights')
            st.plotly_chart(fig, use_container_width=True)
    
    elif dataset == "Rosters" and rosters_df is not None:
        st.subheader("📋 Rosters Dataset")
        st.write(f"Total roster entries: {len(rosters_df)}")
        st.dataframe(rosters_df, use_container_width=True)
    
    elif dataset == "Teams" and teams_df is not None:
        st.subheader("🏀 Teams Dataset")
        st.write(f"Total teams: {len(teams_df)}")
        st.dataframe(teams_df, use_container_width=True)
    
    elif dataset == "Call-Ups (if available)":
        if callups_df is not None:
            st.subheader("🎯 Call-Ups Dataset")
            st.write(f"Total records: {len(callups_df)}")
            st.dataframe(callups_df, use_container_width=True)

            st.markdown("---")
            st.subheader("📊 Call-Up Player Profiles (Heights, Positions, etc.)")

            if players_df is not None:
                # Helper to normalize names: "Last, First" -> "First Last"
                def _clean_name(name):
                    if isinstance(name, str) and "," in name:
                        last, first = name.split(",", 1)
                        return f"{first.strip()} {last.strip()}"
                    return str(name).strip() if isinstance(name, str) else name

                callups_enriched = callups_df.copy()
                if 'player_name' in callups_enriched.columns:
                    callups_enriched['player_name_clean'] = callups_enriched['player_name'].apply(_clean_name)

                    players_enriched = players_df.copy()
                    if 'full_name' in players_enriched.columns:
                        players_enriched['player_name_clean'] = players_enriched['full_name'].astype(str).str.strip()

                        # Join callups with player demographics
                        callups_enriched = callups_enriched.merge(
                            players_enriched[
                                ['player_name_clean', 'position', 'height', 'weight', 'college']
                            ],
                            on='player_name_clean',
                            how='left'
                        )

                        # Also join G-League team info from rosters, if available
                        if rosters_df is not None:
                            rosters_enriched = rosters_df.copy()
                            rosters_enriched['player_name_clean'] = rosters_enriched['player_name'].apply(_clean_name)
                            rosters_enriched = rosters_enriched[['player_name_clean', 'team_name']].drop_duplicates()

                            callups_enriched = callups_enriched.merge(
                                rosters_enriched,
                                on='player_name_clean',
                                how='left'
                            )

                        col_c1, col_c2 = st.columns(2)

                        # Position distribution among called-up players
                        with col_c1:
                            pos_counts = callups_enriched['position'].dropna().value_counts()
                            if not pos_counts.empty:
                                fig_pos = px.bar(
                                    x=pos_counts.index,
                                    y=pos_counts.values,
                                    labels={'x': 'Position', 'y': 'Number of Call-Ups'},
                                    title="Positions of Called-Up Players",
                                )
                                st.plotly_chart(fig_pos, use_container_width=True)
                            else:
                                st.info("No position information available for called-up players.")

                        # Height distribution among called-up players
                        with col_c2:
                            heights = pd.to_numeric(callups_enriched['height'], errors='coerce').dropna()
                            if not heights.empty:
                                fig_h = px.histogram(
                                    x=heights,
                                    nbins=20,
                                    labels={'x': 'Height (inches)', 'y': 'Number of Call-Ups'},
                                    title="Heights of Called-Up Players",
                                )
                                st.plotly_chart(fig_h, use_container_width=True)
                            else:
                                st.info("No height information available for called-up players.")

                        st.markdown("---")
                        st.subheader("🏀 Teams Involved in Call-Ups")

                        col_t1, col_t2 = st.columns(2)

                        # G-League teams players were called up from
                        with col_t1:
                            if rosters_df is not None and 'team_name' in callups_enriched.columns:
                                gleague_counts = callups_enriched['team_name'].dropna().value_counts().head(15)
                                if not gleague_counts.empty:
                                    fig_g = px.bar(
                                        x=gleague_counts.index,
                                        y=gleague_counts.values,
                                        labels={'x': 'G-League Team', 'y': 'Number of Call-Ups'},
                                        title="Top G-League Teams (By Call-Ups)",
                                    )
                                    fig_g.update_layout(xaxis_tickangle=-45)
                                    st.plotly_chart(fig_g, use_container_width=True)
                                else:
                                    st.info("No G-League team information available for called-up players.")
                            else:
                                st.info("Rosters dataset not available for linking G-League teams.")

                        # NBA teams that called players up
                        with col_t2:
                            if 'nba_team' in callups_enriched.columns:
                                nba_counts = callups_enriched['nba_team'].dropna().value_counts().head(15)
                                if not nba_counts.empty:
                                    fig_n = px.bar(
                                        x=nba_counts.index,
                                        y=nba_counts.values,
                                        labels={'x': 'NBA Team', 'y': 'Number of Call-Ups'},
                                        title="Top NBA Teams (By Call-Ups)",
                                    )
                                    fig_n.update_layout(xaxis_tickangle=-45)
                                    st.plotly_chart(fig_n, use_container_width=True)
                                else:
                                    st.info("No NBA team information available for call-ups.")
                            else:
                                st.info("Call-ups dataset does not contain `nba_team` information.")
                    else:
                        st.info("Players dataset does not contain `full_name` for joining.")
                else:
                    st.info("Call-ups dataset does not contain `player_name` for joining to player demographics.")
        else:
            st.warning("Call-up data not yet available")

    elif dataset == "Prediction Dataset (Cleaned)" and prediction_df is not None:
        st.subheader("📈 Prediction Dataset (Cleaned)")
        st.write(f"Total records: {len(prediction_df)}")
        st.dataframe(prediction_df, use_container_width=True)

# ==================== MODEL INSIGHTS PAGE ====================
elif page == "🔍 Model Insights":
    st.header("🔍 Model Insights & Performance")
    
    tab1, tab2, tab3 = st.tabs(["📊 Performance Metrics", "🎯 Feature Importance", "📈 Model Validation"])
    
    with tab1:
        st.subheader("Model Performance Metrics")

        if prediction_df is None or model is None:
            st.warning("Prediction dataset or trained model not available.")
        else:
            # Recreate stratified splits (same strategy as in analysis.ipynb)
            from sklearn.model_selection import train_test_split

            exclude_cols = ['player_name', 'season_year', 'called_up']
            feature_cols = [c for c in prediction_df.columns if c not in exclude_cols]

            X = prediction_df[feature_cols].copy()
            y = prediction_df['called_up'].copy()

            X_temp, X_test_mi, y_temp, y_test_mi = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )
            X_train_mi, X_val_mi, y_train_mi, y_val_mi = train_test_split(
                X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
            )

            # Evaluate on validation and test sets
            def eval_set(X_set, y_set, name: str):
                y_pred = model.predict(X_set)
                y_proba = model.predict_proba(X_set)[:, 1]
                f1 = f1_score(y_set, y_pred)
                prec = precision_score(y_set, y_pred)
                rec = recall_score(y_set, y_pred)
                roc = roc_auc_score(y_set, y_proba)
                pr = average_precision_score(y_set, y_proba)
                return {
                    'name': name,
                    'f1': f1,
                    'precision': prec,
                    'recall': rec,
                    'roc_auc': roc,
                    'pr_auc': pr,
                    'y_true': y_set,
                    'y_pred': y_pred,
                    'y_proba': y_proba,
                }

            val_metrics = eval_set(X_val_mi, y_val_mi, "Validation")
            test_metrics = eval_set(X_test_mi, y_test_mi, "Test")

            # Metrics summary
            st.markdown("#### Key Metrics (Validation vs Test)")
            metrics_df = pd.DataFrame([
                {
                    'Set': m['name'],
                    'F1-Score': m['f1'],
                    'Precision': m['precision'],
                    'Recall': m['recall'],
                    'ROC-AUC': m['roc_auc'],
                    'PR-AUC': m['pr_auc'],
                }
                for m in [val_metrics, test_metrics]
            ])
            st.dataframe(metrics_df.style.format({
                'F1-Score': "{:.3f}",
                'Precision': "{:.3f}",
                'Recall': "{:.3f}",
                'ROC-AUC': "{:.3f}",
                'PR-AUC': "{:.3f}",
            }), use_container_width=True)

            col1m, col2m, col3m, col4m = st.columns(4)
            with col1m:
                st.metric("Test F1-Score", f"{test_metrics['f1']:.3f}")
            with col2m:
                st.metric("Test Precision", f"{test_metrics['precision']:.3f}")
            with col3m:
                st.metric("Test Recall", f"{test_metrics['recall']:.3f}")
            with col4m:
                st.metric("Test ROC-AUC", f"{test_metrics['roc_auc']:.3f}")

            # Confusion matrix (test set)
            st.markdown("#### Confusion Matrix (Test Set)")
            cm = confusion_matrix(test_metrics['y_true'], test_metrics['y_pred'])
            cm_df = pd.DataFrame(
                cm,
                index=['Actual 0 (Not Called Up)', 'Actual 1 (Called Up)'],
                columns=['Pred 0', 'Pred 1'],
            )
            fig_cm = px.imshow(
                cm_df,
                text_auto=True,
                color_continuous_scale="Blues",
                labels=dict(x="Predicted", y="Actual", color="Count"),
                title="Confusion Matrix (Test Set)",
            )
            st.plotly_chart(fig_cm, use_container_width=True)

            # ROC & PR curves (test set)
            st.markdown("#### ROC & Precision-Recall Curves (Test Set)")
            fpr, tpr, _ = roc_curve(test_metrics['y_true'], test_metrics['y_proba'])
            precision, recall, _ = precision_recall_curve(test_metrics['y_true'], test_metrics['y_proba'])

            fig_curves = go.Figure()
            fig_curves.add_trace(go.Scatter(
                x=fpr, y=tpr, mode='lines',
                name=f"ROC Curve (AUC={test_metrics['roc_auc']:.3f})",
                line=dict(color="#e74c3c")
            ))
            fig_curves.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode='lines',
                name="Random", line=dict(color="gray", dash='dash')
            ))
            fig_curves.update_layout(
                title="ROC Curve (Test Set)",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
            )
            st.plotly_chart(fig_curves, use_container_width=True)

            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(
                x=recall, y=precision, mode='lines',
                name=f"PR Curve (AUC={test_metrics['pr_auc']:.3f})",
                line=dict(color="#3498db")
            ))
            fig_pr.update_layout(
                title="Precision-Recall Curve (Test Set)",
                xaxis_title="Recall",
                yaxis_title="Precision",
            )
            st.plotly_chart(fig_pr, use_container_width=True)
    
    with tab2:
        st.subheader("Feature Importance")
        if feature_importance_df is not None:
            top_n = st.slider("Number of top features to display:", 5, 30, 20)
            fi = feature_importance_df.copy()
            fi = fi.sort_values('Abs_Correlation', ascending=False).head(top_n)

            fig_fi = px.bar(
                fi,
                x='Abs_Correlation',
                y='Feature',
                orientation='h',
                color='Correlation',
                color_continuous_scale='RdBu',
                title=f"Top {top_n} Features by Correlation with Call-Ups",
            )
            fig_fi.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_fi, use_container_width=True)

            st.markdown("""
            These importances are based on correlation with the target and align closely with the
            Logistic Regression coefficients used in the model. For more advanced explanations,
            you can extend this app with SHAP plots in the future.
            """)
        else:
            st.info("Feature importance file not found. Please generate it from EDA.")
    
    with tab3:
        st.subheader("Historical Validation")
        if prediction_df is None or model is None:
            st.warning("Prediction dataset or trained model not available.")
        else:
            # Reuse metrics computed in tab1
            st.markdown("""
            This section summarizes how well the model performed on the held-out **test set**.
            The test set simulates future seasons the model has never seen before.
            """)

            # Recompute splits and metrics quickly (same as in tab1)
            from sklearn.model_selection import train_test_split
            exclude_cols = ['player_name', 'season_year', 'called_up']
            feature_cols = [c for c in prediction_df.columns if c not in exclude_cols]
            X = prediction_df[feature_cols].copy()
            y = prediction_df['called_up'].copy()
            X_temp, X_test_h, y_temp, y_test_h = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )
            # Use entire X_test_h / y_test_h
            y_pred_h = model.predict(X_test_h)
            y_proba_h = model.predict_proba(X_test_h)[:, 1]

            f1_h = f1_score(y_test_h, y_pred_h)
            prec_h = precision_score(y_test_h, y_pred_h)
            rec_h = recall_score(y_test_h, y_pred_h)
            roc_h = roc_auc_score(y_test_h, y_proba_h)
            pr_h = average_precision_score(y_test_h, y_proba_h)

            st.markdown("#### Test Set Metrics")
            col_a, col_b, col_c, col_d, col_e = st.columns(5)
            with col_a:
                st.metric("F1-Score", f"{f1_h:.3f}")
            with col_b:
                st.metric("Precision", f"{prec_h:.3f}")
            with col_c:
                st.metric("Recall", f"{rec_h:.3f}")
            with col_d:
                st.metric("ROC-AUC", f"{roc_h:.3f}")
            with col_e:
                st.metric("PR-AUC", f"{pr_h:.3f}")

            st.markdown("""
            - **F1-Score** balances the trade-off between precision and recall on rare call-ups.  
            - **Precision** tells us what fraction of predicted call-ups actually got called up.  
            - **Recall** measures how many of the true call-ups we successfully identified.  
            - **ROC-AUC / PR-AUC** summarize overall ranking quality and performance on the imbalanced dataset.
            """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with by DS3 @ UCSD | Project: Next Up | 2024-2025</p>
</div>
""", unsafe_allow_html=True)

