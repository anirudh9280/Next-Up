"""
Next Up: G-League Call-Up Predictor
Streamlit Application Entry Point

This file serves as the root-level entry point for Streamlit Cloud deployment.
It's a copy of app/app.py with adjusted paths for root-level execution.
"""

import ast
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os
import joblib
import xgboost as xgb
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

def _parse_list_cell(value):
    """Convert stringified list columns from CSV back into Python lists."""
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    if text.startswith('[') and text.endswith(']'):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
    return [text]


@st.cache_data
def load_data():
    """Load base G-League datasets (players, rosters, teams, callups)."""
    base_path = Path(__file__).parent
        
    def _safe_read_csv(path: Path):
        try:
            return pd.read_csv(path)
        except FileNotFoundError:
            return None

    try:
        players = _safe_read_csv(base_path / 'raw' / 'gleague_players.csv')
        rosters = _safe_read_csv(base_path / 'raw' / 'gleague_rosters.csv')
        teams = _safe_read_csv(base_path / 'raw' / 'gleague_teams.csv')

        callups = None
        aggregated_path = base_path / 'data' / 'callups_nba_2019_2025_aggregated.csv'
        if aggregated_path.exists():
            callups = pd.read_csv(aggregated_path)
            list_cols = ['gleague_teams', 'nba_teams', 'callup_dates', 'contract_type']
            for col in list_cols:
                if col in callups.columns:
                    callups[col] = callups[col].apply(_parse_list_cell)
        else:
            # Legacy fallback (older manual call-up sources)
            callups = _safe_read_csv(base_path / 'data' / 'external' / 'callups.csv')
            if callups is None:
                callups = _safe_read_csv(base_path / 'data' / 'callups_10day_tidy.csv')
            
        return players, rosters, teams, callups
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.error(f"Looking in: {base_path}")
        return None, None, None, None


@st.cache_data
def load_prediction_data(pred_version: float = None, feat_version: float = None):
    """Load cleaned prediction dataset and feature importance (for modeling)."""
    base_path = Path(__file__).parent
    pred_path = base_path / 'data' / 'prediction_dataset_callups_nba_cleaned.csv'
    feat_imp_path = base_path / 'data' / 'feature_importance_callups_nba.csv'

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
def load_log_reg_model(pred_version: float = None):
    """Load the pre-trained Logistic Regression pipeline (or train/save a fallback)."""
    base_path = Path(__file__).parent
    version_suffix = f"_{int(pred_version)}" if pred_version else ""
    model_path = base_path / 'models' / f'log_reg_callup_pipeline{version_suffix}.joblib'

    if model_path.exists():
        try:
            model = joblib.load(model_path)
            st.success("✅ Loaded Logistic Regression pipeline from saved artifact.")
            return model
        except Exception:
            # Artifact likely came from an older sklearn version; remove and retrain silently.
            try:
                model_path.unlink(missing_ok=True)
            except Exception:
                pass

    pred_df, _ = load_prediction_data(pred_version=pred_version)
    if pred_df is None:
        st.error("Prediction dataset not available; cannot initialize model.")
        return None

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
        C=0.01,
        max_iter=2000,
        random_state=42,
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', log_reg),
    ])

    X = pred_df[feature_cols].copy()
    y = pred_df['called_up'].copy()

    try:
        pipeline.fit(X, y)
    except Exception as exc:
        st.error(f"Failed to train fallback Logistic Regression model: {exc}")
        return None

    try:
        os.makedirs(model_path.parent, exist_ok=True)
        joblib.dump(pipeline, model_path)
    except Exception as exc:
        st.warning(f"Trained model but could not save artifact ({exc}).")

    st.success("✅ Trained and cached Logistic Regression model from the cleaned dataset.")
    return pipeline


@st.cache_resource
def load_xgb_model(pred_version: float = None):
    """Load the pre-trained XGBoost pipeline (or train/save a fallback)."""
    base_path = Path(__file__).parent
    version_suffix = f"_{int(pred_version)}" if pred_version else ""
    model_path = base_path / 'models' / f'xgb_callup_pipeline{version_suffix}.joblib'

    if model_path.exists():
        try:
            model = joblib.load(model_path)
            st.success("✅ Loaded XGBoost pipeline from saved artifact.")
            return model
        except Exception:
            try:
                model_path.unlink(missing_ok=True)
            except Exception:
                pass

    pred_df, _ = load_prediction_data(pred_version=pred_version)
    if pred_df is None:
        st.error("Prediction dataset not available; cannot initialize XGBoost model.")
        return None

    exclude_cols = ['player_name', 'season_year', 'called_up']
    feature_cols = [c for c in pred_df.columns if c not in exclude_cols]
    categorical_cols = ['position'] if 'position' in feature_cols else []
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_cols),
            ('cat', categorical_transformer, categorical_cols),
        ]
    )

    scale_pos_weight = (pred_df['called_up'] == 0).sum() / (pred_df['called_up'] == 1).sum()

    xgb_clf = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.75,
        min_child_weight=2,
        gamma=0.1,
        reg_lambda=1.5,
        reg_alpha=0.25,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        tree_method='hist'
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb_clf),
    ])

    X = pred_df[feature_cols].copy()
    y = pred_df['called_up'].copy()

    try:
        pipeline.fit(X, y)
    except Exception as exc:
        st.error(f"Failed to train fallback XGBoost model: {exc}")
        return None

    try:
        os.makedirs(model_path.parent, exist_ok=True)
        joblib.dump(pipeline, model_path)
    except Exception as exc:
        st.warning(f"Trained XGBoost model but could not save artifact ({exc}).")

    st.success("✅ Trained and cached XGBoost model from the cleaned dataset.")
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

pred_path = Path(__file__).parent / 'data' / 'prediction_dataset_callups_nba_cleaned.csv'
feat_path = Path(__file__).parent / 'data' / 'feature_importance_callups_nba.csv'
pred_version = pred_path.stat().st_mtime if pred_path.exists() else None
feat_version = feat_path.stat().st_mtime if feat_path.exists() else None
prediction_df, feature_importance_df = load_prediction_data(pred_version=pred_version, feat_version=feat_version)
log_reg_model = load_log_reg_model(pred_version=pred_version)
xgb_model = load_xgb_model(pred_version=pred_version)


def get_latest_player_predictions(log_model=None, xgb_pipeline=None):
    """Return latest season row per player with cached model predictions."""
    if prediction_df is None or (log_model is None and xgb_pipeline is None):
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

    latest = latest.copy()

    if log_model is not None:
        try:
            probs_lr = log_model.predict_proba(latest[feature_cols])[:, 1]
            latest['pred_prob_lr'] = probs_lr
            latest['pred_prob'] = probs_lr  # backward compatibility
        except Exception:
            st.warning("Unable to compute Logistic Regression probabilities for leaderboard view.")

    if xgb_pipeline is not None:
        try:
            probs_xgb = xgb_pipeline.predict_proba(latest[feature_cols])[:, 1]
            latest['pred_prob_xgb'] = probs_xgb
            latest['pred_callup_xgb'] = xgb_pipeline.predict(latest[feature_cols])
        except Exception:
            st.warning("Unable to compute XGBoost predictions for leaderboard view.")

    if 'pred_prob_lr' not in latest.columns and 'pred_callup_xgb' not in latest.columns:
        return None

    return latest


latest_predictions_df = get_latest_player_predictions(log_reg_model, xgb_model)

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
        if prediction_df is not None and 'called_up' in prediction_df.columns:
            callup_rate = prediction_df['called_up'].mean() * 100
            callup_rate_value = f"{callup_rate:.1f}%"
        elif callups_df is not None and 'times_called_up' in callups_df.columns:
            total_players = len(prediction_df) if prediction_df is not None else None
            total_callups = callups_df['times_called_up'].sum()
            if total_players:
                callup_rate_value = f"{(total_callups / total_players) * 100:.1f}%"

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
        - Player-level call-up probability predictions
        - Team-level call-up analysis with predicted probabilities
        - Data exploration and visualization
        - Player demographic insights
        - Model performance and feature importance views
        
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
        # Player selection (auto-updates when the user changes the dropdown)
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
        
        if player_name:
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
            if latest_predictions_df is None or pred_row is None:
                st.warning("⚠️ Prediction data not available for this player.")
            else:
                season_display = pred_row.get('season_year', 'N/A')
                actual_called_up = int(pred_row.get('called_up', 0))

                st.subheader("🤖 Model Predictions")
                lr_tab, xgb_tab = st.tabs(["Logistic Regression", "XGBoost"])

                with lr_tab:
                    if log_reg_model is None or 'pred_prob_lr' not in pred_row:
                        st.warning("Logistic Regression model is not available.")
                    else:
                        prob = float(pred_row['pred_prob_lr'])
                        st.write(f"Using data from season **{season_display}**.")
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

                with xgb_tab:
                    if xgb_model is None or 'pred_callup_xgb' not in pred_row:
                        st.warning("XGBoost model is not available.")
                    else:
                        pred_class = int(pred_row['pred_callup_xgb'])
                        prob_xgb = float(pred_row.get('pred_prob_xgb', pred_class))
                        class_label = "Called Up" if pred_class == 1 else "Not Called Up"
                        st.metric("Predicted Outcome", class_label)
                        st.metric("Class (0 = No, 1 = Yes)", pred_class)
                        st.markdown(f"**Model confidence**: `{prob_xgb:.3f}` probability of call-up")

                # Actual historical outcome
                actual_text = "WAS CALLED UP (1)" if actual_called_up == 1 else "WAS NOT CALLED UP (0)"
                st.markdown(f"**Historical Outcome**: {actual_text}")

        # Player leaderboard (all players)
        if latest_predictions_df is not None and log_reg_model is not None:
            st.markdown("---")
            st.subheader("📈 Player Leaderboard (Model Predictions)")

            base_cols = ['player_name', 'season_year', 'position', 'pred_prob', 'called_up']
            if 'pred_callup_xgb' in latest_predictions_df.columns:
                base_cols.insert(3, 'pred_callup_xgb')

            leaderboard_df = latest_predictions_df[base_cols].drop_duplicates()

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
            has_xgb_preds = 'pred_callup_xgb' in leaderboard_df.columns

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

            display_cols = ['player_name', 'season_year', 'position']
            if has_xgb_preds:
                display_cols.append('pred_callup_xgb')
            display_cols.extend(['pred_prob_pct', 'called_up_label'])

            rename_map = {
                'pred_prob_pct': 'pred_prob (%)',
                'called_up_label': 'actually_called_up',
            }
            if has_xgb_preds:
                rename_map['pred_callup_xgb'] = 'pred_callup (0-1)'

            st.dataframe(
                leaderboard_df[display_cols]
                .rename(columns=rename_map)
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
        
        if latest_predictions_df is not None and 'pred_prob' in latest_predictions_df.columns:
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
        if team_pred_merge is not None and log_reg_model is not None:
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
        if compare_teams and team_pred_merge is not None and log_reg_model is not None:
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
        if team_pred_merge is not None and log_reg_model is not None:
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
    dataset_options = ["Players", "Rosters", "Teams", "Call-Ups"]
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
    
    elif dataset == "Call-Ups":
        if callups_df is not None:
            st.subheader("🎯 NBA.com Call-Ups (2019-2025 Aggregated)")
            st.write(
                "Official call-up events pulled directly from NBA.com, aggregated at the player-season level."
            )
            st.write(f"Total player-season rows: {len(callups_df)}")
            st.dataframe(callups_df, use_container_width=True)
            
            st.markdown("---")
            st.subheader("📊 Season & Contract Trends")

            season_cols = {'season_year', 'season_label', 'times_called_up'}
            if season_cols.issubset(callups_df.columns):
                season_counts = (
                    callups_df
                    .groupby(['season_label', 'season_year'], as_index=False)['times_called_up']
                    .sum()
                    .rename(columns={'times_called_up': 'callups'})
                    .sort_values('season_year')
                )
                fig_season = px.bar(
                    season_counts,
                    x='season_label',
                    y='callups',
                    labels={'season_label': 'Season', 'callups': 'Number of Call-Ups'},
                    title='Call-Ups by Season',
                    category_orders={'season_label': season_counts['season_label'].tolist()},
                )
                fig_season.update_layout(xaxis_tickangle=-25)
                st.plotly_chart(fig_season, use_container_width=True)
                st.caption(
                    "Note: Certain pandemic-shortened seasons (e.g., 2020-21 bubble) are omitted "
                    "from NBA.com call-up logs, so gaps around 2020 reflect missing league play rather than zero activity."
                )
            elif 'season_year' in callups_df.columns:
                legacy_counts = (
                    callups_df.groupby('season_year')['times_called_up']
                    .sum()
                    .reset_index(name='callups')
                )
                fig_season = px.bar(
                    legacy_counts,
                    x='season_year',
                    y='callups',
                    labels={'season_year': 'Season', 'callups': 'Number of Call-Ups'},
                    title='Call-Ups by Season',
                )
                st.plotly_chart(fig_season, use_container_width=True)
                st.caption(
                    "Note: Certain pandemic-shortened seasons (e.g., 2020-21 bubble) are omitted "
                    "from NBA.com call-up logs, so gaps around 2020 reflect missing league play rather than zero activity."
                )

            if 'contract_type' in callups_df.columns:
                contract_counts = (
                    callups_df['contract_type']
                    .explode()
                    .dropna()
                    .value_counts()
                    .reset_index(name='count')
                    .rename(columns={'index': 'contract_type'})
                )
                if not contract_counts.empty:
                    fig_contract = px.pie(
                        contract_counts,
                        names='contract_type',
                        values='count',
                        title='Contract Types Issued',
                                )
                    st.plotly_chart(fig_contract, use_container_width=True)

                    st.markdown("---")
                    st.subheader("🏀 Teams Involved in Call-Ups")

            if 'gleague_teams' in callups_df.columns:
                gleague_counts = (
                    callups_df['gleague_teams']
                    .explode()
                    .dropna()
                    .value_counts()
                    .reset_index(name='count')
                    .head(15)
                )
                if not gleague_counts.empty:
                    fig_g = px.bar(
        gleague_counts,
        x='gleague_teams',
        y='count',
        labels={'gleague_teams': 'G-League Team', 'count': 'Call-Ups'},
        title='Top G-League Teams (Call-Ups)',
                    )
                    fig_g.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig_g, use_container_width=True)

            if 'nba_teams' in callups_df.columns:
                nba_counts = (
                    callups_df['nba_teams']
                    .explode()
                    .dropna()
                    .value_counts()
                    .reset_index(name='count')
                    .head(15)
                )
                if not nba_counts.empty:
                            fig_n = px.bar(
                nba_counts,
                x='nba_teams',
                y='count',
                labels={'nba_teams': 'NBA Team', 'count': 'Call-Ups'},
                title='Top NBA Teams (Call-Ups)',
                            )
                            fig_n.update_layout(xaxis_tickangle=-45)
                            st.plotly_chart(fig_n, use_container_width=True)

            if players_df is not None and 'player_name' in callups_df.columns:
                merged_players = players_df.copy()
                name_col = 'full_name' if 'full_name' in merged_players.columns else 'player_name'
                merged_players[name_col] = merged_players[name_col].astype(str).str.strip()
                callups_players = callups_df.merge(
                    merged_players[[name_col, 'position']].rename(columns={name_col: 'player_name'}),
                    on='player_name',
                    how='left'
                )
                pos_counts = callups_players['position'].dropna().value_counts()
                if not pos_counts.empty:
                    st.markdown("---")
                    st.subheader("📌 Positions of Called-Up Players")
                    fig_pos = px.bar(
                        x=pos_counts.index,
                        y=pos_counts.values,
                        labels={'x': 'Position', 'y': 'Number of Call-Ups'},
                        title='Positions Receiving Call-Ups'
                    )
                    st.plotly_chart(fig_pos, use_container_width=True)
        else:
            st.warning("Call-up data not yet available")

    elif dataset == "Prediction Dataset (Cleaned)" and prediction_df is not None:
        st.subheader("📈 Prediction Dataset (Cleaned)")
        st.write(f"Total records: {len(prediction_df)}")
        st.dataframe(prediction_df, use_container_width=True)

# ==================== MODEL INSIGHTS PAGE ====================
elif page == "🔍 Model Insights":
    st.header("🔍 Model Insights & Performance")
    
    tab1, tab2 = st.tabs(["📊 Performance Metrics", "🎯 Feature Importance"])
    model_metrics_summary = {}
    
    with tab1:
        st.subheader("Model Performance Metrics")

        if prediction_df is None or (log_reg_model is None and xgb_model is None):
            st.warning("Prediction dataset or trained models not available.")
        else:
            exclude_cols = ['player_name', 'season_year', 'called_up']
            feature_cols = [c for c in prediction_df.columns if c not in exclude_cols]

            top_feature_list = None
            if feature_importance_df is not None:
                top_feature_list = feature_importance_df.sort_values(
                    'Abs_Correlation', ascending=False
                )['Feature'].head(5).tolist()
            else:
                top_feature_list = feature_cols[:5] if feature_cols else []

            st.markdown("""
            **Feature set**: {} engineered player-season statistics{}.
            """.format(
                len(feature_cols),
                f" (top signals: {', '.join(top_feature_list)})" if top_feature_list else ""
            ))

            # Historical benchmarks from analysis.ipynb (validation set results)
            LOGREG_TUNED_DISPLAY = {
                'Model': 'Logistic Regression',
                'F1-Score': 0.280702,
                'Precision': 0.173913,
                'Recall': 0.727273,
                'ROC-AUC': 0.846591,
                'PR-AUC': 0.338023,
            }
            XGB_BASELINE_DISPLAY = {
                'Model': 'XGBoost',
                'F1-Score': 0.300000,
                'Precision': 0.333333,
                'Recall': 0.272727,
                'ROC-AUC': 0.830966,
                'PR-AUC': 0.230214,
            }

            st.markdown("#### Model Performance (from `analysis.ipynb` validation results)")
            historical_rows = []
            if log_reg_model is not None:
                historical_rows.append(LOGREG_TUNED_DISPLAY)
                model_metrics_summary['Logistic Regression'] = {
                    'f1': LOGREG_TUNED_DISPLAY['F1-Score'],
                    'precision': LOGREG_TUNED_DISPLAY['Precision'],
                    'recall': LOGREG_TUNED_DISPLAY['Recall'],
                    'roc_auc': LOGREG_TUNED_DISPLAY['ROC-AUC'],
                    'pr_auc': LOGREG_TUNED_DISPLAY['PR-AUC'],
                }
            if xgb_model is not None:
                historical_rows.append(XGB_BASELINE_DISPLAY)
                model_metrics_summary['XGBoost'] = {
                    'f1': XGB_BASELINE_DISPLAY['F1-Score'],
                    'precision': XGB_BASELINE_DISPLAY['Precision'],
                    'recall': XGB_BASELINE_DISPLAY['Recall'],
                    'roc_auc': XGB_BASELINE_DISPLAY['ROC-AUC'],
                    'pr_auc': XGB_BASELINE_DISPLAY['PR-AUC'],
                }

            if historical_rows:
                st.dataframe(
                    pd.DataFrame(historical_rows).style.format({
                        'F1-Score': "{:.3f}",
                        'Precision': "{:.3f}",
                        'Recall': "{:.3f}",
                        'ROC-AUC': "{:.3f}",
                        'PR-AUC': "{:.3f}",
                    }),
                    use_container_width=True,
                )
            else:
                st.info("Run `analysis.ipynb` to populate model benchmarks.")
    
    with tab2:
        st.subheader("Feature Importance")
        if feature_importance_df is not None:
            top_n = st.slider("Number of top features to display:", 5, 30, 20)
            fi = feature_importance_df.copy()
            fi = fi[~fi['Feature'].isin(['times_called_up', 'times_called_up_sqrt'])]
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

    st.markdown("---")
    st.markdown("### Why We Use Both Logistic Regression and XGBoost")

    if model_metrics_summary:
        lr_metrics = model_metrics_summary.get('Logistic Regression')
        xgb_metrics = model_metrics_summary.get('XGBoost')

        bullet_lines = []
        if lr_metrics:
            bullet_lines.append(
                f"- **Logistic Regression** keeps recall high "
                f"({lr_metrics['recall']:.2%}) and surfaces calibrated probabilities "
                f"({lr_metrics['precision']:.2%} precision) so the gauge component remains easy to explain."
            )
        if xgb_metrics:
            bullet_lines.append(
                f"- **XGBoost** offers crisper precision "
                f"({xgb_metrics['precision']:.2%}) and stronger overall F1 ({xgb_metrics['f1']:.2%}), "
                "which we use for binary call-up flags and leaderboard rankings."
            )
        if bullet_lines:
            st.markdown("\n".join(bullet_lines))

    st.markdown(
        "- Together, LR provides transparent probability estimates for every player, while XGBoost flags the "
        "most call-up ready prospects with fewer false positives. Coaches get interpretable insights and a "
        "high-precision shortlist in the same view."
    )
# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with by DS3 @ UCSD | Project: Next Up | 2024-2025</p>
</div>
""", unsafe_allow_html=True)

