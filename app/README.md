# Next Up - Streamlit Application

## 🚀 Quick Start

### Run the app locally:

```bash
# From the project root directory
cd Next-Up
streamlit run app/app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📱 Application Structure

### Current Pages:

1. **🏠 Home** - Project overview and statistics
2. **👤 Player Prediction** - Predict call-up probability for individual players
3. **🏀 Team Analysis** - Analyze team rosters and call-up rates
4. **📊 Data Explorer** - Explore the datasets
5. **🔍 Model Insights** - Model performance and SHAP analysis

---

## ✅ What's Implemented (Barebones)

- ✅ Multi-page navigation
- ✅ Data loading and caching
- ✅ Basic visualizations (position distribution, team rosters)
- ✅ Player and team selection dropdowns
- ✅ Responsive layout
- ✅ Placeholder sections for model integration

---

## 🚧 TODO for Students (Week 5-6)

### Priority 1: Model Integration

- [ ] **Load trained model** from `models/callup_model.pkl`
- [ ] **Implement predictions** in Player Prediction page
- [ ] **Display confidence scores** with gauge charts
- [ ] **Show top contributing features** for each prediction

### Priority 2: Enhanced Analytics

- [ ] **Team comparison tool** - compare call-up rates across teams
- [ ] **Historical trends** - call-ups over time
- [ ] **Position-based analysis** - which positions get called up most
- [ ] **Filter functionality** - by date, team, position

### Priority 3: Model Insights

- [ ] **SHAP plots** - feature importance visualization
- [ ] **Confusion matrix** - model performance on test set
- [ ] **ROC and PR curves** - classification metrics
- [ ] **Historical validation** - predicted vs actual call-ups

### Priority 4: Polish

- [ ] **Export functionality** - download predictions as CSV
- [ ] **Improved styling** - custom CSS, better layouts
- [ ] **Error handling** - graceful degradation when data missing
- [ ] **Loading states** - spinners for long operations

---

## 🗂️ File Structure

```
app/
├── app.py              # Main Streamlit application
├── README.md           # This file
└── models/             # TODO: Create this folder for trained models
    └── callup_model.pkl  # TODO: Save your model here
```

---

## 📦 Dependencies

Make sure you have these installed (already in `requirements.txt`):

```
streamlit>=1.20.0
pandas>=1.5.0
numpy>=1.23.0
plotly>=5.11.0
```

---

## 🎨 Customization Guide

### Adding a New Page

1. Add option to sidebar radio:

```python
page = st.sidebar.radio(
    "Go to",
    ["Home", "Player Prediction", "Team Analysis", "YOUR NEW PAGE"]
)
```

2. Add page logic:

```python
elif page == "YOUR NEW PAGE":
    st.header("Your New Page")
    # Your code here
```

### Integrating Your Model

Replace the `load_model()` function:

```python
@st.cache_resource
def load_model():
    import joblib
    return joblib.load('models/callup_model.pkl')
```

Then use it in predictions:

```python
model = load_model()
prediction = model.predict_proba(player_features)
call_up_prob = prediction[0][1]  # Probability of call-up
```

---

## 🐛 Common Issues

### "Unable to load data" error

- Make sure you're running from project root: `streamlit run app/app.py`
- Check that `raw/` folder exists with CSV files

### Model not loading

- You need to train a model first in `analysis.ipynb`
- Save it with: `joblib.dump(model, 'app/models/callup_model.pkl')`
- Create the `app/models/` directory if it doesn't exist

---

## 📚 Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Documentation](https://plotly.com/python/)
- [SHAP Integration Guide](https://shap.readthedocs.io/)

---

## 🎯 Deployment (Week 6)

Once complete, deploy to Streamlit Cloud:

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io/)
3. Connect your repo and select `app/app.py`
4. Add secrets (if needed)
5. Deploy!

---

**Good luck building! 🚀**
