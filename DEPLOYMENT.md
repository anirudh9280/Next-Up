# 🚀 Deployment Guide: Next Up Streamlit App

This guide will help you deploy your Next Up application to Streamlit Cloud (free and easy) or other platforms.

## 📋 Prerequisites

Before deploying, make sure you have:

1. ✅ All data files committed to Git (CSV files in `raw/` and `data/` directories)
2. ✅ `requirements.txt` is up to date with all dependencies
3. ✅ Your code is pushed to a GitHub repository
4. ✅ A Streamlit Cloud account (free at [share.streamlit.io](https://share.streamlit.io))

---

## 🌐 Option 1: Streamlit Cloud (Recommended - Free & Easy)

### Step 1: Prepare Your Repository

1. **Ensure all data files are committed:**

   ```bash
   git add raw/*.csv
   git add data/*.csv
   git commit -m "Add data files for deployment"
   git push
   ```

2. **Verify your repository structure:**
   ```
   Next-Up/
   ├── streamlit_app.py          # Root entry point
   ├── requirements.txt           # Dependencies
   ├── .streamlit/
   │   └── config.toml           # Streamlit config
   ├── raw/                       # Data files
   │   ├── gleague_players.csv
   │   ├── gleague_rosters.csv
   │   └── gleague_teams.csv
   └── data/                      # Additional data
       └── external/
           └── callups.csv (optional)
   ```

### Step 2: Deploy to Streamlit Cloud

1. **Go to [share.streamlit.io](https://share.streamlit.io)**

   - Sign in with your GitHub account

2. **Click "New app"**

3. **Fill in the deployment form:**

   - **Repository**: Select your `Next-Up` repository
   - **Branch**: `main` (or your default branch)
   - **Main file path**: `streamlit_app.py`
   - **App URL**: Choose a custom subdomain (e.g., `next-up-gleague`)

4. **Click "Deploy"**

5. **Wait for deployment** (usually 1-2 minutes)

6. **Your app is live!** 🎉

### Step 3: Update App (After Code Changes)

Simply push to your GitHub repository:

```bash
git add .
git commit -m "Update app"
git push
```

Streamlit Cloud will automatically redeploy your app!

---

## 🐳 Option 2: Docker Deployment

If you prefer Docker or need more control:

### Step 1: Create Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Step 2: Build and Run

```bash
# Build image
docker build -t next-up-app .

# Run container
docker run -p 8501:8501 next-up-app
```

### Step 3: Deploy to Cloud

- **Heroku**: Use `heroku container:push web`
- **AWS ECS/Fargate**: Push to ECR and deploy
- **Google Cloud Run**: `gcloud run deploy`
- **Azure Container Instances**: Use Azure CLI

---

## ☁️ Option 3: Other Cloud Platforms

### Heroku

1. Create `Procfile`:

   ```
   web: streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. Deploy:
   ```bash
   heroku create next-up-app
   git push heroku main
   ```

### AWS EC2 / Google Cloud Compute

1. SSH into your instance
2. Clone repository
3. Install dependencies: `pip install -r requirements.txt`
4. Run: `streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0`
5. Configure firewall to allow port 8501

---

## 🔧 Configuration

### Streamlit Config (`.streamlit/config.toml`)

Already created! This file configures:

- Theme colors
- Server settings
- Browser behavior

### Environment Variables

If you need API keys or secrets:

1. **Streamlit Cloud**: Go to app settings → Secrets → Add secrets
2. **Local**: Create `.streamlit/secrets.toml` (not committed to Git)

Example `.streamlit/secrets.toml`:

```toml
[api]
api_key = "your-api-key-here"
```

Access in code:

```python
import streamlit as st
api_key = st.secrets["api"]["api_key"]
```

---

## ✅ Pre-Deployment Checklist

- [ ] All data CSV files are in the repository
- [ ] `requirements.txt` includes all dependencies
- [ ] `streamlit_app.py` exists at root level
- [ ] Paths in code work correctly (use `Path(__file__).parent`)
- [ ] Test app locally: `streamlit run streamlit_app.py`
- [ ] Code is pushed to GitHub
- [ ] No hardcoded API keys or secrets
- [ ] `.env` file is in `.gitignore` (already done)

---

## 🐛 Troubleshooting

### "ModuleNotFoundError" or Import Errors

- Check `requirements.txt` has all packages
- Verify package names are correct
- Try: `pip install -r requirements.txt` locally first

### "FileNotFoundError" for Data Files

- Ensure CSV files are committed to Git
- Check file paths in code match repository structure
- Verify files exist: `ls raw/*.csv`

### App Won't Start

- Check Streamlit Cloud logs (click "Manage app" → "Logs")
- Verify `streamlit_app.py` is at root level
- Ensure Python version is compatible (3.8+)

### Slow Loading

- Data files might be too large (>100MB)
- Consider using `@st.cache_data` (already implemented)
- Optimize CSV files or use parquet format

### Port Already in Use (Local)

```bash
# Kill existing Streamlit process
pkill -f streamlit

# Or use different port
streamlit run streamlit_app.py --server.port=8502
```

---

## 📊 Monitoring & Updates

### Streamlit Cloud

- **View logs**: App settings → Logs
- **Restart app**: App settings → Restart app
- **View metrics**: App settings → Metrics (usage, errors)

### Update Dependencies

1. Update `requirements.txt`
2. Commit and push
3. Streamlit Cloud auto-redeploys

---

## 🔒 Security Notes

- ✅ Never commit `.env` files (already in `.gitignore`)
- ✅ Use Streamlit Secrets for API keys
- ✅ Don't hardcode credentials
- ✅ Review what data is public (CSV files in repo)

---

## 📚 Additional Resources

- [Streamlit Cloud Documentation](https://docs.streamlit.io/streamlit-community-cloud)
- [Streamlit Deployment Guide](https://docs.streamlit.io/deploy)
- [Streamlit Secrets Management](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app/secrets-management)

---

## 🎉 Success!

Once deployed, share your app URL with:

- Your team members
- Instructors
- Anyone interested in G-League call-up predictions!

**Example URL**: `https://next-up-gleague.streamlit.app`

---

**Need Help?** Check the troubleshooting section or Streamlit community forums.
