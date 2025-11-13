# SportsRadar API Setup & Data Collection

## ✅ Your API Configuration

**API Key**: V8Wju3W5auIw8fwxWpInSgVEEE7VNvMDMaiiDjXq  
**Base URL**: https://api.sportradar.us/nbdl  
**Version**: v8  
**Access Level**: trial  
**Language**: en

## 🚀 Quick Start

The data collection notebook (`data/data.ipynb`) is now configured with your API key and ready to use.

### 1. Install Dependencies

```bash
pip install pandas numpy requests python-dotenv
```

### 2. Run the Notebook

Open `data/data.ipynb` in Jupyter and follow these steps:

1. **Run Cells 1-3**: Import libraries and set up API configuration
2. **Run Cell 5**: Test API connection (verify it works)
3. **Run Cell 20**: **RUN THIS FIRST** - Test with single season (2023)
4. **If test succeeds**, set `TEST_MODE = False` in Cell 20
5. **Run Cells 12-18**: Collect all 6 seasons of data

### 3. Expected Runtime

- **Test Mode** (1 season): ~2-5 minutes
- **Full Collection** (6 seasons): ~15-45 minutes (depends on API rate limits)

## 📊 SportsRadar NBDL API Endpoints

The notebook uses these endpoints:

### League Structure

```
GET /nbdl/trial/v8/en/league/hierarchy.json?api_key=YOUR_KEY
```

Returns: Teams, divisions, conferences

### Season Standings

```
GET /nbdl/trial/v8/en/seasons/2023/REG/standings.json?api_key=YOUR_KEY
```

Returns: Team standings for a season

### Team Roster

```
GET /nbdl/trial/v8/en/teams/{team_id}/profile.json?api_key=YOUR_KEY
```

Returns: Team profile with player roster

### Player Profile

```
GET /nbdl/trial/v8/en/players/{player_id}/profile.json?api_key=YOUR_KEY
```

Returns: Player details and career stats

### NBA Transactions (for call-up data)

```
GET /nba/trial/v8/en/league/2023/REG/transactions.json?api_key=YOUR_KEY
```

Returns: NBA transactions including G-League call-ups

## 📁 Output Files

After running the full collection, you'll have:

1. **test_output.csv** - Test run data (from single season)
2. **gleague_teams.csv** - All G-League teams
3. **gleague_player_stats.csv** - Player statistics (2018-2024)
4. **nba_callup_records.csv** - Call-up transaction history
5. **gleague_stats_with_callups.csv** - **Main dataset** with call-up labels
6. **hierarchy_raw.json** - Raw API response (league structure)
7. **raw_seasons_data.json** - Raw API responses (all seasons)

## 🔍 Data Fields Collected

### Player Demographics

- Full name, first name, last name
- Player ID (unique identifier)
- Position, height, weight
- Birth date, birth place
- College, high school
- Draft year, round, pick, team

### Performance Stats (Per Season)

- Games played, games started
- Minutes (total and per game)
- Points (total and per game)
- Rebounds (total, offensive, defensive, per game)
- Assists, steals, blocks (total and per game)
- Turnovers, personal fouls
- Field goal %, 3-point %, free throw %
- Shot attempts and makes for each category
- Plus/minus

### Team Context

- Team ID and name
- Season year
- Team market/location

### Target Variable

- **called_up**: Binary (1 = called up to NBA, 0 = not called up)

## ⚠️ Troubleshooting

### "HTTP Error 401: Unauthorized"

- API key may be invalid or expired
- Check if key has been activated
- Verify no extra spaces in the key

### "HTTP Error 403: Forbidden"

- Endpoint may require higher subscription tier
- Trial accounts may have limited access to certain endpoints
- Contact SportsRadar support to verify NBDL API access

### "HTTP Error 429: Too Many Requests"

- Rate limit exceeded
- Increase the `delay` parameter in `make_api_request()` (try 2.0 or 3.0 seconds)
- Wait a few minutes before retrying

### "No players extracted"

- API response structure may differ from expected
- Check the raw JSON files (`hierarchy_raw.json`, `test_standings.json`)
- You may need to adjust the extraction functions based on actual response structure

### Empty standings or teams

- Season year format may be wrong
- Try different season years (2022, 2023, 2024)
- Check if G-League uses different season notation

## 🔄 Alternative Data Collection

If SportsRadar API doesn't work or is too limited:

### Basketball Reference (Web Scraping)

```python
# Example URL structure
https://www.basketball-reference.com/gleague/years/2024.html
```

- Comprehensive stats
- Free to scrape (check their ToS)
- No authentication required

### RealGM

```
https://basketball.realgm.com/nba/transactions
```

- Great for call-up transactions
- Agent information available
- Team news and roster moves

### NBA Stats API (Unofficial)

```python
# G-League endpoints may exist
https://stats.nba.com/stats/...
```

- Free, no API key
- Undocumented, may change
- Need to inspect network requests

## 📝 Next Steps After Data Collection

1. **Verify data quality**

   - Check for missing values
   - Verify player counts match expected (~1000-1500 player-seasons)
   - Ensure call-up rate is reasonable (5-15%)

2. **Run EDA** (`eda.ipynb`)

   - Distribution of stats
   - Correlation analysis
   - Class balance (called up vs not)

3. **Feature engineering**

   - Calculate age from birth date
   - Create efficiency metrics (TS%, eFG%, PER)
   - Encode categorical variables
   - Add per-36-minute stats

4. **Collect supplementary data**
   - Agent affiliations (HoopsHype)
   - NBA team injury reports
   - Team roster needs analysis
   - Prior NBA experience

## 💡 Tips

- **Start with test mode**: Always run the test cell first to validate API access
- **Save intermediate results**: The notebook saves raw JSON responses for debugging
- **Monitor rate limits**: Trial accounts typically allow 1 request/second
- **Check response structures**: API responses may vary - inspect the raw JSON files
- **Be patient**: Full collection with 6 seasons and ~30 teams will take time

## 📚 Resources

- **SportsRadar Docs**: https://developer.sportradar.com/
- **NBDL/G-League**: Official G-League stats
- **Project GitHub**: [Your repo URL]
- **DS3**: [Project page]

## 🆘 Getting Help

If you encounter issues:

1. Check the raw JSON files to understand API response structure
2. Review SportsRadar documentation
3. Contact DS3 mentor or TritonBall mentor
4. Consider alternative data sources if API access is limited

---

**Last Updated**: November 2024  
**API Key Status**: Active (as of setup)  
**Notebook Version**: v1.0
