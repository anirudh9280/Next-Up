import pandas as pd

players = pd.read_csv('../raw/gleague_players.csv')
template = pd.DataFrame({
    'player_name': players['full_name'],
    'nba_team': '',
    'callup_date': '',
    'contract_type': '',
    'called_up': 0
})
template.to_csv('external/callups_TEMPLATE.csv', index=False)
print(f"✅ Created callups_TEMPLATE.csv with {len(template)} players")
print(f"📍 Location: data/external/callups_TEMPLATE.csv")
print(f"\n📋 Next: Follow instructions in COLLECT_CALLUPS.md to fill it in")

