import pandas as pd

fisier = "Transactions-1744573527540.xlsx"

try:
    # Citește doar coloana D, ignorând primele 2 rânduri
    df = pd.read_excel(fisier, engine="openpyxl", usecols="D", skiprows=2, header=None)
except Exception as e:
    print("Eroare la citirea fișierului:", e)
    exit()

# Curățare: înlocuire ',' cu '.', eliminare 'lei' și spații
df.columns = ["Suma"]
df["Suma"] = df["Suma"].astype(str)
df["Suma"] = df["Suma"].str.replace("lei", "", case=False, regex=False)
df["Suma"] = df["Suma"].str.replace(",", ".", regex=False)
df["Suma"] = df["Suma"].str.replace(" ", "", regex=False)

# Convertim în float doar valorile valide numeric
df = df[pd.to_numeric(df["Suma"], errors='coerce').notnull()]
df["Suma"] = df["Suma"].astype(float)

# Verificare date
if df.empty:
    print("⚠️ Nu au fost găsite sume numerice valide în coloana D.")
    exit()

# Statistici
total_bets = len(df)
winning_bets = df[df["Suma"] > 0].shape[0]
losing_bets = df[df["Suma"] < 0].shape[0]
total_profit = df["Suma"].sum()
winning_pct = (winning_bets / total_bets) * 100
losing_pct = (losing_bets / total_bets) * 100

# Afișare rezultate
print("\n===== ANALIZĂ PARIURI =====")
print(f"Total pariuri: {total_bets}")
print(f"Pariuri câștigătoare: {winning_bets} ({winning_pct:.2f}%)")
print(f"Pariuri pierdute: {losing_bets} ({losing_pct:.2f}%)")
print(f"Profit total: {total_profit:.2f} lei")
