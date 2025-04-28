import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss
import matplotlib.pyplot as plt
import seaborn as sns

# Verificarea versiunii XGBoost
print(f"Versiune XGBoost: {xgb.__version__}")

# Funcție pentru citirea și pregătirea datelor
def load_and_prepare_data(file_path):
    try:
        # Citirea datelor din CSV
        data = pd.read_csv(file_path, header=None)
        
        # Dacă datele sunt într-o singură coloană, separate prin virgulă
        if data.shape[1] == 1:
            # Separăm numerele din fiecare rând
            data = data[0].str.split(',', expand=True).astype(int)
        
        print(f"Date încărcate cu succes. Forma: {data.shape}")
        return data
    except Exception as e:
        print(f"Eroare la încărcarea datelor: {e}")
        return None

# Funcție pentru crearea caracteristicilor
def create_features(data):
    # Determine if input is DataFrame or numpy array
    if isinstance(data, pd.DataFrame):
        data_values = data.values
    else:
        data_values = np.array(data)
    
    # Get number of draws and numbers per draw
    num_draws = len(data_values)
    try:
        nums_per_draw = data_values.shape[1]
    except IndexError:
        nums_per_draw = 1
    
    # Create features matrix
    features = np.zeros((num_draws, 80))
    
    for i in range(num_draws):
        draw = data_values[i]
        # Ensure draw is iterable
        if isinstance(draw, (int, np.integer)):
            draw = [draw]
        for num in draw:
            if 1 <= num <= 80:
                features[i, num-1] = 1
    
    return features

# Funcție pentru antrenarea modelului XGBoost
def train_xgboost_model(X_train, y_train, X_val, y_val):
    # Parametrii modelului XGBoost
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'eta': 0.1,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'gamma': 0,
        'seed': 42
    }
    
    print("Începerea antrenamentului cu XGBoost...")
    
    # Convertirea datelor în formatul DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Lista de evaluare
    evallist = [(dtrain, 'train'), (dval, 'validation')]
    
    # Antrenarea modelului
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=200,
        evals=evallist,
        early_stopping_rounds=20,
        verbose_eval=20
    )
    
    print("Antrenament finalizat!")
    return model

# Funcție pentru predicție
def predict_next_draw(model, recent_draws):
    # Calculate frequencies for each number
    frequencies = np.zeros(80)
    for draw in recent_draws:
        # Ensure draw is iterable
        if isinstance(draw, (int, np.integer)):
            draw = [draw]
        for num in draw:
            if 1 <= num <= 80:
                frequencies[num-1] += 1
    
    # Normalize frequencies
    if np.max(frequencies) > 0:
        frequencies = frequencies / np.max(frequencies)
    
    # Create features for recent draws
    X_recent = create_features(recent_draws)
    
    # Make prediction with XGBoost
    drecent = xgb.DMatrix(X_recent)
    predictions = model.predict(drecent)
    
    # Reshape predictions if needed (for multi-output case)
    if len(predictions.shape) == 1 and predictions.shape[0] == X_recent.shape[0]:
        # Only one value per row, need to reshape
        reshaped_predictions = np.zeros((X_recent.shape[0], 80))
        for i in range(X_recent.shape[0]):
            reshaped_predictions[i] = predictions[i]
        predictions = reshaped_predictions
    
    # Calculate final score combining model prediction with frequencies
    prediction_avg = np.mean(predictions, axis=0)
    final_score = prediction_avg * 0.7 + frequencies * 0.3
    
    return final_score, frequencies, prediction_avg

# Funcție pentru vizualizarea importanței caracteristicilor
def plot_feature_importance(model):
    importance = model.get_score(importance_type='gain')
    features = list(importance.keys())
    values = list(importance.values())
    
    # Convertim în numere reale (1-80)
    features_real = [int(f.replace('f', '')) + 1 for f in features]
    
    # Sortăm caracteristicile după importanță
    sorted_idx = np.argsort(values)
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(sorted_idx)), [values[i] for i in sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), [features_real[i] for i in sorted_idx])
    plt.title('Importanța numerelor în model')
    plt.xlabel('Importanță (Gain)')
    plt.ylabel('Număr')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("Grafic cu importanța caracteristicilor salvat ca 'feature_importance.png'")
    plt.close()

# Funcție pentru vizualizarea predicțiilor comparative
def plot_prediction_components(final_score, frequencies, prediction_avg):
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Predicția finală
    plt.subplot(3, 1, 1)
    plt.bar(range(1, 81), final_score)
    plt.title('Scorul final de predicție (70% model + 30% frecvențe)')
    plt.xlabel('Număr')
    plt.ylabel('Scor final')
    plt.xticks(range(1, 81, 5))
    plt.grid(axis='y', alpha=0.3)
    
    # Plot 2: Predicția modelului
    plt.subplot(3, 1, 2)
    plt.bar(range(1, 81), prediction_avg, color='green')
    plt.title('Predicția modelului XGBoost')
    plt.xlabel('Număr')
    plt.ylabel('Scor model')
    plt.xticks(range(1, 81, 5))
    plt.grid(axis='y', alpha=0.3)
    
    # Plot 3: Frecvențele istorice
    plt.subplot(3, 1, 3)
    plt.bar(range(1, 81), frequencies, color='orange')
    plt.title('Frecvențele din extragerile recente')
    plt.xlabel('Număr')
    plt.ylabel('Frecvență normalizată')
    plt.xticks(range(1, 81, 5))
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('prediction_components.png')
    print("Grafic cu componentele de predicție salvat ca 'prediction_components.png'")
    plt.close()

# Funcție pentru a vizualiza distribuția scorurilor
def plot_score_distribution(final_score, top_numbers):
    plt.figure(figsize=(14, 6))
    
    # Sortăm scorurile pentru vizualizare
    sorted_scores = np.sort(final_score)[::-1]
    
    # Plot pentru distribuția generală a scorurilor
    plt.subplot(1, 2, 1)
    plt.bar(range(1, 81), sorted_scores)
    plt.title('Distribuția scorurilor (sortate)')
    plt.xlabel('Rang')
    plt.ylabel('Scor')
    plt.grid(axis='y', alpha=0.3)
    
    # Plot pentru topul scorurilor
    plt.subplot(1, 2, 2)
    plt.bar(range(1, len(top_numbers) + 1), [final_score[num-1] for num in top_numbers])
    plt.xticks(range(1, len(top_numbers) + 1), top_numbers)
    plt.title(f'Top {len(top_numbers)} numere recomandate')
    plt.xlabel('Număr')
    plt.ylabel('Scor')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('score_distribution.png')
    print("Grafic cu distribuția scorurilor salvat ca 'score_distribution.png'")
    plt.close()

# Funcția principală
def main(file_path):
    # Încărcăm datele
    print("\n=== ÎNCĂRCAREA DATELOR ===")
    data = load_and_prepare_data(file_path)
    if data is None:
        return
    
    print(f"Numărul total de înregistrări din set: {len(data)}")
    print(f"Numărul de coloane/numere per înregistrare: {data.shape[1]}")
    print(f"Primele 3000 rânduri din set:\n{data.head(3000)}") # Afișăm primele 3000 rânduri
    print(f"Ultimele 3000 rânduri din set:\n{data.tail(3000)}") # Afișăm ultimele 3000 rânduri
    print(f"Tipul datelor: {data.dtypes}")
    

    
    # Determinăm câte trageri recente să folosim pentru predicție
    num_recent_draws = min(10, len(data))
    recent_draws = data.iloc[-num_recent_draws:]
    print(f"\n=== DATE RECENTE PENTRU PREDICȚIE ===")
    print(f"Folosim ultimele {num_recent_draws} extrageri pentru predicții")
    print(f"Forma datelor recente: {recent_draws.shape}")
    
    # Creăm caracteristicile
    print("\n=== CREAREA CARACTERISTICILOR ===")
    X = create_features(data)
    print(f"Dimensiunea matricei de caracteristici X: {X.shape}")
    print(f"Suma totală a elementelor din X: {np.sum(X)}")
    
    # Obiectivul este de a prezice numerele din următoarea extragere
    y = np.roll(X, -1, axis=0)
    y = y[:-1]  # Eliminăm ultima linie care nu are țintă
    X = X[:-1]  # Eliminăm ultima linie pentru a păstra dimensiunile
    
    print(f"După pregătirea pentru antrenare - X: {X.shape}, y: {y.shape}")
    
    # Verificăm distribuția numerelor
    num_frequency = np.sum(X, axis=0)
    print(f"Top 5 cele mai frecvente numere: {np.argsort(num_frequency)[-5:]+1}")
    print(f"Top 5 cele mai rare numere: {np.argsort(num_frequency)[:5]+1}")
    
    # Împărțim datele în set de antrenare și validare
    print("\n=== ÎMPĂRȚIREA DATELOR ===")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Set antrenare: X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Set validare: X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"Procentul de date folosite pentru antrenare: {(len(X_train) / len(X)) * 100:.1f}%")
    print(f"Procentul de date folosite pentru validare: {(len(X_val) / len(X)) * 100:.1f}%")
    
    # Antrenăm modelul XGBoost
    print("\n=== ANTRENAREA MODELULUI XGBOOST ===")
    model = train_xgboost_model(X_train, y_train, X_val, y_val)
    
    # Evaluăm performanța modelului pe setul de validare
    print("\n=== EVALUAREA MODELULUI ===")
    dval = xgb.DMatrix(X_val)
    y_pred = model.predict(dval)
    
    # Pentru evaluarea binară
    y_pred_binary = (y_pred > 0.5).astype(int)
    accuracy = accuracy_score(y_val.flatten(), y_pred_binary.flatten())
    loss = log_loss(y_val.flatten(), y_pred.flatten())
    
    print(f"Performanța finală pe setul de validare: Pierdere = {loss:.4f}, Acuratețe = {accuracy:.4f}")
    
    # Vizualizăm importanța caracteristicilor
    print("\n=== VIZUALIZAREA IMPORTANȚEI CARACTERISTICILOR ===")
    plot_feature_importance(model)
    
    # Facem predicția pentru următoarea extragere
    print("\n=== PREDICȚIA PENTRU URMĂTOAREA EXTRAGERE ===")
    final_score, frequencies, prediction_avg = predict_next_draw(model, recent_draws)
    
    # Verificăm scorurile de predicție
    print(f"Scor minim: {np.min(final_score):.4f}, Scor maxim: {np.max(final_score):.4f}")
    print(f"Scor mediu: {np.mean(final_score):.4f}, Scor median: {np.median(final_score):.4f}")
    
    # Sortăm numerele în funcție de scor și obținem indexurile
    sorted_indices = np.argsort(final_score)[::-1]
    
    # Afișăm top 5 numere recomandate
    top_5_numbers = sorted_indices[:5] + 1  # Adăugăm 1 pentru a transforma din index în număr
    print("\nTop 5 numere recomandate (în ordine descrescătoare a probabilității):")
    for i, num in enumerate(top_5_numbers, 1):
        print(f"{i}. Numărul {num} - Scor: {final_score[num-1]:.4f}")
    
    # Afișăm top 4 numere recomandate
    top_4_numbers = sorted_indices[:4] + 1
    print("\nTop 4 numere recomandate (în ordine descrescătoare a probabilității):")
    for i, num in enumerate(top_4_numbers, 1):
        print(f"{i}. Numărul {num} - Scor: {final_score[num-1]:.4f}")
    
    # Generăm grafice suplimentare
    print("\n=== GENERAREA GRAFICELOR COMPARATIVE ===")
    plot_prediction_components(final_score, frequencies, prediction_avg)
    plot_score_distribution(final_score, top_5_numbers)
    
    # Vizualizăm heatmap pentru distribuția numerelor
    plt.figure(figsize=(12, 8))
    reshaped_data = final_score.reshape(8, 10)
    sns.heatmap(reshaped_data, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title('Heatmap al scorurilor de predicție')
    plt.xlabel('Cifra unităților (0-9)')
    plt.ylabel('Grupul de zeci (1-8)')
    plt.xticks(np.arange(0.5, 10.5), range(0, 10))
    plt.yticks(np.arange(0.5, 8.5), range(1, 9))
    plt.savefig('score_heatmap.png')
    print("Heatmap salvat ca 'score_heatmap.png'")
    plt.close()
    
    print("\n=== SUMAR FINAL ===")
    print(f"Date procesate: {len(data)} extrageri")
    print(f"Date folosite pentru antrenare: {len(X_train)} extrageri")
    print(f"Date folosite pentru validare: {len(X_val)} extrageri")
    print(f"Date folosite pentru predicție: {num_recent_draws} extrageri recente")
    
    return {
        'top_5': top_5_numbers.tolist(),
        'top_4': top_4_numbers.tolist(),
        'model': model
    }

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Utilizare: python script.py cale_fisier.csv")
        file_path = input("Introduceți calea către fișierul CSV: ")
    else:
        file_path = sys.argv[1]
    
    result = main(file_path)
    
    print("\nAnaliză completă. Verificați imaginile generate pentru detalii suplimentare.")