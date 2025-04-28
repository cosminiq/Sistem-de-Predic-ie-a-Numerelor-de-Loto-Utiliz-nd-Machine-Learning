import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Verificarea versiunii TensorFlow
print(f"Versiune TensorFlow: {tf.__version__}")

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

# Funcție pentru construirea modelului
def build_model(input_shape):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(80, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Funcție pentru antrenarea modelului
def train_model(model, X_train, y_train, X_val, y_val, epochs=250):
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        verbose=1
    )
    
    return model, history

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
    
    # Make prediction
    predictions = model.predict(X_recent)
    
    # Calculate final score combining model prediction with frequencies
    prediction_avg = np.mean(predictions, axis=0)
    final_score = prediction_avg * 0.7 + frequencies * 0.3
    
    return final_score



# Funcția principală actualizată cu verificări suplimentare
def main(file_path):
    # Încărcăm datele
    print("\n=== ÎNCĂRCAREA DATELOR ===")
    data = load_and_prepare_data(file_path)
    if data is None:
        return
    
    print(f"Numărul total de înregistrări din set: {len(data)}")
    print(f"Numărul de coloane/numere per înregistrare: {data.shape[1]}")
    print(f"Primele 300 rânduri din set:\n{data.head(300)}") # Afișăm primele 3 rânduri pentru verificare
    print(f"Ultimele 300 rânduri din set:\n{data.tail(300)}")  # Afișăm ultimele 3 rânduri pentru verificare
    
    # Determinăm câte trageri recente să folosim pentru predicție
    num_recent_draws = min(10, len(data))
    recent_draws = data.iloc[-num_recent_draws:]  # Keep as DataFrame
    print(f"\n=== DATE RECENTE PENTRU PREDICȚIE ===")
    print(f"Folosim ultimele {num_recent_draws} extrageri pentru predicții")
    print(f"Forma datelor recente: {recent_draws.shape}")
    
    # Creăm caracteristicile
    print("\n=== CREAREA CARACTERISTICILOR ===")
    X = create_features(data)
    print(f"Dimensiunea matricei de caracteristici X: {X.shape}")
    print(f"Suma totală a elementelor din X: {np.sum(X)}")  # Verificăm dacă avem elemente nenule
    
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
    
    # Construim și antrenăm modelul
    print("\n=== CONSTRUIREA ȘI ANTRENAREA MODELULUI ===")
    model = build_model(X.shape[1])
    print("Arhitectura modelului:")
    model.summary()
    
    print("\nÎnceperea antrenamentului...")
    model, history = train_model(model, X_train, y_train, X_val, y_val)
    print("Antrenament finalizat!")
    
    # Evaluam performanța modelului pe setul de validare
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Performanța finală pe setul de validare: Pierdere = {val_loss:.4f}, Acuratețe = {val_acc:.4f}")
    
    # Plot pentru a vizualiza antrenamentul
    print("\n=== GENERAREA GRAFICELOR DE ANTRENAMENT ===")
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Grafic de antrenament salvat ca 'training_history.png'")
    plt.close()
    
    # Facem predicția pentru următoarea extragere
    print("\n=== PREDICȚIA PENTRU URMĂTOAREA EXTRAGERE ===")
    final_score = predict_next_draw(model, recent_draws)
    
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
    
    # Afișăm distribuția scorurilor pentru toate numerele
    print("\n=== GENERAREA GRAFICULUI DE PREDICȚIE ===")
    plt.figure(figsize=(14, 6))
    plt.bar(range(1, 81), final_score)
    plt.title('Scoruri de predicție pentru toate numerele')
    plt.xlabel('Număr')
    plt.ylabel('Scor de predicție')
    plt.xticks(range(1, 81, 5))
    plt.grid(axis='y', alpha=0.3)
    
    # Evidențiem top 5 numere
    for num in top_5_numbers:
        plt.bar(num, final_score[num-1], color='red')
    
    plt.savefig('prediction_scores.png')
    print("Grafic de predicție salvat ca 'prediction_scores.png'")
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