import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import datetime
import base64
from io import BytesIO

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
def train_model(model, X_train, y_train, X_val, y_val, epochs=300):
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

# Funcție nouă pentru a salva rezultatele într-un fișier HTML interactiv
def save_results_to_html(top_numbers, scores, history, recent_draws):
    now = datetime.datetime.now()
    date_str = now.strftime("%Y%m%d")
    time_str = now.strftime("%H%M")
    filename = f"rezultat_{date_str}_{time_str}.html"
    
    # Convertim imaginile pentru a le include în HTML
    # Training history plot
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
    
    buf = BytesIO()
    plt.savefig(buf, format="png")
    img_str_training = base64.b64encode(buf.getvalue()).decode()
    plt.close()
    
    # Prediction scores plot
    plt.figure(figsize=(14, 6))
    all_scores = np.zeros(80)
    for i in range(80):
        all_scores[i] = scores[i]
    
    bars = plt.bar(range(1, 81), all_scores)
    plt.title('Scoruri de predicție pentru toate numerele')
    plt.xlabel('Număr')
    plt.ylabel('Scor de predicție')
    plt.xticks(range(1, 81, 5))
    plt.grid(axis='y', alpha=0.3)
    
    # Highlight top 5 numbers
    for num in top_numbers['top_5']:
        bars[num-1].set_color('red')
    
    buf = BytesIO()
    plt.savefig(buf, format="png")
    img_str_prediction = base64.b64encode(buf.getvalue()).decode()
    plt.close()
    
    # Pregătirea datelor pentru JavaScript
    scores_js = [float(score) for score in scores]
    
    # Construirea conținutului HTML
    html_content = f"""
    <!DOCTYPE html>
    <html lang="ro">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Rezultat Predicții - {date_str}</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
        <style>
            :root {{
                --primary-color: #4285f4;
                --secondary-color: #db4437;
                --background-color: #f8f9fa;
                --card-bg: #ffffff;
                --text-color: #202124;
                --grid-item-bg: #e8f0fe;
                --accent-color: #fbbc05;
            }}
            
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 0;
                background-color: var(--background-color);
                color: var(--text-color);
                line-height: 1.6;
            }}
            
            .container {{
                width: 100%;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                box-sizing: border-box;
            }}
            
            header {{
                background-color: var(--primary-color);
                color: white;
                padding: 20px 0;
                text-align: center;
                border-radius: 0 0 10px 10px;
                margin-bottom: 30px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }}
            
            h1, h2, h3 {{
                margin-top: 0;
            }}
            
            .card {{
                background-color: var(--card-bg);
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
                margin-bottom: 25px;
                padding: 20px;
                transition: transform 0.3s ease;
            }}
            
            .card:hover {{
                transform: translateY(-5px);
            }}
            
            .chart-container {{
                position: relative;
                height: 400px;
                width: 100%;
                margin-bottom: 30px;
            }}
            
            .results-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                grid-gap: 20px;
                margin-top: 20px;
            }}
            
            .num-card {{
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                background-color: var(--grid-item-bg);
                border-radius: 10px;
                padding: 15px;
                box-shadow: 0 3px 5px rgba(0, 0, 0, 0.1);
                transition: all 0.3s ease;
            }}
            
            .num-card.top-pick {{
                background-color: var(--secondary-color);
                color: white;
            }}
            
            .num-card:hover {{
                transform: scale(1.05);
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
            }}
            
            .number {{
                font-size: 2.5rem;
                font-weight: bold;
                margin-bottom: 5px;
            }}
            
            .score {{
                font-size: 0.9rem;
                color: #5f6368;
            }}
            
            .num-card.top-pick .score {{
                color: rgba(255, 255, 255, 0.9);
            }}
            
            .summary {{
                margin-bottom: 30px;
            }}
            
            .recent-draws {{
                margin-top: 20px;
            }}
            
            .draws-table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 10px;
            }}
            
            .draws-table th, .draws-table td {{
                padding: 10px;
                text-align: center;
                border: 1px solid #e0e0e0;
            }}
            
            .draws-table th {{
                background-color: var(--primary-color);
                color: white;
            }}
            
            .draws-table tr:nth-child(even) {{
                background-color: rgba(0, 0, 0, 0.03);
            }}
            
            .toggle-container {{
                display: flex;
                justify-content: center;
                margin: 20px 0;
            }}
            
            .toggle-btn {{
                padding: 8px 16px;
                margin: 0 10px;
                background-color: var(--primary-color);
                color: white;
                border: none;
                border-radius: 20px;
                cursor: pointer;
                transition: background-color 0.3s;
            }}
            
            .toggle-btn:hover {{
                background-color: #3367d6;
            }}
            
            .toggle-btn.active {{
                background-color: var(--secondary-color);
            }}
            
            .images-section {{
                margin-top: 30px;
            }}
            
            .img-container {{
                max-width: 100%;
                margin-bottom: 20px;
                overflow-x: auto;
            }}
            
            .img-container img {{
                max-width: 100%;
                height: auto;
            }}
            
            footer {{
                text-align: center;
                margin-top: 40px;
                padding: 20px 0;
                color: #5f6368;
                font-size: 0.9rem;
                border-top: 1px solid #e0e0e0;
            }}

            /* Dark Mode Toggle */
            .theme-switch {{
                position: fixed;
                top: 20px;
                right: 20px;
                z-index: 1000;
            }}
            
            .theme-switch input {{
                opacity: 0;
                width: 0;
                height: 0;
            }}
            
            .theme-switch label {{
                cursor: pointer;
                padding: 10px;
                background-color: #333;
                color: white;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                display: flex;
                align-items: center;
                justify-content: center;
                box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            }}
            
            /* Dark Theme */
            .dark-theme {{
                --primary-color: #8ab4f8;
                --secondary-color: #f28b82;
                --background-color: #202124;
                --card-bg: #303134;
                --text-color: #e8eaed;
                --grid-item-bg: #3c4043;
                --accent-color: #fdd663;
            }}
            
            .dark-theme .score {{
                color: #bdc1c6;
            }}
            
            .dark-theme .draws-table th {{
                background-color: #525355;
            }}
            
            .dark-theme .draws-table td {{
                border-color: #5f6368;
            }}
            
            .dark-theme footer {{
                color: #9aa0a6;
                border-color: #5f6368;
            }}
            
            @media (max-width: 768px) {{
                .container {{
                    padding: 10px;
                }}
                
                .chart-container {{
                    height: 300px;
                }}
                
                .results-grid {{
                    grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
                }}
                
                .number {{
                    font-size: 2rem;
                }}
                
                .toggle-container {{
                    flex-direction: column;
                    align-items: center;
                }}
                
                .toggle-btn {{
                    margin: 5px 0;
                    width: 200px;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="theme-switch">
            <input type="checkbox" id="theme-toggle">
            <label for="theme-toggle">☀️</label>
        </div>
        
        <header>
            <div class="container">
                <h1>Predicții pentru Următoarea Extragere</h1>
                <p>Generată la: {now.strftime("%d.%m.%Y %H:%M")}</p>
            </div>
        </header>
        
        <div class="container">
            <div class="card summary">
                <h2>Sumar Rezultate</h2>
                <p>Analiza bazată pe datele istorice a generat următoarele recomandări:</p>
                
                <h3>Top 5 Numere Recomandate:</h3>
                <div class="top-numbers">
                    {', '.join([str(num) for num in top_numbers['top_5']])}
                </div>
                
                <h3>Top 4 Numere Recomandate:</h3>
                <div class="top-numbers">
                    {', '.join([str(num) for num in top_numbers['top_4']])}
                </div>
            </div>
            
            <div class="card">
                <h2>Vizualizare Interactivă a Scorurilor</h2>
                <div class="toggle-container">
                    <button class="toggle-btn active" id="allNumbersBtn">Toate Numerele</button>
                    <button class="toggle-btn" id="top20Btn">Top 20 Numere</button>
                    <button class="toggle-btn" id="top10Btn">Top 10 Numere</button>
                </div>
                
                <div class="chart-container">
                    <canvas id="scoresChart"></canvas>
                </div>
            </div>
            
            <div class="card">
                <h2>Top Numere Recomandate</h2>
                <div class="results-grid" id="resultsGrid">
                    <!-- Numerele vor fi generate prin JavaScript -->
                </div>
            </div>
            
            <div class="card recent-draws">
                <h2>Extrageri Recente Utilizate pentru Predicție</h2>
                <div class="table-container">
                    <table class="draws-table">
                        <thead>
                            <tr>
                                <th>Extragerea</th>
                                <th>Numere</th>
                            </tr>
                        </thead>
                        <tbody id="recentDrawsBody">
                            <!-- Conținut generat prin JavaScript -->
                        </tbody>
                    </table>
                </div>
            </div>
            
            <div class="card images-section">
                <h2>Grafice de Analiză</h2>
                
                <h3>Performanța Modelului în Timpul Antrenamentului</h3>
                <div class="img-container">
                    <img src="data:image/png;base64,{img_str_training}" alt="Grafic antrenament">
                </div>
                
                <h3>Distribuția Scorurilor pentru Toate Numerele</h3>
                <div class="img-container">
                    <img src="data:image/png;base64,{img_str_prediction}" alt="Grafic predicție">
                </div>
            </div>
        </div>
        
        <footer>
            <div class="container">
                <p>© {now.year} Predictor de Numere - Generat Automat</p>
            </div>
        </footer>
        
        <script>
            // Date pentru grafice și vizualizări
            const allScores = {scores_js};

            //const top5Numbers = {top_numbers['top_5']};
            const top5Numbers = [{', '.join(map(str, top_numbers['top_5']))}];
            
            // Sortăm scorurile pentru a găsi top 20 și top 10
            const indexedScores = allScores.map((score, index) => {{
                return {{ number: index + 1, score: score }};
            }});
            const sortedScores = [...indexedScores].sort((a, b) => b.score - a.score);
            const top20 = sortedScores.slice(0, 20);
            const top10 = sortedScores.slice(0, 10);
            
            // Crearea graficului
            const ctx = document.getElementById('scoresChart').getContext('2d');
            let currentChart = null;
            
            function createChart(data, highlightNumbers = []) {{
                // Dacă există un grafic, îl distrugem
                if (currentChart) {{
                    currentChart.destroy();
                }}
                
                // Pregătirea datelor pentru grafic
                const labels = data.map(item => item.number);
                const values = data.map(item => item.score);
                const backgroundColor = data.map(item => 
                    highlightNumbers.includes(item.number) ? '#db4437' : '#4285f4'
                );
                
                currentChart = new Chart(ctx, {{
                    type: 'bar',
                    data: {{
                        labels: labels,
                        datasets: [{{
                            label: 'Scor de Predicție',
                            data: values,
                            backgroundColor: backgroundColor,
                            borderColor: backgroundColor.map(color => color === '#db4437' ? '#c53929' : '#3367d6'),
                            borderWidth: 1
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {{
                            y: {{
                                beginAtZero: true,
                                grid: {{
                                    color: 'rgba(0, 0, 0, 0.1)'
                                }}
                            }},
                            x: {{
                                grid: {{
                                    display: false
                                }}
                            }}
                        }},
                        plugins: {{
                            legend: {{
                                display: false
                            }},
                            tooltip: {{
                                callbacks: {{
                                    title: function(tooltipItems) {{
                                        return 'Numărul ' + tooltipItems[0].label;
                                    }},
                                    label: function(context) {{
                                        return 'Scor: ' + context.raw.toFixed(4);
                                    }}
                                }}
                            }}
                        }}
                    }}
                }});
            }}
            
            // Generăm grid-ul cu numere
            function generateNumbersGrid(count = 20) {{
                const grid = document.getElementById('resultsGrid');
                grid.innerHTML = '';
                
                for (let i = 0; i < count; i++) {{
                    const item = sortedScores[i];
                    const isTopPick = top5Numbers.includes(item.number);
                    
                    const card = document.createElement('div');
                    card.className = `num-card ${{isTopPick ? 'top-pick' : ''}}`;
                    
                    const number = document.createElement('div');
                    number.className = 'number';
                    number.textContent = item.number;
                    
                    const score = document.createElement('div');
                    score.className = 'score';
                    score.textContent = `Scor: ${{item.score.toFixed(4)}}`;
                    
                    card.appendChild(number);
                    card.appendChild(score);
                    grid.appendChild(card);
                }}
            }}
            
            // Populăm tabelul cu extrageri recente
            function populateRecentDraws() {{
                const recentDrawsData = {recent_draws.to_json(orient='records')};
                const tbody = document.getElementById('recentDrawsBody');
                
                recentDrawsData.forEach((draw, index) => {{
                    const tr = document.createElement('tr');
                    
                    const tdIndex = document.createElement('td');
                    tdIndex.textContent = index + 1;
                    
                    const tdNumbers = document.createElement('td');
                    const numbers = Object.values(draw).filter(val => val !== null && !isNaN(val));
                    tdNumbers.textContent = numbers.join(', ');
                    
                    tr.appendChild(tdIndex);
                    tr.appendChild(tdNumbers);
                    tbody.appendChild(tr);
                }});
            }}
            
            // Atașăm evenimente pentru butoanele de toggle
            document.getElementById('allNumbersBtn').addEventListener('click', function() {{
                document.querySelectorAll('.toggle-btn').forEach(btn => btn.classList.remove('active'));
                this.classList.add('active');
                createChart(indexedScores, top5Numbers);
            }});
            
            document.getElementById('top20Btn').addEventListener('click', function() {{
                document.querySelectorAll('.toggle-btn').forEach(btn => btn.classList.remove('active'));
                this.classList.add('active');
                createChart(top20, top5Numbers);
            }});
            
            document.getElementById('top10Btn').addEventListener('click', function() {{
                document.querySelectorAll('.toggle-btn').forEach(btn => btn.classList.remove('active'));
                this.classList.add('active');
                createChart(top10, top5Numbers);
            }});
            
            // Comutare temă luminoasă/întunecată
            const themeToggle = document.getElementById('theme-toggle');
            const themeLabel = document.querySelector('.theme-switch label');
            
            themeToggle.addEventListener('change', function() {{
                if (this.checked) {{
                    document.body.classList.add('dark-theme');
                    themeLabel.textContent = '🌙';
                }} else {{
                    document.body.classList.remove('dark-theme');
                    themeLabel.textContent = '☀️';
                }}
                
                // Actualizăm graficul pentru a se potrivi cu tema
                if (currentChart) {{
                    const currentActive = document.querySelector('.toggle-btn.active');
                    if (currentActive.id === 'allNumbersBtn') {{
                        createChart(indexedScores, top5Numbers);
                    }} else if (currentActive.id === 'top20Btn') {{
                        createChart(top20, top5Numbers);
                    }} else {{
                        createChart(top10, top5Numbers);
                    }}
                }}
            }});
            
            // Inițializăm pagina
            window.addEventListener('DOMContentLoaded', () => {{
                // Inițializăm graficul cu toate numerele
                createChart(indexedScores, top5Numbers);
                
                // Generăm grid-ul cu rezultate
                generateNumbersGrid(20);
                
                // Populăm tabelul cu extrageri recente
                populateRecentDraws();
            }});
        </script>
    </body>
    </html>
    """
    
    # Salvăm HTML-ul în fișier
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"\nRezultatele au fost salvate în fișierul HTML: {filename}")
    return filename

# Funcția principală actualizată cu verificări suplimentare
def main(file_path):
    # Încărcăm datele
    print("\n=== ÎNCĂRCAREA DATELOR ===")
    data = load_and_prepare_data(file_path)
    if data is None:
        return
    
    print(f"Numărul total de înregistrări din set: {len(data)}")
    print(f"Numărul de coloane/numere per înregistrare: {data.shape[1]}")
    print(f"Primele 3000 rânduri din set:\n{data.head(3000)}") # Afișăm primele 3000 rânduri pentru verificare
    print(f"Ultimele 3000 rânduri din set:\n{data.tail(3000)}") # Afișăm ultimele 3000 rânduri pentru verificare
    
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
    
    # Salvăm rezultatele într-un fișier HTML interactiv
    html_file = save_results_to_html(
        {'top_5': top_5_numbers, 'top_4': top_4_numbers},
        final_score,
        history,
        recent_draws
    )
    
    return {
        'top_5': top_5_numbers.tolist(),
        'top_4': top_4_numbers.tolist(),
        'model': model,
        'html_file': html_file
    }



if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Utilizare: python script.py cale_fisier.csv")
        file_path = input("Introduceți calea către fișierul CSV: ")
    else:
        file_path = sys.argv[1]
    
    result = main(file_path)
    
    print("\nAnaliză completă. Verificați imaginile generate și fișierul HTML pentru detalii suplimentare.")