# Readme - Sistem de Predicție a Numerelor de Loto Utilizând Machine Learning

Acest proiect implementează un sistem complex pentru analiza și predicția numerelor de loto (specifc KINO) utilizând tehnici de Machine Learning și procesare de date. Scopul principal este de a extrage date istorice, a le rafina, analiza frecvențele și de a genera predicții pentru extragerile viitoare. Proiectul este structurat în mai multe scripturi Python, fiecare având un rol specific în fluxul de lucru. Mai jos este prezentată o descriere detaliată a proceselor, cu accent pe aspectele tehnice și funcționalitățile implementate.

## Structura Proiectului

Proiectul conține mai multe scripturi Python care acoperă următoarele etape:
1. **Descărcarea datelor** (`1.xls_scraper_versiunea2.py`)
2. **Rafinarea datelor** (`2.1_rafinare_date_din_toate_fisiere_xlsx.py`, `2.rafinare_date_dintr_un_singur_fisier_xlsx.py`)
3. **Analiza frecvențelor** (`4.frecventa_aparitii_numerelor_fara_raport.py`, `4_2_frecventa_aparitii_numerelor_raport_html.py`)
4. **Predicții utilizând Machine Learning** (`3_predictii_ML_fara_raport.py`, `3_1_predictii_ML_cu_raport_HTML.py`, `4_3.Gradient Boosting Machines XGBoost, LightGBM, CatBoost.py`)
5. **Calcul probabilistic** (`5_Calcul_statistic_combinatoric_Hypergeometric_distribution copy.py`)
6. **Analiza profitului/pierderii** (`6_Analiza_profit_pierdere_statistica_betano_XLSX.py`)

## Descrierea Tehnică a Proceselor

### 1. Descărcarea Datelor (`1.xls_scraper_versiunea2.py`)
Acest script descarcă fișiere Excel (.xlsx) cu date istorice de la o sursă online și le salvează local.

- **Funcționalități**:
  - Construiește URL-uri dinamice pentru fișierele Excel folosind un șablon (`link`) și intervale de ani/luni definite.
  - Utilizează biblioteca `requests` pentru a descărca fișierele în mod streaming.
  - Salvează URL-urile într-un fișier `link.txt` pentru referință.
  - Gestionează erorile HTTP și alte excepții, afișând mesaje descriptive.
- **Tehnologii**: `requests`
- **Ieșire**: Fișiere `.xlsx` (ex. `kino_2025_04.xlsx`) și `link.txt`.

### 2. Rafinarea Datelor
#### 2.1 Procesarea Multiplelor Fișiere (`2.1_rafinare_date_din_toate_fisiere_xlsx.py`)
Acest script procesează toate fișierele `.xlsx` din directorul curent, extrăgând și rafinând datele relevante.

- **Funcționalități**:
  - Utilizează `glob` pentru a identifica toate fișierele `.xlsx`.
  - Citește fișierele cu `pandas`, sărind peste primele 3 rânduri (metadate) și extrăgând coloanele D-W (index 3-23).
  - Convertește valorile numerice în format întreg, eliminând valorile `NaN`.
  - Formatează fiecare rând ca un șir de numere separate prin virgulă.
  - Salvează rezultatul într-un fișier CSV (ex. `date_rafinate_X_luni.csv`), cu un rând per linie.
  - Include gestionarea erorilor pentru fișiere corupte sau formate incorecte.
- **Tehnologii**: `pandas`, `numpy`, `glob`, `os`
- **Ieșire**: Fișier CSV cu date rafinate.

#### 2.2 Procesarea unui Singur Fișier (`2.rafinare_date_dintr_un_singur_fisier_xlsx.py`)
Acest script procesează un singur fișier `.xlsx` specificat.

- **Funcționalități**:
  - Citește fișierul Excel, sărind peste primele 3 rânduri și extrăgând coloanele D-W.
  - Salvează direct datele extrase într-un fișier CSV (`output.csv`) fără alte transformări.
- **Tehnologii**: `pandas`
- **Ieșire**: Fișier CSV (`output.csv`).

### 3. Analiza Frecvențelor
#### 3.1 Fără Raport (`4.frecventa_aparitii_numerelor_fara_raport.py`)
Acest script analizează frecvența aparițiilor numerelor în fișierele `.xlsx`.

- **Funcționalități**:
  - Utilizează procesare paralelă cu `multiprocessing` pentru a accelera analiza mai multor fișiere.
  - Citește coloanele D-W din fiecare fișier Excel și numără aparițiile fiecărui număr (1-80).
  - Ignoră valorile non-numerice și gestionează erorile.
  - Generează un grafic simplu al frecvențelor folosind `matplotlib` (afisat interactiv).
  - Afișează numerele cu frecvența maximă și minimă.
- **Tehnologii**: `pandas`, `numpy`, `matplotlib`, `multiprocessing`
- **Ieșire**: Grafic afișat interactiv și statistici în consolă.

#### 3.2 Cu Raport HTML (`4_2_frecventa_aparitii_numerelor_raport_html.py`)
Acest script extinde analiza frecvențelor, generând un raport HTML interactiv.

- **Funcționalități**:
  - Similar cu scriptul anterior, procesează fișierele `.xlsx` în paralel.
  - Generează un grafic al frecvențelor salvat ca `frequency_plot.png`.
  - Creează un raport HTML cu:
    - Rezumat al fișierelor și extragerilor procesate.
    - Tabel sortabil cu frecvențele numerelor.
    - Grafic al distribuției frecvențelor.
    - Evidențierea numerelor cu frecvență maximă și minimă.
  - Include JavaScript pentru sortarea interactivă a tabelului.
- **Tehnologii**: `pandas`, `numpy`, `matplotlib`, `multiprocessing`
- **Ieșire**: Fișier HTML (`4_2_frecventa_aparitii_numerelor_raport_html.html`) și `frequency_plot.png`.

### 4. Predicții Utilizând Machine Learning
#### 4.1 Model Neural (TensorFlow) Fără Raport (`3_predictii_ML_fara_raport.py`)
Acest script utilizează un model de rețea neurală pentru a prezice numerele viitoare.

- **Funcționalități**:
  - **Încărcarea datelor**: Citește un fișier CSV cu date rafinate, separând numerele dacă sunt în format de șir.
  - **Crearea caracteristicilor**: Transformă extragerile într-o matrice binară (80 de coloane), unde `1` indică prezența unui număr.
  - **Modelul**: O rețea neurală secvențială (`Sequential`) cu straturi dense (`Dense`), activări ReLU, dropout pentru regularizare și ieșire sigmoid.
  - **Antrenament**: Folosește `adam` ca optimizator și `binary_crossentropy` ca funcție de pierdere; antrenează pe 80% din date, validând pe 20%.
  - **Predicție**: Combină predicțiile modelului (70%) cu frecvențele recente (30%) pentru a calcula un scor final.
  - **Vizualizări**: Generează grafice pentru pierderea/acuratețea antrenamentului (`training_history.png`) și distribuția scorurilor (`prediction_scores.png`).
  - **Rezultate**: Afișează top 5 și top 4 numere recomandate, împreună cu scorurile lor.
- **Tehnologii**: `tensorflow`, `pandas`, `numpy`, `matplotlib`, `scikit-learn`
- **Ieșire**: Grafice (`training_history.png`, `prediction_scores.png`), statistici în consolă.

#### 4.2 Model Neural (TensorFlow) Cu Raport HTML (`3_1_predictii_ML_cu_raport_HTML.py`)
Acest script extinde modelul anterior, adăugând un raport HTML interactiv.

- **Funcționalități**:
  - Similar cu scriptul anterior pentru procesarea datelor, model și predicții.
  - Generează un raport HTML detaliat cu:
    - Rezumat al predicțiilor (top 5 și top 4 numere).
    - Grafic interactiv al scorurilor (folosind `Chart.js`), cu opțiuni pentru afișarea tuturor numerelor, top 20 sau top 10.
    - Tabel cu extragerile recente utilizate.
    - Grafice statice pentru performanța modelului și distribuția scorurilor (codificate în base64).
    - Temă luminoasă/întunecată comutabilă.
  - Include stiluri CSS moderne și animații pentru o experiență îmbunătățită.
- **Tehnologii**: `tensorflow`, `pandas`, `numpy`, `matplotlib`, `base64`, `Chart.js`
- **Ieșire**: Fișier HTML (ex. `rezultat_YYYYMMDD_HHMM.html`), grafice (`training_history.png`, `prediction_scores.png`).

#### 4.3 Gradient Boosting (XGBoost) (`4_3.Gradient Boosting Machines XGBoost, LightGBM, CatBoost.py`)
Acest script utilizează XGBoost pentru predicții mai robuste.

- **Funcționalități**:
  - **Încărcarea și pregătirea datelor**: Similar cu scripturile TensorFlow.
  - **Modelul**: Utilizează `xgboost` cu parametri optimizați (`eta`, `max_depth`, `subsample`, etc.) și format `DMatrix`.
  - **Antrenament**: Include oprire timpurie (`early_stopping_rounds`) pentru a preveni supra-învățarea.
  - **Predicție**: Combină predicțiile modelului (70%) cu frecvențele recente (30%).
  - **Vizualizări**:
    - Grafic al importanței numerelor (`feature_importance.png`).
    - Grafice comparative pentru scorul final, predicția modelului și frecvențele (`prediction_components.png`).
    - Distribuția scorurilor și top numere (`score_distribution.png`).
    - Heatmap al scorurilor în format 8x10 (`score_heatmap.png`).
  - **Evaluare**: Calculează acuratețea și pierderea (`log_loss`) pe setul de validare.
- **Tehnologii**: `xgboost`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`
- **Ieșire**: Grafice multiple, statistici în consolă.

### 5. Calcul Probabilistic (`5_Calcul_statistic_combinatoric_Hypergeometric_distribution copy.py`)
Acest script calculează probabilitatea de a ghici un anumit număr de numere corecte folosind distribuția hipergeometrică.

- **Funcționalități**:
  - Definește parametrii: `N=80` (total numere), `K=5` (numere alese), `n=20` (numere extrase), `k=3` (numere corecte).
  - Calculează probabilitatea folosind formula distribuției hipergeometrice.
- **Tehnologii**: `math.comb`
- **Ieșire**: Probabilitate afișată în consolă (ex. `0.0214 (~2.14%)`).

### 6. Analiza Profitului/Pierderii (`6_Analiza_profit_pierdere_statistica_betano_XLSX.py`)
Acest script analizează tranzacțiile financiare dintr-un fișier Excel pentru a evalua performanța pariurilor.

- **Funcționalități**:
  - Citește coloana D dintr-un fișier Excel, ignorând primele 2 rânduri.
  - Curăță datele (elimină „lei”, înlocuiește „,” cu „.”, elimină spații) și convertește în valori numerice.
  - Calculează:
    - Numărul total de pariuri.
    - Pariurile câștigătoare/pierdute (procente).
    - Profitul total.
- **Tehnologii**: `pandas`
- **Ieșire**: Statistici afișate în consolă (ex. „Profit total: 150.50 lei”).

## Fluxul de Lucru

1. **Descărcare**: Rulează `1.xls_scraper_versiunea2.py` pentru a descărca fișierele `.xlsx` cu date istorice.
2. **Rafinare**:
   - Pentru toate fișierele: Rulează `2.1_rafinare_date_din_toate_fisiere_xlsx.py` pentru a genera un fișier CSV consolidat.
   - Pentru un singur fișier: Rulează `2.rafinare_date_dintr_un_singur_fisier_xlsx.py`.
3. **Analiza frecvențelor**:
   - Pentru statistici rapide: `4.frecventa_aparitii_numerelor_fara_raport.py`.
   - Pentru raport detaliat: `4_2_frecventa_aparitii_numerelor_raport_html.py`.
4. **Predicții**:
   - Pentru model neural simplu: `3_predictii_ML_fara_raport.py`.
   - Pentru model neural cu raport HTML: `3_1_predictii_ML_cu_raport_HTML.py`.
   - Pentru model XGBoost: `4_3.Gradient Boosting Machines XGBoost, LightGBM, CatBoost.py`.
5. **Analiza probabilistică**: Rulează `5_Calcul_statistic_combinatoric_Hypergeometric_distribution copy.py` pentru a înțelege șansele teoretice.
6. **Evaluarea financiară**: Rulează `6_Analiza_profit_pierdere_statistica_betano_XLSX.py` pentru a analiza performanța pariurilor.

## Cerințe Tehnice

- **Dependențe**:
  - Python 3.8+
  - Biblioteci: `pandas`, `numpy`, `matplotlib`, `tensorflow`, `xgboost`, `scikit-learn`, `seaborn`, `requests`, `openpyxl`
- **Sistem de operare**: Compatibil cu Windows, macOS, Linux.
- **Resurse**: Procesare paralelă (`multiprocessing`) și modelele ML necesită memorie și procesor adecvate (minimum 8GB RAM recomandat).

## Instalare

1. Clonează repository-ul:
   ```bash
   git clone <URL_REPOSITORY>
   cd <NUME_REPOSITORY>
   ```
2. Instalează dependențele:
   ```bash
   pip install -r requirements.txt
   ```
   (Creează un fișier `requirements.txt` cu bibliotecile enumerate mai sus.)
3. Descarcă datele utilizând `1.xls_scraper_versiunea2.py`.

## Utilizare

1. Asigură-te că fișierele `.xlsx` sunt în directorul curent sau specifică calea către fișierul CSV pentru predicții.
2. Rulează scripturile în ordinea descrisă în „Fluxul de lucru”.
3. Pentru predicții, furnizează calea către fișierul CSV rafinat:
   ```bash
   python 3_1_predictii_ML_cu_raport_HTML.py date_rafinate_X_luni.csv
   ```

## Limitări și Îmbunătățiri Viitoare

- **Limitări**:
  - Predicțiile sunt bazate pe date istorice și nu garantează succesul (loto este un joc aleator).
  - Performanța modelelor depinde de calitatea și volumul datelor.
  - Procesarea paralelă poate fi limitată de resursele hardware.
- **Îmbunătățiri**:
  - Adăugarea altor algoritmi ML (ex. LightGBM, CatBoost).
  - Optimizarea hiperparametrilor modelelor folosind `GridSearchCV` sau `Optuna`.
  - Integrarea într-o aplicație web pentru utilizare mai accesibilă.
  - Analiza suplimentară a tiparelor temporale în extrageri.

## Concluzie

Acest sistem oferă o soluție completă pentru analiza și predicția numerelor de loto, combinând procesarea datelor, analiza statistică și Machine Learning. Rapoartele interactive și vizualizările detaliate facilitează înțelegerea rezultatelor, iar utilizarea modelelor avansate (TensorFlow, XGBoost) sporește robustețea predicțiilor. Proiectul poate fi extins pentru alte jocuri de loto sau aplicații similare de predicție.