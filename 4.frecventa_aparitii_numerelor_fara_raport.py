from operator import itemgetter
import matplotlib.pyplot as plt
import multiprocessing as mp
import pandas as pd
import numpy as np
import os
import warnings

# Suprimăm avertismentul openpyxl
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl.styles.stylesheet")

def fetchFilenames():
    """Returnează doar fișierele .xlsx din directorul curent."""
    l = []
    for filename in os.listdir():
        if filename.endswith(".xlsx"):  # Verificăm extensia .xlsx
            l.append(filename)
    return l

def printMinMax(data):
    """Afișează numărul cu frecvența maximă și minimă."""
    _max = max(data, key=itemgetter(1))
    print('num {} is max with: {}'.format(_max[0], _max[1]))
    _min = min(data, key=itemgetter(1))
    print('num {} is min with: {}'.format(_min[0], _min[1]))

def plotVals(data):
    """Generează un grafic al frecvențelor."""
    plt.xticks(range(1, 81))
    plt.plot(*zip(*data))
    plt.plot(*zip(*data), 'or')
    plt.xlabel("Număr (1-80)")
    plt.ylabel("Frecvență")
    plt.title("Distribuția numerelor KINO")
    plt.show()

def merge(list1, list2):
    """Combină două liste într-o listă de tupluri."""
    merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))]
    return merged_list

def accumulateFile(filename):
    """Numără frecvențele numerelor într-un fișier Excel."""
    try:
        vals = np.array([*range(1, 81)])
        counts = np.array([0] * 80)  # Corectăm la 80, nu 81

        df = pd.read_excel(filename, engine='openpyxl')
        df = df.iloc[:, 3:23]  # Coloanele D–W

        c = 0
        for col in df:
            for val in df[col]:
                try:
                    val_int = int(val)
                    if 1 <= val_int <= 80:  # Verificăm intervalul
                        counts[val_int - 1] += 1
                        c += 1
                except (ValueError, TypeError):
                    continue  # Ignorăm valorile non-numerice

        merged = merge(vals, counts)
        # Estimăm extragerile corect: c / 20 (20 numere per extracție)
        merged.append(c // 20)
        return merged
    except Exception as e:
        print(f"Eroare la procesarea {filename}: {str(e)}")
        return None  # Returnăm None pentru fișierele invalide

if __name__ == '__main__':
    vals = np.array([*range(1, 81)], dtype=np.int64)
    counts = np.array([0] * 80)  # Corectăm la 80

    raffleCount = 0
    final = merge(vals, counts)

    filenames = fetchFilenames()
    if not filenames:
        print("Nu s-au găsit fișiere .xlsx în director.")
        exit(1)

    with mp.Pool() as pool:
        results = pool.map(accumulateFile, filenames)
        for result in results:
            if result is None:
                continue  # Ignorăm fișierele cu erori
            raffleCount += result[-1]  # Numărul de extrageri
            for i in range(len(result) - 1):
                final[i] = (final[i][0], final[i][1] + result[i][1])

    print('Fișiere procesate: {}'.format(len([r for r in results if r is not None])))
    print('Extrageri procesate: {}'.format(raffleCount))
    printMinMax(final)
    plotVals(final)