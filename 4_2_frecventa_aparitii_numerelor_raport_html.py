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
    return [f for f in os.listdir() if f.endswith(".xlsx")]

def plotVals(data, filename="frequency_plot.png"):
    """Generează și salvează un grafic al frecvențelor."""
    plt.figure(figsize=(12, 6))
    nums, freqs = zip(*data)
    plt.plot(nums, freqs, '-o', markersize=4, color='blue')
    plt.xlabel("Număr (1-80)")
    plt.ylabel("Frecvență")
    plt.title("Distribuția numerelor KINO")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()

def merge(list1, list2):
    """Combină două liste într-o listă de tupluri."""
    return [(list1[i], list2[i]) for i in range(len(list1))]

def accumulateFile(filename):
    """Numără frecvențele numerelor într-un fișier Excel."""
    try:
        vals = np.array(range(1, 81))
        counts = np.zeros(80, dtype=int)

        df = pd.read_excel(filename, engine='openpyxl')
        df = df.iloc[:, 3:23]  # Coloanele D–W

        c = 0
        for col in df:
            for val in df[col]:
                try:
                    val_int = int(val)
                    if 1 <= val_int <= 80:
                        counts[val_int - 1] += 1
                        c += 1
                except (ValueError, TypeError):
                    continue

        merged = merge(vals, counts)
        merged.append(c // 20)  # Număr extrageri
        return merged
    except Exception as e:
        print(f"Eroare la procesarea {filename}: {str(e)}")
        return None

def generate_html_report(data, raffle_count, processed_files, max_freq, min_freq):
    """Generează un raport HTML detaliat."""
    plot_file = "frequency_plot.png"
    plotVals(data, plot_file)

    html_content = f"""
    <!DOCTYPE html>
    <html lang="ro">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Raport Frecvență Numere KINO</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f4f4f9;
                color: #333;
            }}
            .container {{
                max-width: 1000px;
                margin: auto;
                background: #fff;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }}
            h1 {{
                text-align: center;
                color: #2c3e50;
            }}
            h2 {{
                color: #34495e;
                margin-top: 20px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #3498db;
                color: white;
            }}
            tr:hover {{
                background-color: #f1f1f1;
            }}
            img {{
                max-width: 100%;
                height: auto;
                display: block;
                margin: 20px auto;
                border: 1px solid #ddd;
                border-radius: 4px;
            }}
            .highlight-max {{
                color: #27ae60;
                font-weight: bold;
            }}
            .highlight-min {{
                color: #c0392b;
                font-weight: bold;
            }}
            @media (max-width: 600px) {{
                .container {{
                    padding: 10px;
                }}
                th, td {{
                    padding: 8px;
                    font-size: 14px;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Raport Frecvență Numere KINO</h1>
            <h2>Rezumat General</h2>
            <p><strong>Fișiere procesate:</strong> {processed_files}</p>
            <p><strong>Extrageri procesate:</strong> {raffle_count}</p>
            <h2>Analiză Frecvențe</h2>
            <p><strong>Numărul cu cea mai mare frecvență:</strong> Numărul <span class="highlight-max">{max_freq[0]}</span> cu <span class="highlight-max">{max_freq[1]}</span> apariții</p>
            <p><strong>Numărul cu cea mai mică frecvență:</strong> Numărul <span class="highlight-min">{min_freq[0]}</span> cu <span class="highlight-min">{min_freq[1]}</span> apariții</p>
            <h2>Distribuția Frecvențelor</h2>
            <table id="frequencyTable">
                <thead>
                    <tr>
                        <th>Număr</th>
                        <th>Frecvență</th>
                    </tr>
                </thead>
                <tbody>
    """

    for num, freq in data:
        html_content += f"""
                    <tr>
                        <td>{num}</td>
                        <td>{freq}</td>
                    </tr>
        """

    html_content += f"""
                </tbody>
            </table>
            <h2>Grafic Frecvențe</h2>
            <img src="{plot_file}" alt="Grafic Frecvențe Numere KINO">
        </div>
        <script>
            // Adaugă interactivitate pentru sortarea tabelului
            function sortTable(n) {{
                let table = document.getElementById("frequencyTable");
                let rows, switching = true;
                let i, shouldSwitch, dir = "asc", switchcount = 0;
                while (switching) {{
                    switching = false;
                    rows = table.rows;
                    for (i = 1; i < (rows.length - 1); i++) {{
                        shouldSwitch = false;
                        let x = rows[i].getElementsByTagName("TD")[n];
                        let y = rows[i + 1].getElementsByTagName("TD")[n];
                        let xVal = isNaN(parseInt(x.innerHTML)) ? x.innerHTML.toLowerCase() : parseInt(x.innerHTML);
                        let yVal = isNaN(parseInt(y.innerHTML)) ? y.innerHTML.toLowerCase() : parseInt(y.innerHTML);
                        if (dir == "asc") {{
                            if (xVal > yVal) {{
                                shouldSwitch = true;
                                break;
                            }}
                        }} else if (dir == "desc") {{
                            if (xVal < yVal) {{
                                shouldSwitch = true;
                                break;
                            }}
                        }}
                    }}
                    if (shouldSwitch) {{
                        rows[i].parentNode.insertBefore(rows[i + 1], rows[i]);
                        switching = true;
                        switchcount++;
                    }} else {{
                        if (switchcount == 0 && dir == "asc") {{
                            dir = "desc";
                            switching = true;
                        }}
                    }}
                }}
            }}

            // Adaugă evenimente de click pe anteturile tabelului
            document.querySelectorAll('#frequencyTable th').forEach((th, index) => {{
                th.addEventListener('click', () => sortTable(index));
                th.style.cursor = 'pointer';
            }});
        </script>
    </body>
    </html>
    """

    with open("4_2_frecventa_aparitii_numerelor_raport_html.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    print("Raport generat: 4_2_frecventa_aparitii_numerelor_raport_html.html")

if __name__ == '__main__':
    vals = np.array(range(1, 81), dtype=np.int64)
    counts = np.zeros(80, dtype=int)

    raffle_count = 0
    final = merge(vals, counts)

    filenames = fetchFilenames()
    if not filenames:
        print("Nu s-au găsit fișiere .xlsx în director.")
        exit(1)

    with mp.Pool() as pool:
        results = pool.map(accumulateFile, filenames)
        for result in results:
            if result is None:
                continue
            raffle_count += result[-1]
            for i in range(len(result) - 1):
                final[i] = (final[i][0], final[i][1] + result[i][1])

    processed_files = len([r for r in results if r is not None])
    print(f'Fișiere procesate: {processed_files}')
    print(f'Extrageri procesate: {raffle_count}')

    # Calculăm frecvența maximă și minimă dintre toate numerele (1-80)
    max_freq = max(final, key=itemgetter(1))
    min_freq = min(final, key=itemgetter(1))

    print(f'Numărul cu cea mai mare frecvență: {max_freq[0]} cu {max_freq[1]} apariții')
    print(f'Numărul cu cea mai mică frecvență: {min_freq[0]} cu {min_freq[1]} apariții')

    # Generăm raportul HTML
    generate_html_report(final, raffle_count, processed_files, max_freq, min_freq)