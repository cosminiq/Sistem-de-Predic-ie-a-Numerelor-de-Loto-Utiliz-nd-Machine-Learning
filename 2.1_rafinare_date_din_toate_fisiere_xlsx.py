import pandas as pd
import os
import glob
import numpy as np

def process_excel_files():
    # Caută toate fișierele .xlsx din directorul curent
    excel_files = glob.glob('*.xlsx')
    
    if not excel_files:
        print("Nu s-au găsit fișiere .xlsx în directorul curent!")
        return
    
    print(f"S-au găsit {len(excel_files)} fișiere .xlsx pentru procesare.")
    
    # Lista pentru a stoca toate rândurile de date procesate
    all_rows = []
    
    # Procesează fiecare fișier Excel
    for file in excel_files:
        try:
            print(f"Se procesează fișierul: {file}")
            
            # Citește fișierul Excel, sărind peste primele 3 rânduri
            df = pd.read_excel(file, skiprows=3)
            
            # Păstrează doar coloanele D-W (indexate 3-23)
            df_extract = df.iloc[:, 3:23]
            
            # Pentru fiecare rând, formatează datele în formatul dorit
            for index, row in df_extract.iterrows():
                # Convertește valorile la int și le unește cu virgule
                numbers = [str(int(x)) for x in row.values if not pd.isna(x)]
                formatted_row = ','.join(numbers)
                
                # Adaugă rândul formatat la lista
                if formatted_row:  # Verifică să nu fie rând gol
                    all_rows.append(formatted_row)
            
            print(f"  - Date extrase: {df_extract.shape[0]} rânduri")
            
        except Exception as e:
            print(f"Eroare la procesarea fișierului {file}: {str(e)}")
    
    # Verifică dacă avem date de procesat
    if not all_rows:
        print("Nu s-au putut extrage date din fișierele găsite!")
        return
    
    # Creează numele fișierului de output
    num_months = len(excel_files)
    output_file = f"date_rafinate_{num_months}_luni.csv"
    
    # Salvează rezultatul ca fișier text, cu un rând per linie
    with open(output_file, 'w') as f:
        f.write('\n'.join(all_rows))
    
    print(f"\nProcesare completă!")
    print(f"Total rânduri procesate: {len(all_rows)}")
    print(f"Fișierul a fost salvat ca: {output_file}")

if __name__ == "__main__":
    process_excel_files()