import requests as req

link = 'https://media.opap.gr/Excel_xlsx/1100/{}/kino_{}_{}.xlsx'

yearList = list(map('{:02}'.format, range(2025, 2026)))
monthList = list(map('{:02}'.format, range(4, 5)))

# Deschide fișierul 'link.txt' în modul 'write' pentru a salva URL-urile
with open('link.txt', 'w', encoding='utf-8') as link_file:
    for year in yearList:
        for month in monthList:
            current_link = link.format(year, year, month)
            
            # Scrie link-ul în fișierul link.txt
            link_file.write(current_link + '\n')
            
            # Afișează link-ul în consolă (opțional)
            print(f"URL: {current_link}")
            
            # Încearcă descărcarea
            try:
                response = req.get(current_link, stream=True)
                if response.ok:
                    filename = f'kino_{year}_{month}.xlsx'
                    with open(filename, 'wb') as f:
                        f.write(response.content)
                    print(f"✓ Descărcat: {filename}")
                else:
                    print(f"✗ Eroare: Link invalid (HTTP {response.status_code})")
            except Exception as e:
                print(f"✗ Eroare la descărcare: {str(e)}")

print("Proces finalizat. Toate link-urile au fost salvate în 'link.txt'.")