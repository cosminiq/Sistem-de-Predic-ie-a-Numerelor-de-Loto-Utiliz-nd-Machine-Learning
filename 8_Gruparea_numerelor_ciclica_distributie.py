import pandas as pd
import numpy as np
from collections import Counter
import random
from datetime import datetime, timedelta

def load_data(file_path):
    """
    Load and parse the CSV file containing the Keno drawings.
    Each row contains 20 numbers drawn.
    """
    try:
        # Read the raw CSV data
        with open(file_path, 'r') as f:
            lines = f.read().strip().split('\n')
        
        # Parse each line into a list of integers
        drawings = []
        for line in lines:
            nums = [int(n) for n in line.split(',')]
            if len(nums) == 20:  # Ensure we have 20 numbers per drawing
                drawings.append(nums)
        
        print(f"Loaded {len(drawings)} drawings")
        return drawings
    except Exception as e:
        print(f"Error loading data: {e}")
        return []

def approach_1_cyclical(drawings, num_to_generate=5, recency_weight=0.7):
    """
    Cyclical approach: Generate numbers based on how long it's been since they last appeared.
    Numbers that haven't appeared recently are given higher weight.
    
    Args:
        drawings: List of past drawings
        num_to_generate: Number of numbers to generate
        recency_weight: Weight given to recency (0-1)
    
    Returns:
        List of generated numbers
    """
    # Flatten all drawings to work with
    all_numbers = [num for drawing in drawings for num in drawing]
    
    # Create a dictionary of the last position each number appeared
    last_seen = {}
    for i, num in enumerate(all_numbers):
        last_seen[num] = i
    
    # Ensure all numbers from 1-80 are in the dictionary
    for i in range(1, 81):
        if i not in last_seen:
            last_seen[i] = -1000  # Hasn't appeared in the dataset
    
    # Calculate weights based on how long it's been since each number appeared
    weights = {}
    max_position = len(all_numbers) - 1
    for num in range(1, 81):
        if num in last_seen:
            # Normalize position to 0-1 range and invert (so higher means "longer ago")
            recency = 1 - (last_seen[num] / max_position) if max_position > 0 else 0.5
            weights[num] = recency_weight * recency + (1 - recency_weight) * 0.5  # Mix with some randomness
        else:
            weights[num] = 1.0  # Maximum weight for numbers never seen
    
    # Convert to list for random.choices
    nums = list(range(1, 81))
    weights_list = [weights[num] for num in nums]
    
    # Generate unique numbers
    selected = []
    while len(selected) < num_to_generate:
        choice = random.choices(nums, weights=weights_list, k=1)[0]
        if choice not in selected:
            selected.append(choice)
    
    return sorted(selected)

def approach_2_grouping(drawings, num_to_generate=5, group_strength=0.6):
    """
    Grouping approach: Analyze which numbers tend to appear together and use this
    to generate new combinations.
    
    Args:
        drawings: List of past drawings
        num_to_generate: Number of numbers to generate
        group_strength: How strongly to rely on common groupings (0-1)
    
    Returns:
        List of generated numbers
    """
    # Count how often each pair of numbers appears together
    pair_counts = Counter()
    for drawing in drawings:
        # Count all pairs in this drawing
        for i in range(len(drawing)):
            for j in range(i+1, len(drawing)):
                pair = tuple(sorted([drawing[i], drawing[j]]))
                pair_counts[pair] += 1
    
    # Start with a random number
    selected = [random.randint(1, 80)]
    
    # Add numbers that tend to appear with the already selected ones
    while len(selected) < num_to_generate:
        # Calculate scores for each candidate number
        scores = {}
        for num in range(1, 81):
            if num not in selected:
                score = 0
                for selected_num in selected:
                    pair = tuple(sorted([num, selected_num]))
                    score += pair_counts.get(pair, 0)
                scores[num] = score
        
        # Normalize scores to use as weights
        max_score = max(scores.values()) if scores else 1
        weights = {}
        for num in scores:
            # Mix between the group score and random selection
            normalized_score = scores[num] / max_score if max_score > 0 else 0
            weights[num] = group_strength * normalized_score + (1 - group_strength) * 0.5
        
        # Convert to list for random.choices
        candidates = list(weights.keys())
        weights_list = [weights[num] for num in candidates]
        
        # Choose the next number
        if candidates:
            next_num = random.choices(candidates, weights=weights_list, k=1)[0]
            selected.append(next_num)
        else:
            # Fallback if we somehow have no candidates
            while True:
                rand_num = random.randint(1, 80)
                if rand_num not in selected:
                    selected.append(rand_num)
                    break
    
    return sorted(selected)

def approach_3_distribution(drawings, num_to_generate=5):
    """
    Distribution approach: Generate numbers based on maintaining similar 
    distributions of odd/even and high/low as in past drawings.
    
    Args:
        drawings: List of past drawings
        num_to_generate: Number of numbers to generate
    
    Returns:
        List of generated numbers
    """
    # Analyze distributions in past drawings
    odd_counts = []
    low_counts = []  # Numbers 1-40
    
    for drawing in drawings:
        odd_count = sum(1 for num in drawing if num % 2 == 1)
        low_count = sum(1 for num in drawing if num <= 40)
        
        odd_counts.append(odd_count)
        low_counts.append(low_count)
    
    # Calculate the average proportions
    avg_odd_ratio = np.mean(odd_counts) / 20 if drawings else 0.5
    avg_low_ratio = np.mean(low_counts) / 20 if drawings else 0.5
    
    # Determine how many odd/even and low/high numbers to pick
    target_odd = round(num_to_generate * avg_odd_ratio)
    target_low = round(num_to_generate * avg_low_ratio)
    
    # Initialize pools of numbers
    odd_low = [n for n in range(1, 41) if n % 2 == 1]
    odd_high = [n for n in range(41, 81) if n % 2 == 1]
    even_low = [n for n in range(1, 41) if n % 2 == 0]
    even_high = [n for n in range(41, 81) if n % 2 == 0]
    
    # Shuffle each pool
    random.shuffle(odd_low)
    random.shuffle(odd_high)
    random.shuffle(even_low)
    random.shuffle(even_high)
    
    # Start building our selection
    selected = []
    
    # Add odd_low numbers
    count_odd_low = min(target_odd, target_low)
    selected.extend(odd_low[:count_odd_low])
    
    # Add odd_high numbers
    count_odd_high = target_odd - count_odd_low
    selected.extend(odd_high[:count_odd_high])
    
    # Add even_low numbers
    count_even_low = target_low - count_odd_low
    selected.extend(even_low[:count_even_low])
    
    # Add even_high numbers
    count_even_high = num_to_generate - len(selected)
    selected.extend(even_high[:count_even_high])
    
    # If we need more numbers (due to rounding), add them
    while len(selected) < num_to_generate:
        pool = list(set(range(1, 81)) - set(selected))
        selected.append(random.choice(pool))
    
    # If we have too many numbers (due to rounding), remove some
    if len(selected) > num_to_generate:
        selected = selected[:num_to_generate]
    
    return sorted(selected)

def get_strategy_recommendations(drawings):
    """
    Analyze the data and provide some strategic insights
    """
    if not drawings:
        return "No data available for analysis."
    
    # Flatten all drawings
    all_numbers = [num for drawing in drawings for num in drawing]
    
    # Count frequency of each number
    frequencies = Counter(all_numbers)
    
    # Find most/least frequent numbers
    most_common = frequencies.most_common(5)
    least_common = frequencies.most_common()[:-6:-1]
    
    # Analyze odd/even distribution
    odd_counts = [sum(1 for num in drawing if num % 2 == 1) for drawing in drawings]
    avg_odd = sum(odd_counts) / len(odd_counts)
    
    # Analyze high/low distribution
    low_counts = [sum(1 for num in drawing if num <= 40) for drawing in drawings]
    avg_low = sum(low_counts) / len(low_counts)
    
    # Calculate recency - when was each number last seen
    recent_draws = drawings[-50:]  # Look at last 50 draws
    last_seen = {}
    
    for i, drawing in enumerate(recent_draws):
        for num in drawing:
            if num not in last_seen:
                last_seen[num] = i
    
    # Find numbers not seen recently
    not_seen = [n for n in range(1, 81) if n not in last_seen]
    
    # Generate some recommendations
    recommendations = [
        f"Most frequent numbers: {', '.join(f'{n[0]} ({n[1]})' for n in most_common)}",
        f"Least frequent numbers: {', '.join(f'{n[0]} ({n[1]})' for n in least_common)}",
        f"Average odd numbers per draw: {avg_odd:.2f}/20 ({avg_odd/20*100:.1f}%)",
        f"Average low numbers (1-40) per draw: {avg_low:.2f}/20 ({avg_low/20*100:.1f}%)",
    ]
    
    if not_seen:
        recommendations.append(f"Numbers not seen in last 50 draws: {', '.join(map(str, not_seen))}")
    
    return "\n".join(recommendations)

def main(file_path, num_to_generate=5):
    """
    Main function to demonstrate all approaches
    """
    # Load data
    drawings = load_data(file_path)
    
    if not drawings:
        print("No data loaded. Please check the file path.")
        return
    
    # Print some strategic recommendations
    print("\nSTRATEGIC ANALYSIS:")
    print(get_strategy_recommendations(drawings))
    
    # Generate numbers using each approach
    print("\nGENERATED NUMBERS:")
    
    cyclical_numbers = approach_1_cyclical(drawings, num_to_generate)
    print(f"Approach 1 (Cyclical): {cyclical_numbers}")
    
    group_numbers = approach_2_grouping(drawings, num_to_generate)
    print(f"Approach 2 (Grouping): {group_numbers}")
    
    distribution_numbers = approach_3_distribution(drawings, num_to_generate)
    print(f"Approach 3 (Distribution): {distribution_numbers}")
    
    # Suggested play combining all approaches
    all_suggestions = cyclical_numbers + group_numbers + distribution_numbers
    counter = Counter(all_suggestions)
    final_suggestion = [num for num, count in counter.most_common() if count > 1]
    
    # If we need more numbers, add some from the highest counters
    while len(final_suggestion) < num_to_generate:
        remaining = [num for num, count in counter.most_common() if num not in final_suggestion]
        if remaining:
            final_suggestion.append(remaining[0])
        else:
            break
    
    # If we still need more, add randomly
    while len(final_suggestion) < num_to_generate:
        num = random.randint(1, 80)
        if num not in final_suggestion:
            final_suggestion.append(num)
    
    # If we somehow have too many, trim
    final_suggestion = sorted(final_suggestion[:num_to_generate])
    
    print(f"\nRECOMMENDED PLAY (Combined Approach): {final_suggestion}")

# Example usage
if __name__ == "__main__":
    # Replace with your actual file path
    file_path = "date_rafinate_4_luni.csv"
    main(file_path, num_to_generate=5)


"""
        Înțeleg că doriți să abordăm teoretic crearea unui generator de numere pentru loteria Keno, unde:
        - Se extrag 20 de numere din 80
        - Jucați 5 numere și câștigați dacă aveți 3, 4 sau 5 numere corecte
        - Avem date despre frecvențele numerelor (39 apare cel mai des cu 7511 apariții, iar 69 cel mai rar cu 7132 apariții)

        O abordare teoretică ar putea include următoarele considerente:

        1. **Distribuția frecvențelor**: Deși există diferențe în frecvențele numerelor (39 apare de 7511 ori, iar 69 de 7132 ori), diferența nu este foarte mare procentual. Într-un sistem cu adevărat aleatoriu, ne-am aștepta ca toate numerele să aibă aproximativ aceeași frecvență pe termen lung.

        2. **Strategii potențiale**:

        a) **Abordarea frecvențelor**: Puteți favoriza numerele cu frecvență mai mare, presupunând că tendința va continua.
        
        b) **Abordarea ciclică**: Puteți căuta modele ciclice - dacă unele numere nu au apărut de mult, ar putea avea o probabilitate mai mare să apară curând.
        
        c) **Gruparea numerelor**: Analizarea dacă anumite grupuri de numere tind să apară împreună.
        
        d) **Analiza distribuției**: Verificarea distribuției numerelor câștigătoare (câte sunt mici/mari, pare/impare).

        3. **Modele matematice**:

        - Puteți calcula probabilitatea de a nimeri k numere din 5 când se extrag 20 din 80 folosind distribuția hipergeometrică: P(X=k) = [C(20,k) × C(60,5-k)] / C(80,5)
        - Probabilitatea de a nimeri exact 3 numere: ~8.5%
        - Probabilitatea de a nimeri exact 4 numere: ~0.8%
        - Probabilitatea de a nimeri toate 5 numerele: ~0.03%

        4. **Pentru un generator**:
        - Puteți atribui o "pondere" fiecărui număr bazată pe frecvența sa istorică
        - Implementați un algoritm de selecție ponderată pentru a favoriza ușor numerele cu frecvență mai mare
        - Includeți opțiuni pentru păstrarea unui echilibru între numere pare/impare și numere mici/mari

        Un aspect important: deși analiza frecvențelor poate fi interesantă, în teoretic, extragerile Keno sunt independente și aleatorii. Frecvențele istorice nu garantează rezultate viitoare. Un generator eficient ar trebui să țină cont de acest lucru și să găsească un echilibru între utilizarea datelor istorice și menținerea unui grad sănătos de aleatoriu.

        Doriți să explorăm mai detaliat vreuna dintre aceste abordări?



        1. Abordarea Ciclică (approach_1_cyclical)

        Analizează cât de recent a apărut fiecare număr
        Acordă o pondere mai mare numerelor care nu au apărut de mult timp
        Folosește un parametru recency_weight pentru a echilibra între ciclicitate și aleatoriu

        2. Abordarea Grupării (approach_2_grouping)

        Identifică perechi de numere care tind să apară împreună
        Creează un sistem de scor bazat pe frecvența cu care numerele apar împreună
        Începe cu un număr aleatoriu și adaugă numere care tind să apară împreună cu cele deja selectate

        3. Abordarea Distribuției (approach_3_distribution)

        Analizează distribuția numerelor par/impar și mare/mic (1-40 vs 41-80)
        Generează numere care mențin aproximativ aceeași distribuție ca în tragerile anterioare

        Scriptul include și:

        O funcție de încărcare a datelor din fișierul CSV cu tragerile
        O analiză strategică care oferă recomandări bazate pe datele istorice
        O combinare a celor trei abordări pentru o recomandare finală

        Pentru a utiliza scriptul, trebuie doar să modificați variabila file_path din funcția main cu calea către fișierul CSV cu tragerile istorice.
        Vă recomand să experimentați cu parametrii fiecărei abordări pentru a vedea care oferă cele mai bune rezultate pentru situația dumneavoastră specifică. Doriți să discutăm mai în detaliu despre vreuna dintre aceste abordări sau aveți întrebări despre implementare?
"""