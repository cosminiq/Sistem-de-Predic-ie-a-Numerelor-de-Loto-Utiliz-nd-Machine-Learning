import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import random
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import base64
from io import BytesIO
import seaborn as sns
import json
from datetime import datetime
import math


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
    
    # Return both the selected numbers and the weights for reporting
    return sorted(selected), weights

def approach_2_grouping(drawings, num_to_generate=5, group_strength=0.6):
    """
    Grouping approach: Analyze which numbers tend to appear together and use this
    to generate new combinations.
    
    Args:
        drawings: List of past drawings
        num_to_generate: Number of numbers to generate
        group_strength: How strongly to rely on common groupings (0-1)
    
    Returns:
        List of generated numbers and pair_counts for visualization
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
    
    # Convert pair_counts to a format suitable for visualization
    pair_data = []
    for (num1, num2), count in pair_counts.most_common(50):  # Top 50 pairs
        pair_data.append({
            'num1': num1,
            'num2': num2,
            'count': count
        })
    
    return sorted(selected), pair_data

def approach_3_distribution(drawings, num_to_generate=5):
    """
    Distribution approach: Generate numbers based on maintaining similar 
    distributions of odd/even and high/low as in past drawings.
    
    Args:
        drawings: List of past drawings
        num_to_generate: Number of numbers to generate
    
    Returns:
        List of generated numbers and distribution data for visualization
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
    
    # Prepare distribution data for visualization
    distribution_data = {
        'odd_even': {
            'labels': ['Impare', 'Pare'],
            'counts': [np.mean(odd_counts), 20 - np.mean(odd_counts)],
            'percentages': [np.mean(odd_counts)/20*100, (20-np.mean(odd_counts))/20*100]
        },
        'low_high': {
            'labels': ['Mici (1-40)', 'Mari (41-80)'],
            'counts': [np.mean(low_counts), 20 - np.mean(low_counts)],
            'percentages': [np.mean(low_counts)/20*100, (20-np.mean(low_counts))/20*100]
        },
        'odds_by_draw': odd_counts[-50:],  # Last 50 draws
        'lows_by_draw': low_counts[-50:]   # Last 50 draws
    }
    
    return sorted(selected), distribution_data

def get_strategy_recommendations(drawings):
    """
    Analyze the data and provide some strategic insights
    
    Returns:
        Dictionary with recommendation data and metrics
    """
    if not drawings:
        return {"error": "No data available for analysis."}
    
    # Flatten all drawings
    all_numbers = [num for drawing in drawings for num in drawing]
    
    # Count frequency of each number
    frequencies = Counter(all_numbers)
    
    # Find most/least frequent numbers
    most_common = frequencies.most_common(10)
    least_common = frequencies.most_common()[:-11:-1]
    
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
    
    # Prepare frequency data for all numbers
    frequency_data = []
    for num in range(1, 81):
        frequency_data.append({
            'number': num,
            'frequency': frequencies.get(num, 0),
            'is_odd': num % 2 == 1,
            'is_low': num <= 40,
            'last_seen': len(recent_draws) - last_seen.get(num, -1) if num in last_seen else "50+"
        })
    
    # Calculate probabilities for hitting k out of 5 numbers
    probs = []
    for k in range(6):  # 0 to 5 matches
        prob = (math.comb(20, k) * math.comb(60, 5-k)) / math.comb(80, 5)
        probs.append({
            'matches': k,
            'probability': prob,
            'percentage': prob * 100,
            'odds': f"1 în {int(1/prob) if prob > 0 else 'infinit'}"
        })
    
    # Return a structured dictionary with all the analysis
    return {
        'most_common': most_common,
        'least_common': least_common,
        'odd_even': {
            'avg_odd': avg_odd,
            'avg_even': 20 - avg_odd,
            'odd_percentage': avg_odd / 20 * 100,
            'even_percentage': (20 - avg_odd) / 20 * 100
        },
        'low_high': {
            'avg_low': avg_low,
            'avg_high': 20 - avg_low,
            'low_percentage': avg_low / 20 * 100,
            'high_percentage': (20 - avg_low) / 20 * 100
        },
        'not_seen_recently': not_seen,
        'frequency_data': frequency_data,
        'total_drawings': len(drawings),
        'probability_table': probs
    }

def generate_html_report(data, file_path="8_Gruparea_numerelor_ciclica_distributie.html"):
    """
    Generate an HTML report with all the analysis and charts
    """
    # Create directory if it doesn't exist
    dir_name = "8_Gruparea_numerelor_ciclica_distributie"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    # Full path for the file
    full_path = os.path.join(dir_name, file_path)
    
    # Generate HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html lang="ro">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Analiză Keno: Gruparea numerelor, ciclică și distribuție</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/PapaParse/5.3.2/papaparse.min.js"></script>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f8f9fa;
            }}
            h1, h2, h3, h4 {{
                color: #2c3e50;
            }}
            h1 {{
                text-align: center;
                margin-bottom: 30px;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
            }}
            .section {{
                background-color: white;
                padding: 20px;
                margin-bottom: 25px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .chart-container {{
                margin: 20px 0;
            }}
            .recommendations {{
                background-color: #e8f4f8;
                padding: 20px;
                border-left: 4px solid #3498db;
                margin: 20px 0;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                padding: 12px 15px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #3498db;
                color: white;
            }}
            tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
            .highlight {{
                font-weight: bold;
                color: #e74c3c;
            }}
            .grid-container {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(70px, 1fr));
                gap: 10px;
                margin: 20px 0;
            }}
            .number-box {{
                padding: 5px;
                text-align: center;
                border-radius: 4px;
                font-weight: bold;
                display: flex;
                align-items: center;
                justify-content: center;
                height: 40px;
            }}
            .odd {{
                background-color: #FF6B6B;
                color: white;
            }}
            .even {{
                background-color: #4ECDC4;
                color: white;
            }}
            .footer {{
                text-align: center;
                margin-top: 40px;
                padding: 20px;
                font-size: 0.8em;
                color: #7f8c8d;
            }}
            .approach-result {{
                padding: 15px;
                background-color: #f8f9fa;
                border-left: 4px solid #9b59b6;
                margin: 15px 0;
            }}
            .final-recommendation {{
                padding: 15px;
                background-color: #e8f8e8;
                border-left: 4px solid #27ae60;
                margin: 15px 0;
                font-weight: bold;
            }}
        </style>
    </head>
    <body>
        <h1>Analiză Keno: Gruparea numerelor, ciclică și distribuție</h1>
        
        <div class="section">
            <h2>Sumar</h2>
            <p>Acest raport prezintă o analiză detaliată a {data['total_drawings']} extrageri Keno și furnizează recomandări bazate pe trei abordări diferite: ciclică, gruparea numerelor și distribuția.</p>
            
            <div class="recommendations">
                <h3>Recomandări de joc</h3>
                <p>Abordarea ciclică: <strong>{', '.join(map(str, data['cyclical_result']))}</strong></p>
                <p>Abordarea grupării: <strong>{', '.join(map(str, data['grouping_result']))}</strong></p>
                <p>Abordarea distribuției: <strong>{', '.join(map(str, data['distribution_result']))}</strong></p>
                <p class="highlight">Recomandare finală combinată: <strong>{', '.join(map(str, data['final_recommendation']))}</strong></p>
            </div>
        </div>

        <div class="section">
            <h2>Frecvența numerelor</h2>
            <p>Această secțiune prezintă analiza frecvenței de apariție a fiecărui număr în extragerile examinate.</p>
            
            <div class="chart-container">
                <img src="data:image/png;base64,{data['frequency_chart']}" alt="Grafic frecvență" style="max-width: 100%;">
            </div>
            
            <h3>Top 10 cele mai frecvente numere</h3>
            <div class="grid-container">
                {' '.join([f'<div class="number-box {("odd" if num % 2 != 0 else "even")}">{num}<br>({count})</div>' for num, count in data['most_common']])}
            </div>
            
            <h3>Top 10 cele mai puțin frecvente numere</h3>
            <div class="grid-container">
                {' '.join([f'<div class="number-box {("odd" if num % 2 != 0 else "even")}">{num}<br>({count})</div>' for num, count in data['least_common']])}
            </div>
        </div>

        <div class="section">
            <h2>Recența numerelor</h2>
            <p>Această secțiune arată cât de recent a apărut fiecare număr în ultimele 50 de extrageri.</p>
            
            <div class="chart-container">
                <img src="data:image/png;base64,{data['recency_chart']}" alt="Grafic recență" style="max-width: 100%;">
            </div>
            
            <h3>Numere neapărute în ultimele 50 de extrageri</h3>
            <div class="grid-container">
                {' '.join([f'<div class="number-box {("odd" if num % 2 != 0 else "even")}">{num}</div>' for num in data['not_seen_recently']]) if data['not_seen_recently'] else '<p>Toate numerele au apărut cel puțin o dată în ultimele 50 de extrageri.</p>'}
            </div>
        </div>

        <div class="section">
            <h2>Analiza distribuției</h2>
            <p>Această secțiune analizează distribuția numerelor pare/impare și mici/mari în extragerile anterioare.</p>
            
            <div style="display: flex; flex-wrap: wrap; justify-content: space-between; margin-bottom: 20px;">
                <div style="flex: 1; min-width: 300px; margin-right: 20px;">
                    <h3>Distribuția par/impar</h3>
                    <canvas id="oddEvenChart" width="400" height="300"></canvas>
                </div>
                <div style="flex: 1; min-width: 300px;">
                    <h3>Distribuția mic/mare (1-40/41-80)</h3>
                    <canvas id="lowHighChart" width="400" height="300"></canvas>
                </div>
            </div>
            
            <div style="display: flex; flex-wrap: wrap; justify-content: space-between;">
                <div style="flex: 1; min-width: 300px; margin-right: 20px;">
                    <h3>Tendința numerelor impare</h3>
                    <img src="data:image/png;base64,{data['odd_trend_chart']}" alt="Trend numere impare" style="max-width: 100%;">
                </div>
                <div style="flex: 1; min-width: 300px;">
                    <h3>Tendința numerelor mici (1-40)</h3>
                    <img src="data:image/png;base64,{data['low_trend_chart']}" alt="Trend numere mici" style="max-width: 100%;">
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>Analiza grupării</h2>
            <p>Această secțiune analizează cum diferitele numere tind să apară împreună.</p>
            
            <div class="chart-container">
                <img src="data:image/png;base64,{data['pair_heatmap']}" alt="Heatmap perechi" style="max-width: 100%;">
            </div>
            
            <h3>Heatmap de poziție</h3>
            <p>Acest grafic arată frecvența fiecărui număr în fiecare poziție de extragere (1-20).</p>
            <div class="chart-container">
                <img src="data:image/png;base64,{data['position_heatmap']}" alt="Heatmap poziție" style="max-width: 100%;">
            </div>
        </div>
        
        <div class="section">
            <h2>Probabilități și șanse</h2>
            <p>Această secțiune prezintă probabilitățile teoretice de a nimeri un anumit număr de numere când se joacă 5 numere.</p>
            
            <table>
                <thead>
                    <tr>
                        <th>Numere nimerite</th>
                        <th>Probabilitate</th>
                        <th>Procent</th>
                        <th>Șanse</th>
                    </tr>
                </thead>
                <tbody>
                    {' '.join([f'<tr><td>{p["matches"]}</td><td>{p["probability"]:.8f}</td><td>{p["percentage"]:.4f}%</td><td>{p["odds"]}</td></tr>' for p in data['probability_table']])}
                </tbody>
            </table>
        </div>
        
        <div class="section">
            <h2>Detalii abordări</h2>
            
            <h3>Abordarea ciclică</h3>
            <p>Această abordare se bazează pe ideea că numerele care nu au apărut de mult timp ar putea avea o probabilitate mai mare să apară în viitorul apropiat.</p>
            <div class="approach-result">
                <p>Rezultat: <strong>{', '.join(map(str, data['cyclical_result']))}</strong></p>
            </div>
            
            <h3>Abordarea grupării</h3>
            <p>Această abordare analizează care numere tind să apară împreună și generează combinații bazate pe aceste modele.</p>
            <div class="approach-result">
                <p>Rezultat: <strong>{', '.join(map(str, data['grouping_result']))}</strong></p>
            </div>
            
            <h3>Abordarea distribuției</h3>
            <p>Această abordare menține distribuția par/impar și mic/mare similară cu cea observată în extragerile anterioare.</p>
            <div class="approach-result">
                <p>Rezultat: <strong>{', '.join(map(str, data['distribution_result']))}</strong></p>
            </div>
            
            <h3>Recomandarea finală</h3>
            <p>Această recomandare combină cele trei abordări de mai sus, favorizând numerele care apar în mai multe abordări.</p>
            <div class="final-recommendation">
                <p>Recomandare finală: <strong>{', '.join(map(str, data['final_recommendation']))}</strong></p>
            </div>
        </div>
        
        <div class="footer">
            <p>Raport generat la {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>
            <p>Acest raport este generat automat și are scop informativ. Loteria este un joc de noroc, iar rezultatele anterioare nu garantează câștiguri viitoare.</p>
        </div>
        
        <script>
            // Chart.js scripts for interactive charts
            document.addEventListener('DOMContentLoaded', function() {{
                // Odd-Even Distribution Chart
                const oddEvenCtx = document.getElementById('oddEvenChart').getContext('2d');
                const oddEvenChart = new Chart(oddEvenCtx, {{
                    type: 'pie',
                    data: {{
                        labels: {data['distribution_data']['odd_even']['labels']},
                        datasets: [{{ 
                            data: {data['distribution_data']['odd_even']['percentages']},
                            backgroundColor: ['#FF6B6B', '#4ECDC4'],
                            borderWidth: 1
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        plugins: {{
                            legend: {{ position: 'bottom' }},
                            tooltip: {{
                                callbacks: {{
                                    label: function(context) {{
                                        return context.label + ': ' + context.raw.toFixed(1) + '%';
                                    }}
                                }}
                            }}
                        }}
                    }}
                }});
                // Similar pentru Low-High Distribution Chart
                const lowHighCtx = document.getElementById('lowHighChart').getContext('2d');
                const lowHighChart = new Chart(lowHighCtx, {{
                    type: 'pie',
                    data: {{
                        labels: {data['distribution_data']['low_high']['labels']},
                        datasets: [{{ 
                            data: {data['distribution_data']['low_high']['percentages']},
                            backgroundColor: ['#F9ED69', '#3A6EA5'],
                            borderWidth: 1
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        plugins: {{
                            legend: {{ position: 'bottom' }},
                            tooltip: {{
                                callbacks: {{
                                    label: function(context) {{
                                        return context.label + ': ' + context.raw.toFixed(1) + '%';
                                    }}
                                }}
                            }}
                        }}
                    }}
                }});
            }});
        </script>
    </body>
    </html>
    """
    
    # Write the HTML file
    with open(full_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML report generated at: {full_path}")
    return full_path

def plot_frequency_chart(frequency_data):
    """Generate frequency distribution chart"""
    plt.figure(figsize=(12, 6))
    numbers = [x['number'] for x in frequency_data]
    frequencies = [x['frequency'] for x in frequency_data]
    
    sns.barplot(x=numbers, y=frequencies, palette="viridis")
    plt.title('Distribuția frecvenței numerelor')
    plt.xlabel('Număr')
    plt.ylabel('Frecvență apariții')
    plt.xticks(rotation=90, ticks=range(0, 80, 4))
    
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def plot_recency_chart(frequency_data):
    """Generate recency chart (last seen)"""
    plt.figure(figsize=(12, 6))
    numbers = [x['number'] for x in frequency_data]
    last_seen = [x['last_seen'] if isinstance(x['last_seen'], int) else 51 for x in frequency_data]
    
    sns.barplot(x=numbers, y=last_seen, palette="rocket")
    plt.title('Recența apariției numerelor (ultimele 50 extrageri)')
    plt.xlabel('Număr')
    plt.ylabel('Extrageri de la ultima apariție')
    plt.xticks(rotation=90, ticks=range(0, 80, 4))
    plt.ylim(0, 55)
    
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def plot_heatmap(drawings):
    """Generate position heatmap"""
    position_counts = np.zeros((20, 80))
    
    for drawing in drawings:
        for pos, num in enumerate(drawing):
            position_counts[pos][num-1] += 1
    
    plt.figure(figsize=(16, 8))
    sns.heatmap(position_counts, cmap="YlGnBu", cbar_kws={'label': 'Frecvență'})
    plt.title('Frecvența numerelor pe poziții de extragere')
    plt.xlabel('Număr')
    plt.ylabel('Poziție în extragere')
    
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def plot_pair_heatmap(pair_data):
    """Generate pair co-occurrence heatmap"""
    matrix = np.zeros((80, 80))
    
    for pair in pair_data:
        num1 = pair['num1'] - 1
        num2 = pair['num2'] - 1
        matrix[num1][num2] = pair['count']
        matrix[num2][num1] = pair['count']
    
    plt.figure(figsize=(16, 16))
    sns.heatmap(matrix, cmap="viridis", square=True, cbar_kws={'label': 'Apariții comune'})
    plt.title('Heatmap perechi de numere apărute împreună')
    plt.xlabel('Număr')
    plt.ylabel('Număr')
    
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def plot_odd_even_trend(odd_counts):
    """Plot odd number trend"""
    plt.figure(figsize=(12, 6))
    plt.plot(odd_counts, marker='o', linestyle='-', color='#FF6B6B')
    plt.title('Evoluția numărului de numere impare (ultimele 50 extrageri)')
    plt.xlabel('Extragere')
    plt.ylabel('Numere impare')
    plt.ylim(0, 20)
    
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def plot_low_high_trend(low_counts):
    """Plot low numbers trend"""
    plt.figure(figsize=(12, 6))
    plt.plot(low_counts, marker='o', linestyle='-', color='#3A6EA5')
    plt.title('Evoluția numărului de numere mici (1-40) (ultimele 50 extrageri)')
    plt.xlabel('Extragere')
    plt.ylabel('Numere mici')
    plt.ylim(0, 20)
    
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def main(file_path, num_to_generate=5):
    """
    Main function to analyze Keno drawings and generate HTML report
    """
    # Load data
    drawings = load_data(file_path)
    
    if not drawings:
        print("No data loaded. Please check the file path.")
        return
    
    # Get strategy recommendations
    recommendations = get_strategy_recommendations(drawings)
    
    # Get frequency data for charts
    frequency_data = recommendations['frequency_data']
    
    # Generate recommended numbers using each approach
    cyclical_numbers, cyclical_weights = approach_1_cyclical(drawings, num_to_generate)
    grouping_numbers, pair_data = approach_2_grouping(drawings, num_to_generate)
    distribution_numbers, distribution_data = approach_3_distribution(drawings, num_to_generate)
    
    # Generate charts
    frequency_chart = plot_frequency_chart(frequency_data)
    recency_chart = plot_recency_chart(frequency_data)
    position_heatmap = plot_heatmap(drawings)
    pair_heatmap = plot_pair_heatmap(pair_data)
    odd_trend_chart = plot_odd_even_trend(distribution_data['odds_by_draw'])
    low_trend_chart = plot_low_high_trend(distribution_data['lows_by_draw'])
    
    # Generate final recommendation
    all_suggestions = cyclical_numbers + grouping_numbers + distribution_numbers
    counter = Counter(all_suggestions)
    final_recommendation = [num for num, count in counter.most_common() if count > 1]
    
    # Dacă recomandarea finală conține mai puțin de num_to_generate numere,
    # completăm selectând următoarele numere cu cele mai mari frecvențe din all_suggestions
    while len(final_recommendation) < num_to_generate:
        remaining = [num for num, count in counter.most_common() if num not in final_recommendation]
        if remaining:
            final_recommendation.append(remaining[0])
        else:
            break

    # Dacă am acumulat mai multe numere decât este necesar, le restrângem la num_to_generate
    final_recommendation = final_recommendation[:num_to_generate]

    # Construim dicționarul de date pentru raport folosind rezultatele obținute
    data = {
        'total_drawings': len(drawings),
        'cyclical_result': cyclical_numbers,
        'grouping_result': grouping_numbers,
        'distribution_result': distribution_numbers,
        'final_recommendation': final_recommendation,
        'frequency_chart': frequency_chart,
        'recency_chart': recency_chart,
        'position_heatmap': position_heatmap,
        'pair_heatmap': pair_heatmap,
        'odd_trend_chart': odd_trend_chart,
        'low_trend_chart': low_trend_chart,
        'distribution_data': distribution_data,
        'most_common': recommendations['most_common'],
        'least_common': recommendations['least_common'],
        'probability_table': recommendations['probability_table'],
        'not_seen_recently': recommendations['not_seen_recently']
    }

    # Generăm raportul HTML folosind datele construite
    report_path = generate_html_report(data)
    print("Report generated:", report_path)

if __name__ == "__main__":
    # Schimbă 'keno_drawings.csv' cu numele fișierului tău
    main(file_path="date_rafinate_4_luni.csv", num_to_generate=5)