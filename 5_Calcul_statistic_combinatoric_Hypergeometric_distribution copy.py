from math import comb

N = 80  # Total numere
K = 5  # Numere alese de jucător
n = 20  # Numere extrase
k = 3   # Numere corecte ghicite

prob = (comb(K, k) * comb(N - K, n - k)) / comb(N, n)
print(f"Probabilitate: {prob:.4f} (~{prob*100:.2f}%)")