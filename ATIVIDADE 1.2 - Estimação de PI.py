import numpy as np

def pi_aprox(ktermos):   
    n = np.arange(ktermos)
    arctan = (-1)**n / (2*n + 1)
    pi = 4 * np.sum(arctan)
    return pi

k = 100000
pi_calculado = pi_aprox(k)
print(f'Aproximação de pi com {k} termos: {pi_calculado}')
