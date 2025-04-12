###########################################################################
##########################   exercicio 1.1   ##############################

import numpy as np

def sqrt_div_avg(a, x0, iterations=10):
    x = x0
    errors = []
    true_value = np.sqrt(a)
    
    for i in range(iterations):
        x_new = (x + a/x)/2
        rel_error = abs(x_new - true_value)/true_value
        errors.append(rel_error)
        x = x_new
    
    return errors

# Exemplo para √2 começando com x0=1
errors = sqrt_div_avg(2, 1)
for i, error in enumerate(errors, 1):
    print(f"Iteração {i}: Erro relativo = {error:.2e}")


###########################################################################
##########################   exercicio 1.2   ##############################

def machine_epsilon():
    epsilon = 1.0
    while 1.0 + epsilon > 1.0:
        epsilon /= 2
    return 2 * epsilon

epsilon_calculado = machine_epsilon()
epsilon_numpy = np.finfo(float).eps

print(f"Épsilon calculado: {epsilon_calculado:.20e}")
print(f"Épsilon do NumPy: {epsilon_numpy:.20e}")
print(f"Diferença: {abs(epsilon_calculado - epsilon_numpy):.2e}")


###########################################################################
##########################   exercicio 1.3   ##############################

def seq_constante():
    x = 1/3  # Valor inicial
    print(f"x(1) = {x:.15f}")
    
    for n in range(2, 11):  # Vamos calcular até x(10)
        x = 4 * x - 1
        print(f"x({n}) = {x:.15f}")
    

###########################################################################
##########################   exercicio 1.4   ##############################

def comp_dif():
    mu_values = [0.1, 0.01, 0.001, 1e-5, 1e-10, 1e-15]
    
    print("{:<10} {:<25} {:<25} {:<15}".format(
        "μ", "Expressão 1", "Expressão 2", "Diferença"))
    
    for mu in mu_values:
        # Expressão 1: exp(1/μ)/(1 + exp(1/μ))
        try:
            exp1 = np.exp(1/mu) / (1 + np.exp(1/mu))
        except:
            exp1 = float('inf')
        
        # Expressão 2: 1/(exp(-1/μ) + 1)
        try:
            exp2 = 1 / (np.exp(-1/mu) + 1)
        except:
            exp2 = float('inf')
        
        diff = abs(exp1 - exp2) if not (np.isinf(exp1) or np.isinf(exp2)) else np.nan
        
        print("{:<10.0e} {:<25.15f} {:<25.15f} {:<15.2e}".format(
            mu, exp1, exp2, diff))
    

###########################################################################
##########################   exercicio 1.5   ##############################
def f(x):
    return ((1 + x) - 1)/x

x_values = [1e-12, 1e-15, 1e-17]
print("Análise da função f(x) = ((1 + x) - 1)/x:")
print("{:<10} {:<15} {:<15}".format("x", "f(x) calculado", "Erro relativo"))

for x in x_values:
    fx = f(x)
    erro = abs(fx - 1)
    print("{:<10.1e} {:<15.15f} {:<15.2e}".format(x, fx, erro))

############################################################################
##########################   exercicio 1.6   ###############################

import matplotlib.pyplot as plt

def arctan_series(x, terms):
    result = 0
    for n in range(terms):
        term = (-1)**n * x**(2*n + 1)/(2*n + 1)
        result += term
    return result

methods = {
    "Machin": lambda t: 4*(4*arctan_series(1/5, t) - arctan_series(1/239, t)),
    "Hutton": lambda t: 4*(arctan_series(1/2, t) + arctan_series(1/3, t)),
    "Clausen": lambda t: 4*(2*arctan_series(1/3, t) + arctan_series(1/7, t)),
    "Dase": lambda t: 4*(arctan_series(1/2, t) + arctan_series(1/5, t) + arctan_series(1/8, t))
}

max_terms = 20
terms = range(1, max_terms + 1)
pi_true = np.pi

plt.figure(figsize=(10, 6))
for name, method in methods.items():
    errors = [abs(method(t) - pi_true) for t in terms]
    plt.plot(terms, errors, 'o-', label=name)

plt.yscale('log')
plt.xlabel('Número de Termos')
plt.ylabel('Erro Absoluto (escala log)')
plt.title('Comparação de Métodos para Cálculo de π')
plt.legend()
plt.grid(True)
plt.show()