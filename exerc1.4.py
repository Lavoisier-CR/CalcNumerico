import numpy as np
import matplotlib.pyplot as plt


##################################################################################################
##########   1. Mostrar que ( e^{ix} = \cos(x) + i \sin(x) ) usando séries de Taylor    ##########
##################################################################################################

def exp_i_x(x, n_terms=10):
    # Série de Taylor para e^{ix} = sum_{k=0}^{n} (i x)^k / k!
    result = np.zeros_like(x, dtype=complex)
    for k in range(n_terms):
        term = (1j * x) ** k / np.math.factorial(k)
        result += term
    return result

# Ex
x = np.linspace(-2*np.pi, 2*np.pi, 100)
aprox = exp_i_x(x, n_terms=20)
true_cos = np.cos(x)
true_sin = np.sin(x)

plt.plot(x, np.real(aprox), label='Re(aprox)')
plt.plot(x, true_cos, '--', label='cos(x)')
plt.legend()
plt.show()

plt.plot(x, np.imag(aprox), label='Im(aprox)')
plt.plot(x, true_sin, '--', label='sin(x)')
plt.legend()
plt.show()

###############################################################################################
##################   2. Aproximação de (\sin(x) \approx x) para (x \to 0)    ##################
###############################################################################################

x_peq = np.linspace(-0.1, 0.1, 100)
sin_aprox = x_peq  # primeira ordem
atual_sin = np.sin(x_peq)

plt.plot(x_peq, atual_sin, label='sin(x)')
plt.plot(x_peq, sin_aprox, '--', label='aproximacao de x')
plt.legend()
plt.show()


###############################################################################################
##########################   3. Série de Taylor de ( e^{x^2} )    #############################
###############################################################################################

def my_double_exp(x, n):
    # Aproximação de e^{x^2}
    result = np.zeros_like(x, dtype=float)
    for k in range(n):
        term = (x ** (2 * k)) / np.math.factorial(k)
        result += term
    return result

# Teste
x_vals = np.linspace(-2, 2, 100)
approx_exp = my_double_exp(x_vals, 10)
true_exp = np.exp(x_vals ** 2)

plt.plot(x_vals, true_exp, label='exp(x^2)')
plt.plot(x_vals, approx_exp, '--', label='Aproximação de Taylor')
plt.legend()
plt.show()


###############################################################################################
###################   4. Série de Taylor para ( e^x ), ordem de 1 a 7    ######################
###############################################################################################

def my_exp_Taylor(x, n):
    result = np.zeros_like(x, dtype=float)
    for k in range(n+1):
        result += (x ** k) / np.math.factorial(k)
    return result

# Erro
x_value = 1.0  # ponto
errors = []
for n in range(1, 8):
    approx = my_exp_Taylor(x_value, n)
    error = np.abs(np.exp(x_value) - approx)
    errors.append(error)

print("Erros de truncamento para diferentes ordens:")
for i, e in enumerate(errors, start=1):
    print(f'Ordem {i}: erro = {e}')


###############################################################################################
#######   5. Série de Taylor para ( \sin(x) ), ( \cos(x) ) e produto em torno de 0    #########
###############################################################################################

x = np.pi/2

# Série de Taylor de ordem 4
sin_taylor = x - x**3/6 + x**5/120 - x**7/5040
cos_taylor = 1 - x**2/2 + x**4/24 - x**6/720

# Produto das séries
produto_separado = sin_taylor * cos_taylor

# Valor real
sin_real = np.sin(x)
cos_real = np.cos(x)
produto_real = sin_real * cos_real

print(f'Sin(x): real={sin_real}, aproximado={sin_taylor}')
print(f'Cos(x): real={cos_real}, aproximado={cos_taylor}')
print(f'Produto real = {produto_real}')
print(f'Produto pela expansão de Taylor = {produto_separado}')



###############################################################################################
###########################   6. Aproximação de ( \cosh(x) )0    ##############################
###############################################################################################

def my_cosh_approximator(x, n):
    result = np.zeros_like(x, dtype=float)
    for k in range(n+1):
        term = (x ** (2 * k)) / np.math.factorial(2 * k)
        result += term
    return result

# Exemplo
x_vals = np.array([0, 1, 2])
for n in range(0, 5):
    print(f'n={n}')
    print(my_cosh_approximator(x_vals, n))
    print(np.cosh(x_vals))

