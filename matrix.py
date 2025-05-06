import numpy as np

###################################################################################
################################## Exercício 5.1 ##################################
###################################################################################
# This exercise will help you gain familiarity with indexing matrix elements.
# Create a 3 × 4 matrix using np.arange(12).reshape(3,4). Then write Python code to
# extract the element in the second row, fourth column. Use softcoding so that you 
# can select different row/column indices. Print out a message like the following:

# The matrix element at index (2,4) is 7.
###################################################################################

print("\n")
matrix51 = np.arange(12).reshape(3,4)
index24 = matrix51[1,3]
print ("The matrix element at index (2,4) is", index24)
print("\n")

###################################################################################
################################## Exercício 5.2 ##################################
###################################################################################
# This and the following exercise focus on slicing matrices to obtain submatrices.
# Start by creating matrix C in Figure 5-6, and use Python slicing to extract the 
# submatrix comprising the first five rows and five columns. Let’s call this matrix C1.
# Try to reproduce Figure 5-6, but if you are struggling with the Python visualization
# coding, then just focus on extracting the submatrix correctly.
###################################################################################

matrixC = np.arange(100).reshape(10,10)

print(matrixC)

print("\n")

matrixC1 = matrixC[0:5,0:5]

print(matrixC1)

print("\n")

###################################################################################
################################## Exercício 5.3 ##################################
###################################################################################
# Expand this code to extract the other four 5 × 5 blocks. Then create a new matrix
# with those blocks reorganized according to Figure 5-7.
###################################################################################

matrixC11 = matrixC[5:10,5:10]
matrixC12 = matrixC[5:10,0:5]
matrixC21 = matrixC[0:5,5:10]
matrixC22 = matrixC[0:5,0:5]

lin1 = np.concatenate((matrixC11,matrixC12),axis=0)
lin2 = np.concatenate((matrixC21,matrixC22),axis=0)

print(np.concatenate((lin1,lin2),axis=1))

print("\n")

###################################################################################
################################## Exercício 5.4 ##################################
###################################################################################
# Implement matrix addition element-wise using two for loops over rows and columns.
# What happens when you try to add two matrices with mismatching sizes? This
# exercise will help you think about breaking down a matrix into rows, columns,
# and individual elements.
###################################################################################

def sum_matrix(M1,M2):
    if len(M1) != len(M2) or len(M1[0]) != len(M2[0]):
        print("Impssível calcular a soma dessas matrizes")
        return
    
    somaMatrizes = [[0 for _ in range(len(M1[0]))] for _ in range(len(M1)) ] 

    for i in range(len(M1)):
        for j in range(len(M1[0])):
            somaMatrizes[i][j] = M1[i][j] + M2[i][j]
    
    return somaMatrizes

mSoma = sum_matrix(matrixC12,matrixC21)
print(mSoma)
print("\n")

###################################################################################
################################## Exercício 5.5 ##################################
###################################################################################
# Matrix addition and scalar multiplication obey the mathematical laws of commutivity
# and distributivity. That means that the following equations give the same results
# (assume that the matrices A and B are the same size and that σ is some scalar):
# 
# σ A + B = σA + σB = Aσ + Bσ
# 
# Rather than proving this mathematically, you are going to demonstrate it through
# coding. In Python, create two random-numbers matrices of size 3 × 4 and a random
# scalar. Then implement the three expressions in the previous equation. You’ll need to
# figure out a way to confirm that the three results are equal. Keep in mind that tiny
# computer precision errors in the range of 10−15 should be ignored.
###################################################################################



###################################################################################
################################## Exercício 5.6 ##################################
###################################################################################
# Code matrix multiplication using for loops. Confirm your results against using the
# numpy @ operator. This exercise will help you solidify your understanding of matrix
# multiplication, but in practice, it’s always better to use @ instead of writing out a
# double for loop.
###################################################################################

def matrix_multiply(A, B):
    
    if len(A[0]) != len(B):
        print("Error: matriz incompatível")
        return None
   
    result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]

    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    
    return result

A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]
print("multiplicação padrão:", matrix_multiply(A, B))
print("multiplicação com NumPy:", np.array(A) @ np.array(B))
print("\n")
###################################################################################
################################## Exercício 5.7 ##################################
###################################################################################
# Confirm the LIVE EVIL rule using the following five steps:
# (1) Create four matrices
# of random numbers, setting the sizes to be L ∈ R2 × 6, I ∈ R6 × 3, V ∈ R3 × 5, and
# E ∈ R5 × 2.
# (2) Multiply the four matrices and transpose the product.
# (3) Transpose each matrix individually and multiply them without reversing their order.
# (4) Transpose each matrix individually and multiply them reversing their order according
# to the LIVE EVIL rule. Check whether the result of step 2 matches the results of step 3
# and step 4.
# (5) Repeat the previous steps but using all square matrices.
###################################################################################

#1: Cria as 4 matrizes aleatórias
L = np.random.rand(2, 6)
I = np.random.rand(6, 3)
V = np.random.rand(3, 5)
E = np.random.rand(5, 2)

#2: Multiplica e transpoe


#3: Transpoe individualmente e multiplica na ordem original


#4: Transpoe individualmente e multiplica na ordem inversa


#5: verifica a igualdade


#5: matrizes quadradas




###################################################################################
################################## Exercício 5.8 ##################################
###################################################################################
# In this exercise, you will write a Python function that checks whether a matrix is
# symmetric. It should take a matrix as input, and should output a boolean True if
# the matrix is symmetric or False if the matrix is nonsymmetric. Keep in mind that
# small computer rounding/precision errors can make “equal” matrices appear unequal.
###################################################################################

def is_symmetric(matrix, tolerance=1e-8):
    if len(matrix) != len(matrix[0]):
        return False
 
    for i in range(len(matrix)):
        for j in range(i + 1, len(matrix)):
            if abs(matrix[i][j] - matrix[j][i]) > tolerance:
                return False
    return True

matriz1 = [[1, 2, 3],
            [2, 4, 5],
            [3, 5, 6]]
matriz2 = [[1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]]

print("Matriz simétrica", is_symmetric(matriz1)) 
print("matriz não-simétrica:", is_symmetric(matriz2)) 