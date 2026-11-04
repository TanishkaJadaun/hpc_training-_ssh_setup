Extension Task on top of sycl_device_query.cpp

Task A: Platform Filter (Extension of Section 5)
Modify the device query program to filter and count only the platforms that contain the string `"Intel"` in their name, and print out only those devices.*

Task B: Vector Element-wise Multiplication (Extension of Section 6)
Write a SYCL kernel to perform element-wise multiplication ( C[i]=A[i]×B[i] ) on 5 Million elements using Unified Shared Memory (USM).

Task C: Transposed Matrix Initialization (Extension of Section 7)*
Initialize a  512×512  matrix where each element at  (i,j)  is set to  i×j . Write a simple CPU/OMP parallel loop to verify that the transpose of this matrix matches your target mathematical values

#TASK C

import numpy as np
from numba import njit, prange

N = 512

A = np.zeros((N, N), dtype=np.int32)

@njit(parallel=True)
def initialize_matrix(A):
    for i in prange(N):
        for j in range(N):
            A[i, j] = i * j

@njit(parallel=True)
def transpose_matrix(A):
    T = np.zeros((N, N), dtype=np.int32)

    for i in prange(N):
        for j in range(N):
            T[j, i] = A[i, j]

    return T

@njit(parallel=True)
def verify_transpose(T):
    errors = 0

    for i in prange(N):
        for j in range(N):
            if T[i, j] != i * j:
                errors += 1

    return errors

initialize_matrix(A)

T = transpose_matrix(A)

errors = verify_transpose(T)

if errors == 0:
    print("Transpose verification successful!")
    print("All elements match the mathematical values.")
else:
    print("Transpose verification failed!")
    print("Number of incorrect elements:", errors)

print("\nOriginal Matrix (first 5x5):")
print(A[:5, :5])

print("\nTransposed Matrix (first 5x5):")
print(T[:5, :5])
