Extension Task on top of sycl_device_query.cpp

Task A: Platform Filter (Extension of Section 5)
Modify the device query program to filter and count only the platforms that contain the string `"Intel"` in their name, and print out only those devices.*

Task B: Vector Element-wise Multiplication (Extension of Section 6)
Write a SYCL kernel to perform element-wise multiplication ( C[i]=A[i]×B[i] ) on 5 Million elements using Unified Shared Memory (USM).

Task C: Transposed Matrix Initialization (Extension of Section 7)*
Initialize a  512×512  matrix where each element at  (i,j)  is set to  i×j . Write a simple CPU/OMP parallel loop to verify that the transpose of this matrix matches your target mathematical values
