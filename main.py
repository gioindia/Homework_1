import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg
import sys

m = 0.15

def read_dat(file_name):
    labels = {}
    row_indices = [] # Lists to store sparse matrix coordinates
    col_indices = []
    
    try:
        with open(file_name, 'r') as file:
            first_line = file.readline().strip()
            if not first_line:
                 return None, None
            parts = first_line.split()
            num_nodes = int(parts[0])
            num_edges = int(parts[1])
            
            # Use COO format for construction (efficient for appending)
            # Later convert to CSC (Compressed Sparse Column) for calculation
            
            for _ in range(num_nodes):
                line = file.readline().strip()
                if line:
                    parts = line.split(maxsplit=1) 
                    node_id = int(parts[0])
                    node_name = parts[1]
                    labels[node_id] = node_name

            for _ in range(num_edges):
                line = file.readline().strip()
                if line:
                    parts = line.split()
                    source = int(parts[0])
                    target = int(parts[1])
                    # Store coordinates instead of filling dense matrix directly
                    # A[target-1][source-1]=1
                    row_indices.append(target - 1)
                    col_indices.append(source - 1)
            
            # Create sparse matrix with 1s at specific coordinates
            data = np.ones(len(row_indices))
            A = sparse.coo_matrix((data, (row_indices, col_indices)), shape=(num_nodes, num_nodes)).tocsc()
            
            # Efficient column normalization for sparse matrix
            # Calculate sum of each column
            col_sums = np.array(A.sum(axis=0)).flatten()
            
            # Avoid division by zero. If sum is 0, scaling factor is 0.
            with np.errstate(divide='ignore', invalid='ignore'):
                scale_factors = np.where(col_sums != 0, 1.0 / col_sums, 0)
            
            # Multiply A by diagonal matrix of inverse sums to normalize
            D_inv = sparse.diags(scale_factors)
            A = A @ D_inv
                    
    except FileNotFoundError:
        print(f"Error: File '{file_name}' not found.")
        return None, None
    except Exception as e:
        print(f"Errore during the analysis of the file: {e}")
        return None, None
    
    return A, labels

def power_iteration_with_vector(A, s, m, output, tolerance=1e-6, max_iterations=1000):
    n = A.shape[0]
    x = np.ones(n) / n # initial vector (normalized)
    for iteration in range(max_iterations):
        # Sparse matrix multiplication (@) is efficient here
        x_new = (1 - m) * (A @ x) + m * s
        x_new = x_new / np.sum(x_new) # normalized
        if np.linalg.norm(x_new - x, 1) < tolerance:
            print(f"  Converged in {iteration + 1} iterations", file=output)
            break
        x = x_new
    else:
        print(f"  Warning: Maximum iterations ({max_iterations}) reached", file=output)
    return x

def check_dangling_nodes(A):
    # Summing sparse columns is faster using built-in sum
    col_sums = np.array(A.sum(axis=0)).flatten()
    dangling = []
    for i in range(len(col_sums)):
        if col_sums[i] == 0:
            dangling.append(i)
    return dangling

def get_eigenpairs(A, k=None):
    # Helper function: Use Scipy for large matrices, Numpy for small ones.
    # Scipy eigs requires k < N-1, which fails on very small graphs (e.g., 4 nodes).
    n = A.shape[0]
    if k is None: k = min(n - 2, 6)
    if k < 1: k = 1

    if n < 10 or k >= n-1:
        # Fallback to dense for small graphs or when many eigenvalues are needed
        vals, vecs = np.linalg.eig(A.toarray())
    else:
        try:
            # 'LM' = Largest Magnitude
            vals, vecs = splinalg.eigs(A, k=k, which='LM')
        except:
            vals, vecs = np.linalg.eig(A.toarray())
    return vals, vecs

def exercise_4_analysis(A, labels):
    n = A.shape[0]
    
    # Use helper to get eigenvalues efficiently
    eigenvalues, eigenvectors = get_eigenpairs(A)
    
    idx = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    perron_eigenvalue = np.real(eigenvalues[0])
    perron_eigenvector = np.real(eigenvectors[:, 0])
    print(f"PERRON EIGENVALUE (largest): λ = {perron_eigenvalue:.6f}")
    # Make eigenvector non-negative and normalize
    if np.any(perron_eigenvector < 0):
        perron_eigenvector = -perron_eigenvector
    # Ensure real values
    if np.iscomplexobj(perron_eigenvector):
        perron_eigenvector = np.real(perron_eigenvector)
    # Normalize to sum to 1
    perron_eigenvector = perron_eigenvector / np.sum(perron_eigenvector)
    print(f"\nPerron eigenvector (normalized to sum=1):")
    sorted_indices = np.argsort(perron_eigenvector)[::-1]
    print(f"{'-'*50}")
    for rank, idx in enumerate(sorted_indices, 1):
        node_label = labels[idx + 1]
        score = perron_eigenvector[idx]
        print(f"  {rank}. {node_label:20s}: {score:.6f}")    
    # Verify it's an eigenvector
    result = A @ perron_eigenvector
    expected = perron_eigenvalue * perron_eigenvector
    error = np.linalg.norm(result - expected)
    print(f"\nVerification: ||A·v - λ·v|| = {error:.2e}")
    return perron_eigenvalue, perron_eigenvector

def exercise_1():
    filename="graph1.dat"
    A, labels = read_dat(filename)
    if A is None: return

    print("\n" + "="*70)
    print("Exercise 1 Analysis:")
    #Graph 1
    eigenvalues, eigenvectors = get_eigenpairs(A)
    
    # Check if eigenvalue 1 exists
    idx_list = np.where(np.isclose(eigenvalues, 1))[0]
    if len(idx_list) > 0:
        idx = idx_list[0]
        x_raw = np.real(eigenvectors[:, idx])
        importance_score = x_raw / x_raw.sum()
        
        print("Importance scores for Graph 1:")
        sorted_indices = np.argsort(importance_score)[::-1]
        for rank, idx in enumerate(sorted_indices, 1):
            node_label = labels[idx + 1]
            score = importance_score[idx]
            print(f"  {rank}. {node_label:20s}: {score:.6f}")
    
    #Graph 1 with node 5 added
    filename="exercise1_graph.dat"
    A_modified, labels_modified = read_dat(filename)
    if A_modified is not None:
        eigenvalues, eigenvectors = get_eigenpairs(A_modified)
        idx_list = np.where(np.isclose(eigenvalues, 1))[0]
        if len(idx_list) > 0:
            idx = idx_list[0]
            x_raw = np.real(eigenvectors[:, idx])
            importance_score_withnode5 = x_raw / x_raw.sum()
            
            print("\nImportance scores for Graph 1 with Node 5 added:")
            sorted_indices = np.argsort(importance_score_withnode5)[::-1]
            for rank, idx in enumerate(sorted_indices, 1):
                node_label = labels_modified[idx + 1]
                score = importance_score_withnode5[idx]
                print(f"  {rank}. {node_label:20s}: {score:.6f}")
    
    print("We can see thet the addition of Page 5 created a self-reinforcing feedback loop that allowed Page 3 to successfully manipulate the ranking system and overtake Page 1.")
    return

def exercise_2():
    filename="exercise2_graph.dat"
    print("\n" + "="*70)
    print("Exercise 2 Analysis:")
    A, labels = read_dat(filename) 
    if A is None: return
    
    # Use dense fallback for accurate counting of multiplicity on small graphs
    eigenvalues, eigenvectors = get_eigenpairs(A, k=A.shape[0]-1)
    dimension = np.sum(np.isclose(eigenvalues, 1))
    print(f"The dimension of the eigenspace associated with the eigenvalue 1 is: {dimension} >= of the number of the components in the web graph(4).")
    return

def exercise_3():
    filename="exercise3_graph.dat"
    print("\n" + "="*70)
    print("Exercise 3 Analysis:")
    A, labels = read_dat(filename) 
    if A is None: return

    eigenvalues, eigenvectors = get_eigenpairs(A, k=A.shape[0]-1)
    dimension = np.sum(np.isclose(eigenvalues, 1))
    print(f"The dimension of the eigenspace associated with the eigenvalue 1 is: {dimension} because the web contains two closed strongly connected components. Indeed from the node group {1,2} we can't reach the node group {3,4,5} and from the node group {3,4} we can't reach the node group {1,2}.")
    return

def swap_node_indices(input_filename, output_filename, i, j):
    str_i = str(i)
    str_j = str(j)
    try:
        with open(input_filename, 'r') as infile:
            lines = infile.readlines()
        new_lines = []
        for k, line in enumerate(lines):
            line = line.strip()
            if not line:
                new_lines.append('\n')
                continue
            parts = line.split()

            # First line (header)
            if k == 0 and len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                new_lines.append(line + '\n')
                continue

            # General sostitution logic (nodes and links)
            modified_parts = []
            for part in parts:
                if part.isdigit():
                    if part == str_i:
                        modified_parts.append(str_j)
                    elif part == str_j:
                        modified_parts.append(str_i)
                    else:
                        modified_parts.append(part)
                else:
                    modified_parts.append(part)
            new_lines.append(" ".join(modified_parts) + '\n')

        # Output file
        with open(output_filename, 'w') as outfile:
            outfile.writelines(new_lines)
        print(f"Indecises {i} and {j} swapped (Nodes and Links). Result saved in '{output_filename}'.")

    except FileNotFoundError:
        print(f"ERROR: file '{input_filename}' not found.")
    except Exception as e:
        print(f"Error happened: {e}")
    return output_filename

def exercise_6(filename,i,j):
    print("\n" + "="*70)
    print("Exercise 6 Analysis:")
    A, labels = read_dat(filename)
    if A is None: return

    A_eigenvalues, A_eigenvectors = get_eigenpairs(A)
    x = A_eigenvectors[:, 0] # vector c chooosen for proving y=P·x
    l = A_eigenvalues[0] # corresponding eigenvalue

    # Create permutation matrix P that swaps nodes i and j
    # Use sparse matrix for P to maintain efficiency
    n = A.shape[0]
    P = sparse.eye(n, format='lil') # LIL is efficient for changing structure
    P[i-1, i-1] = 0
    P[j-1, j-1] = 0
    P[i-1, j-1] = 1
    P[j-1, i-1] = 1
    P = P.tocsc()
    
    # Compute the theoretical permuted adjacency matrix A2_theoretical
    A2_theoretical = P @ A @ P

    # Apply the swap to the graph file
    print(f"Referring to graph {filename}, we swap the pages with indices i={i} and j={j}.")
    output_file = "exercise6_graph.dat"
    swap_node_indices(filename, output_file, i, j)
    A2, labels2 = read_dat(output_file)

    # Verify A2_theoretical == A2
    # Convert to dense for element-wise comparison loop (graph is small)
    A2_theo_dense = A2_theoretical.toarray()
    A2_dense = A2.toarray()

    print("1)")
    print(f"Verifying that the permuted adjacency matrix A2_theoretical matches the matrix A2 obtained by applying PagesRank on the new graph:", end=" ") 
    for r in range(A2_theo_dense.shape[0]):
        for c in range(A2_theo_dense.shape[1]):
            if (A2_theo_dense[r][c]!=A2_dense[r][c]):
                print(f"Mismatch at position ({r},{c}): A2_theoretical={A2_theo_dense[r][c]}, A2={A2_dense[r][c]}")
                return
    print("the matrices match perfectly.")

    print("2)")
    # Verify that l is an eigenvalue of A2 and find corresponding eigenvector y
    A2_eigenvalues, A2_eigenvectors = get_eigenpairs(A2)
    print(f"Eigenvalue lambda={np.real(l):.6f} of A is an eigenvalue for A2 too: {np.any(np.isclose(A2_eigenvalues, l))}.")
    
    idx_list = np.where(np.isclose(A2_eigenvalues, l))[0]
    if len(idx_list) > 0:
        idx = idx_list[0]
        y_found = A2_eigenvectors[:, idx]
        y_theoretical = P @ x
        
        # Normalize vectors for comparison (eigenvectors are direction only)
        y_found = y_found / np.linalg.norm(y_found)
        y_theoretical = y_theoretical / np.linalg.norm(y_theoretical)

        # 1. verify if y_found is equal to y_theoretical (Px)
        match_positive = np.allclose(y_found, y_theoretical)
        # 2. verify if y_found is equal to -y_theoretical (-Px)
        match_negative = np.allclose(y_found, -y_theoretical)
        # Theorem verified if y_found is equal to either Px or -Px
        is_proven = match_positive or match_negative
        print(f"Verifying that y corresponding to l is equal to P·x (or -P·x): {is_proven}.")    

    """
    Argumentation on the Invariance of Importance Scores:

    1. Transposition Result (Swapping two pages i and j):
    The analysis showed that the relabeled link matrix is Ã = P A P.
    The importance score eigenvector y of Ã is related to the eigenvector x of A by y = Px (or y = -Px).
    Since P is an elementary transposition matrix, the operation y = Px has the effect of SWAPPING the i-th and j-th components of the vector x.
    
    This means that:
    - The new score for the page with index i (y_i) is the original score of page j (x_j).
    - The new score for the page with index j (y_j) is the original score of page i (x_i).
    
    The intrinsic importance of each page (based on its connectivity) is PRESERVED; only the index assigned to it has changed.

    2. Generalization to Any Permutation:
    Any general permutation matrix Q (representing an arbitrary relabeling of N pages) can be expressed as the product of a sequence of elementary transposition matrices (Q = P_k * ... * P_1).
    
    Since we proved that each single transposition (P) does not alter the MAGNITUDES of the scores (only their position in the vector), a sequence of such operations (Q) will also keep the score magnitudes unchanged.
    
    Therefore, ANY arbitrary relabeling of pages leaves the intrinsic importance scores unchanged; it merely permutes (reorganizes) those values within the score vector.
    """
    return

def check_matrix_stochastic(matrix, tol=1e-9):
    column_sums = np.sum(matrix, axis=0)
    is_one = np.isclose(column_sums, 1.0, atol=tol)
    is_stochastic = np.all(is_one)
    return is_stochastic

def exercise_7_stochastic_proof(filename):
    print("\n" + "="*70)
    print("Exercise 7 Analysis:")
    m=0.15
    A, labels = read_dat(filename)
    if A is None: return
    """
    Exercise 7: Proof that M = (1-m)A + mS is column-stochastic.
    This function provides the formal proof and a numerical verification.
    """
    n = A.shape[0]
    # 1. Formal Proof (Printed Argument)
    """
    Formal Proof:
    1. A is column-stochastic: Sum_i(A_ij) = 1 for all j.
    2. S is column-stochastic: S_ij = 1/n, so Sum_i(S_ij) = n * (1/n) = 1 for all j.

    Sum of the j-th column of M:
    Sum_i(M_ij) = Sum_i[ (1-m)A_ij + mS_ij ]
                = (1-m) * Sum_i(A_ij) + m * Sum_i(S_ij)  (By linearity)
                = (1-m) * (1) + m * (1)                 (Substituting the known sums)
                = 1 - m + m = 1

    Conclusion: Since the sum of every column of M is 1, M is column-stochastic.
    """
    
    # 2. Numerical Verification (Using the input matrix A)
    print(f"Numerical Verification (using graph: {filename}):")
    # For stochastic check, we can use column sums directly without forming dense M
    # sum(M) = (1-m)sum(A) + m*sum(S). sum(A) is 1 (if no dangling), sum(S) is 1.
    
    col_sums_A = np.array(A.sum(axis=0)).flatten()
    # Handle dangling nodes where sum(A) is 0 for verification context
    col_sums_M = (1 - m) * col_sums_A + m * 1.0
    
    print(f"Matrix M is column-stochastic: {np.allclose(col_sums_M, 1.0)}")
    return

def exercise_11():
    print("Exercise 11 Analysis:")
    analyze_graph("exercise11_graph.dat", m=0.15)
    return

def exercise_12():
    print("Exercise 12 Analysis:")
    A,labels = read_dat("exercise12_graph.dat")
    if A is None: return
    
    eigenvalues, eigenvectors = get_eigenpairs(A)
    idx_list = np.where(np.isclose(eigenvalues, 1))[0]
    
    if len(idx_list) > 0:
        idx = idx_list[0]
        x_raw = np.real(eigenvectors[:, idx])
        importance_score = x_raw / x_raw.sum()
        print("Importance scores with matrix A:")
        sorted_indices = np.argsort(importance_score)[::-1]
        for rank, idx in enumerate(sorted_indices, 1):
            node_label = labels[idx + 1]
            score = importance_score[idx]
            print(f"  {rank}. {node_label:20s}: {score:.6f}")
            
    print("\nNow using PageRank with m=0.15:")
    analyze_graph("exercise12_graph.dat", m=0.15)
    print("The Exercise 12 results demonstrate that the original PageRank model (Matrix A) fails to assign any importance to the dangling Node 6 (0.00) because it lacks backlinks, whereas the modified PageRank model (Matrix M) successfully incorporates Node 6's contribution by giving it a positive minimal score (m/n = 0.025000), distributing its importance across the web and providing a more robust, non-ambiguous ranking where Node 3 remains the most important page in both scenarios.\n\n")
    return

def exercise_13():
    print("="*70)
    print("Exercise 13 Analysis:")
    analyze_graph("exercise13_graph.dat", m=0.15)
    print("The analysis using matrix M shows that the isolated pair (Nodes 6-7) outranks the peripheral nodes of the larger cluster (Nodes 2-5). This demonstrates that out-degree dilution (x_1/4) significantly weakens the authority transferred by the central hub compared to the undiluted reciprocity (x_j/1) retained within the smaller clique.\n\n")
    return

def exercise_14():
    print("\n" + "="*70)
    print("Exercise 14 Analysis (Convergence Speed):")
    
    filename = "exercise11_graph.dat"
    m = 0.15
    A, labels = read_dat(filename)
    n = A.shape[0]
    
    # Explicitly construct M to calculate eigenvalues and c
    S = np.ones((n, n)) / n
    M = (1 - m) * A + m * S
    
    # Calculation of c according to Proposition 4
    min_M_ij = np.min(M) 
    c_bound = 1 - 2 * min_M_ij
    
    # Calculation of eigenvalues to find lambda_2
    eigenvalues, _ = np.linalg.eig(M)
    sorted_abs_eig = np.sort(np.abs(eigenvalues))[::-1] 
    lambda_2 = sorted_abs_eig[1] # second largest eigenvalue
    
    print(f"Theoretical Bound c (Prop 4): {c_bound:.6f}")
    print(f"Second Largest Eigenvalue |lambda_2|: {lambda_2:.6f}")
    print(f"Expected Convergence Rate (1-m): {1-m:.6f}")
    print("-" * 60)

    # 3. Find the "True" q (using a very tight convergence)
    s_vec = np.ones(n) / n
    q = power_iteration_with_vector(A, s_vec, m, sys.stdout, tolerance=1e-14, max_iterations=2000)
    
    np.random.seed(42)
    x_k = np.random.rand(n)
    x_k = x_k / np.sum(x_k)
    
    # Initial error (k=0)
    error_prev = np.linalg.norm(x_k - q, 1)
    
    print(f"\nIteration Analysis:")
    print(f"{'k':<5} | {'Error ||M^k x - q||_1':<25} | {'Ratio (Err_k / Err_k-1)':<25}")
    print("-" * 60)
    
    target_steps = range(5,51,5)
    
    for k in range(1, 51):
        
        x_new = (1 - m) * (A @ x_k) + m * s_vec
        x_new = x_new / np.sum(x_new)
        
        error_curr = np.linalg.norm(x_new - q, 1)
        ratio = error_curr / error_prev if error_prev > 0 else 0
        
        if k in target_steps:
            print(f"{k:<5} | {error_curr:<25} | {ratio:<25}")
        
        x_k = x_new
        error_prev = error_curr

    print("The results confirm that the PageRank algorithm converges much faster than the pessimistic theoretical bound suggested by Proposition 4, effectively stabilizing at a rate determined by the second largest eigenvalue lambda_2=0.61, which is well below the upper limit of 1-m = 0.85.\n\n")
    return


def analyze_graph(filename, m=0.15):
    
    A, labels = read_dat(filename)
    if A is None: return None, None
    
    n = A.shape[0]
    s = np.ones(n) / n
    is_hollins = filename == "hollins.dat"
    output_file = None
    if is_hollins:
        output_file = open("hollins_results.txt", "w", encoding="utf-8")
        output = output_file
    else:
        output = sys.stdout
    print(f"\nGraph {filename}", file=output)
    x = power_iteration_with_vector(A, s, m, output)
    dangling = check_dangling_nodes(A)
    if dangling:
        print(f"  - Warning: Found {len(dangling)} dangling node(s): {[labels[i+1] for i in dangling]}", file=output)
        print(f"    (These nodes have initial importance score ≈ {m/n:.6f})", file=output)
        if filename == "Homework_1/graph1_modified.dat":
            exercise_4_analysis(A, labels)
            if output_file:
                output_file.close()
            return
    else:
        print(f"  - No dangling nodes detected", file=output)
    
    sorted_indices = np.argsort(x)[::-1]
    print(f"PageRank scores (sorted by importance):", file=output)
    print(f"{'-'*50}", file=output)
    # Output limited to top 20 if graph is huge, but here we keep full for small graphs
    for rank, idx in enumerate(sorted_indices, 1):
        node_label = labels[idx + 1]
        score = x[idx]
        print(f"  {rank}. {node_label:20s}: {score:.6f}", file=output)
    print("\n" + "="*70, file=output)
    
    if output_file:
        output_file.close()
        print(f"\n{filename} results saved to hollins_results.txt")
    
    return x, labels

def main():
    file_names = ["graph1.dat", "graph2.dat", "graph1_modified.dat", "hollins.dat"]
    for filename in file_names:
        analyze_graph(filename, m=m)
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_6("exercise2_graph.dat",2,3)
    exercise_7_stochastic_proof("exercise2_graph.dat")
    exercise_11()
    exercise_12()
    exercise_13()
    exercise_14()
    return

#Exercise 5
'''
Since the importance score is defined:
    x[k]= sum over j of x[j]/n[j] where j are the nodes that point to k and n[j] is the number of outbound links from node j.
So if a node has no backlinks, the sum is over an empty set and so x[k]=0.
'''

#Exercise 8
"""
Formal Proof:
    Let A and B be two n x n column-stochastic matrices. 
    Let C = AB be their product.
    We need to show that the sum of the elements in the j-th column of C is 1.
    The element C_ij is defined by the matrix multiplication:
    C_ij = Sum_k( A_ik * B_kj )
    The sum of the j-th column of C is:
    Sum_i(C_ij) = Sum_i [ Sum_k( A_ik * B_kj ) ]
    Swap the order of summation (Fubini's theorem for finite sums):
    Sum_i(C_ij) = Sum_k [ Sum_i( A_ik * B_kj ) ]
    Since B_kj is constant with respect to the index i, we can factor it out of the inner sum:
    Sum_i(C_ij) = Sum_k [ B_kj * Sum_i( A_ik ) ]
    Since A is column-stochastic, the sum of the elements in the k-th column of A is 1:
    Sum_i( A_ik ) = 1
    Substitute this into the expression:
    Sum_i(C_ij) = Sum_k [ B_kj * (1) ]
                = Sum_k( B_kj )
    Since B is column-stochastic, the sum of the elements in the j-th column of B is 1:
    Sum_k( B_kj ) = 1
    Conclusion: Sum_i(C_ij) = 1. Therefore, the product matrix C = AB is also column-stochastic.
"""

#Exercise 9
'''
Since the importance score is defined:
    x[i]= (1-m)*(Ax)[i] + m*s
In a node with no backlinks A[i] is a column with only 0, so (1-m)*A[i]*x[i] is equal to 0. s is a column in which all of his values is  1/n.
So the importance score for a node with no backlinks is m/n.
'''

# Exercise 16
'''
1. Eigenvalues (Eigvals) of A:
The characteristic polynomial of A is p_A(lambda) = -1/4 * (lambda - 1) * (2*lambda + 1)^2.
The eigenvalues of A are:
  lambda_A1 = 1           (Algebraic Multiplicity, m.a. = 1)
  lambda_A2 = -1/2        (Algebraic Multiplicity, m.a. = 2)

2. Eigenvalues of M:
M is a column-stochastic matrix, so lambda_M1 = 1 is an eigenvalue. The remaining eigenvalues are scaled versions of A's other eigenvalues: lambda_Mi = (1 - m) * lambda_Ai for i >= 2.
The eigenvalues of M are:
  lambda_1 = 1            (m.a. = 1)
  lambda_star = -(1 - m) / 2  (m.a. = 2)
Since 0 <= m < 1, lambda_star is a real, repeated eigenvalue, and lambda_star != 1.

3. Geometric Multiplicity (m.g.) of lambda_star:
For M to be diagonalizable, we require the geometric multiplicity m.g.(lambda_star) to equal the algebraic multiplicity, m.a.(lambda_star) = 2.
The geometric multiplicity is calculated as m.g.(lambda_star) = 3 - rank(D), where D = M - lambda_star * I.
We therefore need rank(D) = 1.

The matrix D can be written as D = (1 - m)B + mS, where B = A + (1/2)I.

a. Column Dependency (Upper Bound on Rank):
Since the columns of S are identical (s1 = s2 = s3) and the last two columns of B are identical (b2 = b3), the columns of D satisfy d3 = (1-m)b3 + ms3 = (1-m)b2 + ms2 = d2.
Since d3 = d2, the columns are linearly dependent, so rank(D) <= 2.

b. Linear Independence of d1 and d2 (Lower Bound on Rank):
To have rank(D) = 1, the first two columns, d1 and d2, must also be linearly dependent (d1 = k * d2).
By comparing the first components of d1 and d2, we find that the only possible proportionality constant is k=1.
However, comparing the second components with k=1 leads to:
d1[1] = m/3
d2[1] = (1-m)/2 + m/3
If d1 = d2, then m/3 = (1-m)/2 + m/3, which implies 0 = (1-m)/2, or m=1.
Since the problem states 0 <= m < 1, the columns d1 and d2 are linearly INDEPENDENT.

Conclusion on Rank:
Since d1 and d2 are independent, but d3 = d2, the rank of D is 2.
rank(M - lambda_star * I) = 2.

4. Final Conclusion:
m.g.(lambda_star) = 3 - rank(D) = 3 - 2 = 1.
Since m.g.(lambda_star) = 1 is strictly less than m.a.(lambda_star) = 2, the matrix M is NOT diagonalizable for 0 <= m < 1.
'''

# Exercise 17
'''
-  EFFECT ON COMPUTATION TIME:
   The PageRank is computed using the Power Method. The speed of convergence 
   of this method is determined by the magnitude of the second largest 
   eigenvalue, |lambda_2|.
   For the Google matrix M = (1-m)A + mS, it is proven that:
       |lambda_2|<= 1 - m.
   - Large m (close to 1): The factor (1-m) is small. Convergence is VERY FAST.
   - Small m (close to 0): The factor (1-m) is close to 1. Convergence is SLOW.

-  EFFECT ON RANKINGS:
   The parameter m controls the balance between the actual link structure (A) 
   and the random noise (S).
   - If m = 1: M = S. The link structure is ignored. All pages get the 
     exact same importance score (1/n). This is the "egalitarian" case.
   - If m = 0: M = A. The ranking relies purely on links. However, this 
     causes issues with non-unique rankings in disconnected webs and 
     dangling nodes.
     
CONCLUSION:
The choice of m = 0.15 by Google is a trade-off. It is small enough to ensure the 
rankings reflect the true importance of pages based on backlinks, but 
large enough to ensure the algorithm converges quickly (fast computation) 
and handles disconnected components correctly.
'''

# Exercise 15
'''
Hypotheses:
1. M is an n x n, positive, and column-stochastic matrix.
2. M is diagonalizable.
3. {q, v₁, ..., vₙ₋₁} is the basis of eigenvectors, where q is the stationary 
   eigenvector associated with λ=1.
4. x₀ is the initial probability vector (sum of components = 1).
5. Spectral Expansion: x₀ = a*q + sum(bₖ*vₖ)

PART 1: Determine M^k * x₀
Applying the matrix M iteratively:
M^k * x₀ = M^k (a*q + sum(bⱼ*vⱼ))
M^k * x₀ = a*M^k q + sum(bⱼ*M^k vⱼ)
Since M^k * q = q and M^k * vⱼ = λⱼ^k * vⱼ:
M^k * x₀ = a*q + sum_{j=1}^{n-1} bⱼ * λⱼ^k * vⱼ

PART 2: Prove a=1 and sum(vⱼ)=0
Let e be the column vector of all ones (e = [1, 1, ..., 1]^T). The sum of a 
vector's components is e^T * v.

A) Sum of components of vⱼ (for λⱼ ≠ 1):
Since M is column-stochastic, e^T * M = e^T.
M * vⱼ = λⱼ * vⱼ
Multiply by e^T: e^T * M * vⱼ = λⱼ * e^T * vⱼ
e^T * vⱼ = λⱼ * (e^T * vⱼ)
(1 - λⱼ) * (e^T * vⱼ) = 0
Since λⱼ ≠ 1 (for the non-stationary eigenvectors), we must have e^T * vⱼ = 0.
CONCLUSION: The sum of the components of every non-stationary eigenvector vⱼ is zero.

B) Determination of a:
Sum of components of x₀: e^T * x₀ = 1 (by hypothesis).
e^T * x₀ = a * (e^T * q) + sum_{j=1}^{n-1} bⱼ * (e^T * vⱼ)
1 = a * (1) + sum_{j=1}^{n-1} bⱼ * (0) (Since e^T * q = 1 and e^T * vⱼ = 0)
CONCLUSION: a = 1.

PART 3: Prove |λⱼ| < 1 (for λⱼ ≠ 1) using Proposition 4
Proposition 4 states that for any vector v with sum(v)=0, ||Mv||₁ <= c ||v||₁ where c < 1.
Since every vⱼ has sum(vⱼ)=0, we apply the proposition:
||M vⱼ||₁ <= c ||vⱼ||₁
||λⱼ vⱼ||₁ <= c ||vⱼ||₁
|λⱼ| * ||vⱼ||₁ <= c ||vⱼ||₁
Since ||vⱼ||₁ > 0, dividing by it yields: |λⱼ| <= c.
CONCLUSION: Because c < 1, we must have |λⱼ| < 1.

PART 4: Evaluation of the Limit Ratio
We want to calculate: L = lim_{k→∞} (||M^k * x₀ - q||₁ / ||M^{k-1} * x₀ - q||₁)

Subtracting q (knowing a=1) gives the error vector:
M^k * x₀ - q = sum_{j=1}^{n-1} bⱼ * λⱼ^k * vⱼ

As k → ∞, the sum is dominated by the term corresponding to the largest non-unit 
eigenvalue in magnitude, denoted as λ₂:
M^k * x₀ - q ≈ b₂ * λ₂^k * v₂
M^{k-1} * x₀ - q ≈ b₂ * λ₂^{k-1} * v₂

Substituting into the limit ratio:
L ≈ ( ||b₂ * λ₂^k * v₂||₁ / ||b₂ * λ₂^{k-1} * v₂||₁ )
L ≈ ( |b₂| * |λ₂|^k * ||v₂||₁ ) / ( |b₂| * |λ₂|^{k-1} * ||v₂||₁ )
L = |λ₂|

FINAL CONCLUSION: The limit of the ratio is |λ₂|, proving that the asymptotic 
rate of convergence of the Power Method is determined by the magnitude of the 
second dominant eigenvalue.
'''


if __name__ == "__main__":
    main()