import numpy as np
import sys
from scipy import sparse
from scipy.sparse import linalg as splinalg

m = 0.15

def read_dat(file_name):
    labels = {}
    row_indices = [] # Row indices for sparse matrix construction
    col_indices = [] # Column indices for sparse matrix construction
    
    try:
        with open(file_name, 'r') as file:
            first_line = file.readline().strip()
            if not first_line:
                return None, None
            parts = first_line.split()
            num_nodes = int(parts[0])
            num_edges = int(parts[1])
            
            # Lettura nodi
            for _ in range(num_nodes):
                line = file.readline().strip()
                if line:
                    parts = line.split(maxsplit=1) 
                    node_id = int(parts[0])
                    node_name = parts[1]
                    labels[node_id] = node_name

            # Lettura archi per costruire matrice sparsa
            # Nota: PageRank uses A[target][source] = 1
            for _ in range(num_edges):
                line = file.readline().strip()
                if line:
                    parts = line.split()
                    source = int(parts[0])
                    target = int(parts[1])
                    # Salviamo le coordinate. A[riga][colonna]
                    row_indices.append(target - 1)
                    col_indices.append(source - 1)
            
            # Creazione matrice sparsa (valori tutti a 1 inizialmente)
            data = np.ones(len(row_indices))
            # Use COO format for fast construction, then convert to CSC (Compressed Sparse Column) 
            # for efficient column operations (needed for PageRank normalization).
            A = sparse.coo_matrix((data, (row_indices, col_indices)), shape=(num_nodes, num_nodes)).tocsc()
            
            # Normalizzazione colonne (Stochastic Matrix)
            # Calcoliamo la somma di ogni colonna
            col_sums = np.array(A.sum(axis=0)).flatten()
            
            # Evitiamo divisione per zero per i nodi dangling (quelli con somma 0 rimangono 0 per ora)
            # Create scaling factors: 1/sum if sum != 0, otherwise 0
            with np.errstate(divide='ignore', invalid='ignore'):
                scale_factors = np.where(col_sums != 0, 1.0 / col_sums, 0)
            
            # Moltiplicazione efficiente per la matrice diagonale dei fattori di scala
            # A_new = A @ D^(-1)
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
        # A is sparse, but the multiplication @ with dense vector x returns a dense vector
        # PageRank formula iteration: x_new = (1-m)Ax + ms
        ax = A @ x
        x_new = (1 - m) * ax + m * s
        
        # Normalizzazione L1
        x_new = x_new / np.sum(x_new) 
        
        if np.linalg.norm(x_new - x, 1) < tolerance:
            print(f"  Converged in {iteration + 1} iterations", file=output)
            break
        x = x_new
    else:
        print(f"  Warning: Maximum iterations ({max_iterations}) reached", file=output)
    return x

def check_dangling_nodes(A):
    # Sum axis=0 is efficient for CSC matrix
    col_sums = np.array(A.sum(axis=0)).flatten()
    # Dangling nodes have a column sum of 0 (considering float tolerance)
    dangling = np.where(np.isclose(col_sums, 0))[0]
    return dangling.tolist()

def get_eigenpairs(A, k=None):
    """
    Helper function to get eigenvalues/vectors.
    Uses scipy.sparse.linalg.eigs for large matrices, numpy.linalg.eig for small ones.
    """
    n = A.shape[0]
    # Scipy eigs requires k < n-1, so choose a reasonable k.
    if k is None: 
        k = min(n - 2, 6) 
        if k < 1: k = 1

    # If the matrix is very small or k is too close to n, use dense numpy calculation
    if n < 10 or k >= n - 1:
        dense_A = A.toarray()
        evals, evecs = np.linalg.eig(dense_A)
    else:
        # 'LM' = Largest Magnitude
        try:
            evals, evecs = splinalg.eigs(A, k=k, which='LM')
        except:
            # Fallback to dense if sparse convergence fails
            evals, evecs = np.linalg.eig(A.toarray())
            
    return evals, evecs

def exercise_4_analysis(A, labels):
    n = A.shape[0]
    # Use the helper to get eigenvalues
    eigenvalues, eigenvectors = get_eigenpairs(A)
    
    idx = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Get the Perron eigenvalue (largest magnitude)
    perron_eigenvalue = np.real(eigenvalues[0])
    perron_eigenvector = np.real(eigenvectors[:, 0])
    
    print(f"PERRON EIGENVALUE (largest): λ = {perron_eigenvalue:.6f}")
    
    # Ensure non-negativity (Perron-Frobenius theorem) and normalize
    if np.any(perron_eigenvector < 0):
        perron_eigenvector = -perron_eigenvector

    perron_eigenvector = perron_eigenvector / np.sum(perron_eigenvector)
    
    print(f"\nPerron eigenvector (normalized to sum=1):")
    sorted_indices = np.argsort(perron_eigenvector)[::-1]
    print(f"{'-'*50}")
    for rank, idx in enumerate(sorted_indices, 1):
        node_label = labels[idx + 1]
        score = perron_eigenvector[idx]
        print(f"  {rank}. {node_label:20s}: {score:.6f}")    
    
    # Verification of eigenvector property: A*v = lambda*v
    result = A @ perron_eigenvector
    expected = perron_eigenvalue * perron_eigenvector
    error = np.linalg.norm(result - expected)
    print(f"\nVerification: ||A·v - λ·v|| = {error:.2e}")
    return perron_eigenvalue, perron_eigenvector

def exercise_1():
    filename="graph1.dat"
    # Reading returns sparse matrix
    A, labels = read_dat(filename)
    if A is None: return

    print("\n" + "="*70)
    print("Exercise 1 Analysis:")
    
    # Graph 1
    vals, vecs = get_eigenpairs(A)
    # Search for eigenvalue ~1
    idx_list = np.where(np.isclose(vals, 1))[0]
    if len(idx_list) > 0:
        idx = idx_list[0]
        x_raw = np.real(vecs[:, idx])
        importance_score = x_raw / x_raw.sum()
        
        print("Importance scores for Graph 1:")
        sorted_indices = np.argsort(importance_score)[::-1]
        for rank, idx in enumerate(sorted_indices, 1):
            node_label = labels[idx + 1]
            score = importance_score[idx]
            print(f"  {rank}. {node_label:20s}: {score:.6f}")
    
    # Graph 1 with node 5
    filename="exercise1_graph.dat"
    A_modified, labels_modified = read_dat(filename)
    if A_modified is not None:
        vals, vecs = get_eigenpairs(A_modified)
        idx_list = np.where(np.isclose(vals, 1))[0]
        if len(idx_list) > 0:
            idx = idx_list[0]
            x_raw = np.real(vecs[:, idx])
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

    # We need to find the multiplicity of eigenvalue 1, which corresponds to the number of 
    # closed strongly connected components in the graph.
    vals, _ = get_eigenpairs(A, k=min(A.shape[0]-2, 10)) 
    
    dimension = np.sum(np.isclose(vals, 1))
    print(f"The dimension of the eigenspace associated with the eigenvalue 1 is: {dimension} >= of the number of the components in the web graph(4).")
    return

def exercise_3():
    filename="exercise3_graph.dat"
    print("\n" + "="*70)
    print("Exercise 3 Analysis:")
    A, labels = read_dat(filename) 
    if A is None: return

    vals, _ = get_eigenpairs(A, k=min(A.shape[0]-2, 10))
    dimension = np.sum(np.isclose(vals, 1))
    print(f"The dimension of the eigenspace associated with the eigenvalue 1 is: {dimension} because the web contains two closed strongly connected components. Indeed from the node group {{1,2}} we can't reach the node group {{3,4,5}} and from the node group {{3,4}} we can't reach the node group {{1,2}}.")
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
            if k == 0: # Header
                new_lines.append(line + '\n')
                continue
            modified_parts = []
            # Swap indices in the file content (nodes and links)
            for part in parts:
                if part.isdigit():
                    if part == str_i: modified_parts.append(str_j)
                    elif part == str_j: modified_parts.append(str_i)
                    else: modified_parts.append(part)
                else:
                    modified_parts.append(part)
            new_lines.append(" ".join(modified_parts) + '\n')

        with open(output_filename, 'w') as outfile:
            outfile.writelines(new_lines)
        print(f"Indecises {i} and {j} swapped (Nodes and Links). Result saved in '{output_filename}'.")
    except Exception as e:
        print(f"ERROR: file '{input_filename}' not found or other error: {e}")
    return output_filename

def exercise_6(filename, i, j):
    print("\n" + "="*70)
    print("Exercise 6 Analysis:")
    A, labels = read_dat(filename)
    if A is None: return

    vals, vecs = get_eigenpairs(A)
    # Assume the first returned pair is the dominant one (or the one of interest)
    x = vecs[:, 0] 
    l = vals[0]

    # Create permutation matrix P that swaps nodes i and j (Sparse LIL for easy manipulation)
    n = A.shape[0]
    P = sparse.eye(n, format='lil')
    P[i-1, i-1] = 0
    P[j-1, j-1] = 0
    P[i-1, j-1] = 1
    P[j-1, i-1] = 1
    P = P.tocsc() # Convert back to CSC for efficient multiplication
    
    # Compute the theoretical permuted adjacency matrix A2_theoretical
    A2_theoretical = P @ A @ P

    print(f"Referring to graph {filename}, we swap the pages with indices i={i} and j={j}.")
    output_file = "exercise6_graph.dat"
    swap_node_indices(filename, output_file, i, j)
    A2, labels2 = read_dat(output_file)

    print("1) Verifying that the permuted adjacency matrix A2_theoretical matches the matrix A2 obtained from the new graph file:")
    # Check if the difference matrix is the zero matrix
    diff = (A2_theoretical - A2)
    if np.allclose(diff.toarray(), 0):
        print("the matrices match perfectly.")
    else:
        print("Mismatch found.")


    print("2) Verify the theorem: λ is an eigenvalue of A2, and y = Px is the corresponding eigenvector.")
    vals2, vecs2 = get_eigenpairs(A2)
    
    # Check eigenvalue match
    match_val = np.isclose(vals2, l).any()
    print(f"Eigenvalue lambda={np.real(l):.6f} of A is an eigenvalue for A2 too: {match_val}.")
    
    if match_val:
        idx = np.where(np.isclose(vals2, l))[0][0]
        y_found = vecs2[:, idx]
        y_theoretical = P @ x
        
        # Normalize to check collinearity (eigenvectors are only determined up to a scalar factor)
        y_found = y_found / np.linalg.norm(y_found)
        y_theoretical = y_theoretical / np.linalg.norm(y_theoretical)
        
        # Check if vectors are equal up to a sign flip (np.abs handles this)
        match_pos = np.allclose(np.abs(y_found), np.abs(y_theoretical)) 
        print(f"Verifying that y corresponding to l is equal to P·x (or -P·x): {match_pos}.")    

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

def check_matrix_stochastic(matrix_sparse, tol=1e-9):
    # Sum axis=0 for CSC matrix
    column_sums = np.array(matrix_sparse.sum(axis=0)).flatten()
    is_one = np.isclose(column_sums, 1.0, atol=tol)
    is_stochastic = np.all(is_one)
    return is_stochastic

def exercise_7_stochastic_proof(filename):
    print("\n" + "="*70)
    print("Exercise 7: Proof that M = (1-m)A + mS is column-stochastic.")
    m=0.15
    A, labels = read_dat(filename)
    if A is None: return
    n = A.shape[0]
    
    """
    Formal Proof:
    1. A is column-stochastic (by design): Sum_i(A_ij) = 1 for all j (or 0 for dangling nodes, but A' is used here).
    2. S is column-stochastic: S_ij = 1/n, so Sum_i(S_ij) = n * (1/n) = 1 for all j.

    Sum of the j-th column of M:
    Sum_i(M_ij) = Sum_i[ (1-m)A_ij + mS_ij ]
                = (1-m) * Sum_i(A_ij) + m * Sum_i(S_ij)  (By linearity)
                = (1-m) * (1) + m * (1)                 (Substituting the known sums, assuming A is fully stochastic for non-dangling)
                = 1 - m + m = 1

    Conclusion: Since the sum of every column of M is 1, M is column-stochastic.
    """
    
    # 2. Numerical Verification (Using the input matrix A)
    print(f"Numerical Verification (using graph: {filename}):")
    # M = (1-m)A + mS
    # We check the column sums: Sum(M) = (1-m)Sum(A) + m*Sum(S)
    sum_A = np.array(A.sum(axis=0)).flatten()
    # Sum(S) is always 1 for all columns
    sum_M = (1 - m) * sum_A + m * 1.0
    
    # The sum should be 1.0 for all columns
    print(f"Matrix M is column-stochastic: {np.allclose(sum_M, 1.0)}")
    
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
        dangling_labels = [labels[i+1] for i in dangling]
        print(f"  - Warning: Found {len(dangling)} dangling node(s): {dangling_labels}", file=output)
        print(f"    (These nodes have initial importance score ≈ {m/n:.6f})", file=output)
        if filename == "Homework_1/graph1_modified.dat":
            exercise_4_analysis(A, labels)
            if output_file: output_file.close()
            return x, labels
    else:
        print(f"  - No dangling nodes detected", file=output)
    
    sorted_indices = np.argsort(x)[::-1]
    print(f"PageRank scores (sorted by importance):", file=output)
    print(f"{'-'*50}", file=output)
    # Show top 20 if many nodes, otherwise all nodes
    limit = 20 if n > 50 else n
    for rank, idx in enumerate(sorted_indices[:limit], 1):
        node_label = labels[idx + 1]
        score = x[idx]
        print(f"  {rank}. {node_label:20s}: {score:.6f}", file=output)
    print("\n" + "="*70, file=output)
    
    if output_file:
        output_file.close()
        print(f"\n{filename} results saved to hollins_results.txt")
    
    return x, labels

def main():
    # File names to analyze. Ensure these files exist or comment them out.
    file_names = ["graph1.dat", "graph2.dat", "exercise1_graph.dat", "hollins.dat"]
    
    for filename in file_names:
        analyze_graph(filename, m=m)
        
    # Exercise functions execution
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_6("exercise2_graph.dat", 2, 3)
    exercise_7_stochastic_proof("exercise2_graph.dat")
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

if __name__ == "__main__":
    main()