import numpy as np
import sys
m=0.15

def read_dat(file_name):
    labels = {}
    num_nodes = 0
    num_edges = 0
    try:
        with open(file_name, 'r') as file:
            first_line = file.readline().strip()
            parts = first_line.split()
            num_nodes = int(parts[0])
            num_edges = int(parts[1])
            
            A=np.zeros((num_nodes,num_nodes))

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
                    A[target-1][source-1]=1
                    
            for i in range(num_nodes):
                count=np.count_nonzero(A[:,i], axis=0)
                if count != 0:
                    A[:,i]=A[:,i]/count
                    
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
    n = A.shape[0]
    dangling = []
    for i in range(n):
        if np.sum(A[:, i]) == 0:
            dangling.append(i)
    return dangling

def exercise_4_analysis(A, labels):
    n = A.shape[0]
    eigenvalues, eigenvectors = np.linalg.eig(A)
    idx = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    perron_eigenvalue = eigenvalues[0]
    perron_eigenvector = eigenvectors[:, 0]
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
    print("\n" + "="*70)
    print("Exercise 1 Analysis:")
    #Graph 1
    eigenvalues, eigenvectors = np.linalg.eig(A)
    idx = np.where(np.isclose(eigenvalues, 1))[0][0]
    x_raw = np.real(eigenvectors[:, idx])
    importance_score = x_raw / x_raw.sum()
    
    #Graph 1 with node 5 added
    filename="exercise1_graph.dat"
    A_modified, labels_modified = read_dat(filename)
    eigenvalues, eigenvectors = np.linalg.eig(A_modified)
    idx = np.where(np.isclose(eigenvalues, 1))[0][0]
    x_raw = np.real(eigenvectors[:, idx])
    importance_score_withnode5 = x_raw / x_raw.sum()
    
    print("Importance scores for Graph 1:")
    sorted_indices = np.argsort(importance_score)[::-1]
    for rank, idx in enumerate(sorted_indices, 1):
        node_label = labels[idx + 1]
        score = importance_score[idx]
        print(f"  {rank}. {node_label:20s}: {score:.6f}")
    
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
    eigenvalues, eigenvectors = np.linalg.eig(A)
    dimension = np.sum(np.isclose(eigenvalues, 1))
    print(f"The dimension of the eigenspace associated with the eigenvalue 1 is: {dimension} >= of the number of the components in the web graph(4).")
    return

def exercise_3():
    filename="exercise3_graph.dat"
    print("\n" + "="*70)
    print("Exercise 3 Analysis:")
    A, labels = read_dat(filename) 
    eigenvalues, eigenvectors = np.linalg.eig(A)
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

    A_eigenvalues, A_eigenvectors = np.linalg.eig(A)
    x = A_eigenvectors[:, 0] # vector c chooosen for proving y=P·x
    l = A_eigenvalues[0] # corresponding eigenvalue

    # Create permutation matrix P that swaps nodes i and j
    P=np.eye(A.shape[0])
    P[[i-1,j-1]] = P[[j-1,i-1]]
    # Compute the theoretical permuted adjacency matrix A2_theoretical
    A2_theoretical = P @ A @ P

    # Apply the swap to the graph file
    print(f"Referring to graph {filename}, we swap the pages with indices i={i} and j={j}.")
    output_file = "exercise6_graph.dat"
    swap_node_indices(filename, output_file, i, j)
    A2, labels2 = read_dat(output_file)

    # Verify A2_theoretical == A2
    print("1)")
    print(f"Verifying that the permuted adjacency matrix A2_theoretical matches the matrix A2 obtained by applying PagesRank on the new graph:", end=" ") 
    for r in range(A2_theoretical.shape[0]):
        for c in range(A2_theoretical.shape[1]):
            if (A2_theoretical[r][c]!=A2[r][c]):
                print(f"Mismatch at position ({r},{c}): A2_theoretical={A2_theoretical[r][c]}, A2={A2[r][c]}")
                return
    print("the matrices match perfectly.")

    print("2)")
    # Verify that l is an eigenvalue of A2 and find corresponding eigenvector y
    A2_eigenvalues, A2_eigenvectors = np.linalg.eig(A2)
    print(f"Eigenvalue lambda={l:.6f} of A is an eigenvalue for A2 too: {l in A2_eigenvalues}.")
    idx = np.where(np.isclose(A2_eigenvalues, l))[0][0]
    y_found = A2_eigenvectors[:, idx]
    y_theoretical = P @ x
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

def check_matrix_stochastic(matrix, tol=1e-9):
    column_sums = np.sum(matrix, axis=0)
    is_one = np.isclose(column_sums, 1.0, atol=tol)
    is_stochastic = np.all(is_one)
    return is_stochastic

def exercise_7_stochastic_proof(filename):
    print("\n" + "="*70)
    print("Exercise 6 Analysis:")
    m=0.15
    A, labels = read_dat(filename)
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
    S = np.ones((n, n)) / n
    M = (1 - m) * A + m * S
    print(f"Matrix M is column-stochastic: {check_matrix_stochastic(M)}")
    return

def exercise_11():
    print("Exercise 11 Analysis:")
    analyze_graph("exercise11_graph.dat", m=0.15)
    return

def exercise_12():
    print("Exercise 12 Analysis:")
    A,labels = read_dat("exercise12_graph.dat")
    eigenvalues, eigenvectors = np.linalg.eig(A)
    idx = np.where(np.isclose(eigenvalues, 1))[0][0]
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
    print("Exercise 13 Analysis:")
    analyze_graph("exercise13_graph.dat", m=0.15)
    print("The analysis using matrix M shows that the isolated pair (Nodes 6-7) outranks the peripheral nodes of the larger cluster (Nodes 2-5). This demonstrates that out-degree dilution (x_1/4) significantly weakens the authority transferred by the central hub compared to the undiluted reciprocity (x_j/1) retained within the smaller clique.\n\n")
    return


def analyze_graph(filename, m=0.15):
    A, labels = read_dat(filename)
    n = A.shape[0]
    s = np.ones(n) / n
    is_hollins = filename == "hollins.dat"
    output_file = None
    if is_hollins:
        output_file = open("hollins_results.txt", "w", encoding="utf-8")
        output = output_file
    else:
        import sys
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


main()
