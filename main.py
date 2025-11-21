import numpy as np
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
    x = np.ones(n) / n
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
    return



#Exercise 5
'''
Since the importance score is defined:
    x[k]= sum over j of x[j]/n[j] where j are the nodes that point to k and n[j] is the number of outbound links from node j.
So if a node has no backlinks, the sum is over an empty set and so x[k]=0.
'''

#Exercise 9
'''
Since the importance score is defined:
    x[i]= (1-m)*(Ax)[i] + m*s
In a node with no backlinks A[i] is a column with only 0, so (1-m)*A[i]*x[i] is equal to 0. s is a column in which all of his values is  1/n.
So the importance score for a node with no backlinks is m/n.
'''


main()
