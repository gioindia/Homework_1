import numpy as np
m=0.15

# Implement PagesRank + exercises 4, 9

# Algoritm

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

def power_iteration_with_vector(A, s, m, tolerance=1e-6, max_iterations=1000):
    n = A.shape[0]
    x = np.ones(n) / n
    for iteration in range(max_iterations):
        x_new = (1 - m) * (A @ x) + m * s
        x_new = x_new / np.sum(x_new) # normalized
        if np.linalg.norm(x_new - x, 1) < tolerance:
            print(f"  Converged in {iteration + 1} iterations")
            break
        x = x_new
    else:
        print(f"  Warning: Maximum iterations ({max_iterations}) reached")
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

def analyze_graph(filename, m=0.15):
    A, labels = read_dat(filename)
    n = A.shape[0]
    s = np.ones(n) / n
    print(f"\nGraph {filename}")
    x = power_iteration_with_vector(A, s, m)
    dangling = check_dangling_nodes(A)
    if dangling:
        print(f"  - Warning: Found {len(dangling)} dangling node(s): {[labels[i+1] for i in dangling]}")
        print(f"    (These nodes have importance score ≈ {m/n:.6f})")
        if filename=="Homework_1/graph1_modified.dat":
            exercise_4_analysis(A, labels)
            return
    else:
        print(f"  - No dangling nodes detected")
    sorted_indices = np.argsort(x)[::-1]
    print(f"PageRank scores (sorted by importance):")
    print(f"{'-'*50}")
    for rank, idx in enumerate(sorted_indices, 1):
        node_label = labels[idx + 1]
        score = x[idx]
        print(f"  {rank}. {node_label:20s}: {score:.6f}")
    print("\n" + "="*70)
    return x, labels

def main():
    file_names = ["Homework_1/graph1.dat", "Homework_1/graph2.dat", "Homework_1/graph1_modified.dat", "Homework_1/hollins.dat"]
    for filename in file_names:
        analyze_graph(filename, m=m)
    return

# TODO: Exercise 9

main()
