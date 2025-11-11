import numpy as np

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
                A[:,i]=A[:,i]/count
                    
    except FileNotFoundError:
        print(f"Error: File '{file_name}' not found.")
        return None, None
    except Exception as e:
        print(f"Errore during the analysis of the file: {e}")
        return None, None
    
    return A, labels

def build_S_columnvector(n):
    S=np.ones((n,1))/n
    return S

def find_eigenvector(A,S,m):
    x=[1,1,1,1]
    return x

def graph1():
    filename="graph1.dat"
    m=0.15
    A,labels =read_dat(filename)
    S=build_S_columnvector(int(A.shape[0]))
    x=find_eigenvector(A,S,m)
    sorted_index=np.argsort(x)
    sorted_index=sorted_index[::-1]
    print(f"The importance score for the graph {filename} is:")
    for i in range(A.shape[0]):
        print(f"{labels[sorted_index[i]+1]}: {x[sorted_index[i]]}")
    return 

def graph2():
    filename="graph2.dat"
    m=0.15
    A,labels =read_dat(filename)
    S=build_S_columnvector(int(A.shape[0]))
    x=find_eigenvector(A,S,m)
    sorted_index=np.argsort(x)
    sorted_index=sorted_index[::-1]
    print(f"The importance score for the graph {filename} is:")
    for i in range(A.shape[0]):
        print(f"{labels[sorted_index[i]+1]}: {x[sorted_index[i]]}")
    return 

def graph1_modified():
    filename="graph1_modified.dat"
    m=0.15
    A,labels =read_dat(filename)
    S=build_S_columnvector(int(A.shape[0]))
    x=find_eigenvector(A,S,m)
    sorted_index=np.argsort(x)
    sorted_index=sorted_index[::-1]
    print(f"The importance score for the graph {filename} is:")
    for i in range(A.shape[0]):
        print(f"{labels[sorted_index[i]+1]}: {x[sorted_index[i]]}")
    return

def main():
    graph1()
    graph2()
    graph1_modified()
    return

# Exercise 4

# Exercise 9

main()
