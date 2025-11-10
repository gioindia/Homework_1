import numpy as np

# Implement PagesRank + exercises 4, 9

# Algoritm

def read_dat(filename):
    A=[[1,2],[3,4]]
    return A

def build_S_matrix(n):
    S=[]
    for i in range(1,n):
        for j in range(1,n):
            S[i]=[]
            S[i][j]=1/n
    return S

def find_eigenvector(M):
    x=[1,1,1]
    return x

def main():
    filename="graph1.dat"
    m=0.15
    A=read_dat(filename)
    S=build_S_matrix(len(A[0]))
    M=(1-m)*A+m*S
    x=find_eigenvector(M)
    print(f"The eigenvcetor for the graph {filename} is: {x}")
    return

# Exercise 4

# Exercise 9

main()
