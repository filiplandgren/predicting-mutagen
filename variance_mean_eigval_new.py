# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 11:57:34 2022

@author: fl5g21
"""

import pickle
import networkx as nx
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


dataset = pickle.load(open('mutag_processed.pkl', 'rb'))

def adjacency_ith_molecule(i):
    G = nx.Graph()  # create a graph object
    list_edg = []   # create empty list of edges
    list_bond_type = [] # create empty list of bond types
    mutagen = dataset[i]['class'] # YES if 1, NO if 0
    
    # Running through edge_list entry
    for elem in dataset[i]['edge_list']:
        list_edg.append(elem[0])  # add each edge
        list_bond_type.append(elem[1])
        
    G.add_edges_from(list_edg)  # add edges to the graph object
    
    #### BOND TYPES
    nx.set_edge_attributes(G, values = 0, name = 'bond_type')  # add null bond type values
    for j, e in enumerate(list(G.edges())):
        u, v = e[0], e[1]
        G[u][v]['bond_type']= list_bond_type[j] # adding bond types
    
    
    #### VERT TYPES
    vert_type = list(dataset[i]['vert_type']) # atoms types
    nx.set_node_attributes(G, values = 0, name = 'vert_type') # add null vert types
    for k, node in enumerate(list(G.nodes(data=True))):
        node[1]['vert_type'] = vert_type[k]
        
        
    #### ADJACENCY MATRIX
    A = nx.adjacency_matrix(G)  # this gives a sparse matrix object
    A_dense = A.todense() # this gives a dense matrix-type object
    A_array = np.array(A_dense) # this gives an array-type adjacency matrix
    
    #### EIGENVALUES OF A 
    A_evalues, A_evectors = np.linalg.eig(A_array)
    
    #### Laplacian MATRIX
    L = nx.laplacian_matrix(G)  # this gives a sparse matrix object
    L_dense = L.todense() # this gives a dense matrix-type object
    L_array = np.array(L_dense) # this gives an array-type adjacency matrix
    #### EIGENVALUES OF A 
    L_evalues, L_evectors = np.linalg.eig(L_array)

    return G, A_array, L_array, A_evalues, L_evalues, mutagen



no_mols = len(dataset) #Lenghth of dataset


mutagen_list = [] #List with mutagen atoms
vertices = [] 
edges = []
cycle_no=[]
quant_L = [] #Quantiles Laplacian matrix 0.25, 0.5, 0.75
quant_A = [] #Quantiles Agencency matrix
mean_L = [] #Mean of Laplacian matrix
var_L = [] #Variance of Laplacian matrix
var_A = [] #Variance of Agencency matrix

#Sort above quantities into mutagen vs normal (not mutagen)
L_eigval_mut=[] 
L_eigval_nor=[]
A_eigval_mut=[]
A_eigval_nor=[]
L_aver_mut=[]
L_aver_nor=[]
L_var_mut=[]
L_var_nor=[]
A_var_mut=[]
A_var_nor=[]



for i in range(np.size(dataset)):
    G, A_array, L_array, A_evalues, L_evalues, mutagen = adjacency_ith_molecule(i)
    L_aver = np.mean(L_evalues, dtype=float)
    L_var = np.var(L_evalues, dtype=float)
    A_var = np.var(A_evalues, dtype=float)
    
    if dataset[i]['class']==1:
        L_eigval_mut.append(L_evalues)
        A_eigval_mut.append(A_evalues)
        L_aver_mut.append(L_aver)
        L_var_mut.append(L_var)
        A_var_mut.append(A_var)
    else:
        L_eigval_nor.append(L_evalues)
        A_eigval_nor.append(A_evalues)
        L_aver_nor.append(L_aver)
        L_var_nor.append(L_var)
        A_var_nor.append(A_var)


#Second smallest Eigenvalue
L_second_small_mut = []
L_second_small_nor = []
for i in range(np.size(L_eigval_mut)):
    L_second_small_mut.append(sorted(L_eigval_mut[i])[1])
for i in range(np.size(L_eigval_nor)):
    L_second_small_nor.append(sorted(L_eigval_nor[i])[1])
    
    
    
### EXPORT DATA INTO CSV

tree_df = pd.DataFrame({'Mutagen': mutagen,
                   'No. of vertices': vertices,
                   'No. of edges': edges, 'No. of cycles': cycle_no, 'Q of L-eigenv distr': quant_L, 'Q of A-eigenv distr': quant_A, 'M of L-eigenv distr': mean_L, 'V of A-eigenv distr': var_A })

tree_df.to_csv('tree_data.csv')

    



### PLOT: MEAN VS 2ND SMALLEST EIGENVALUE


s = {' L_second_small_nor': L_second_small_nor, 'L_aver_nor': L_aver_nor}
ds = pd.DataFrame(data=s)
ds = ds.round(2)
ds = ds.value_counts().reset_index()
size = 200*np.array(ds[0])

s2 = {'L_second_small_mut': L_second_small_mut, 'L_aver_mut': L_aver_mut}
ds2 = pd.DataFrame(data=s2)
ds2 = ds2.round(2)
ds2 = ds2.value_counts().reset_index()
size2 = 200*np.array(ds2[0])


# PLOT: MEAN VS VARIANCE OF L MATRIX

d = {' L_var_nor': L_var_nor, 'L_aver_nor': L_aver_nor}
df = pd.DataFrame(data=d)
df = df.round(2)
df = df.value_counts().reset_index()
size = 200*np.array(df[0])

d2 = {'L_var_mut': L_var_mut, 'L_aver_mut': L_aver_mut}
df2 = pd.DataFrame(data=d2)
df2 = df2.round(2)
df2 = df2.value_counts().reset_index()
size2 = 200*np.array(df2[0])






# Plot function

 

def dicts_scatter_plot(dict_1, label_1, dict_2, label_2, x_axis_name, y_axis_name):

    """

    Creates a scatterplot from two dictionaries

    Each dictionary contain two keys

    Fig saved as 'fig_{x_axis_name}_vs_{y_axis_name}.png'

    """

 

    df_1 = pd.DataFrame(data=dict_1).round(2)

    df_1 = df_1.value_counts().reset_index()

    size = 200*np.array(df_1[0])

 

    df_2 = pd.DataFrame(data=dict_2).round(2)

    df_2 = df_2.value_counts().reset_index()

    size2 = 200*np.array(df_2[0])

 

    fig = plt.figure(2, figsize=(10,10))

    plt.scatter(np.array(df_1.iloc[:, 0]), np.array(df_1.iloc[:, 1]), color='green', alpha=0.3, s=size, label="class 0 (not a mutagen)")

    plt.scatter(np.array(df_2.iloc[:, 0]), np.array(df_2.iloc[:, 1]), color='red', alpha=0.3, s=size2, label="class 1 (mutagen)")

    plt.xlabel(x_axis_name, fontsize=15)

    plt.ylabel(y_axis_name, fontsize=16)

    #plt.yticks(np.arange(2, 2.5, 0.1), fontsize=12)

    #plt.xticks(np.arange(0, 0.4, 0.1), fontsize=12)

    lgnd = plt.legend(loc="upper right", fontsize=15)

    lgnd.legendHandles[0]._sizes = [150]

    lgnd.legendHandles[1]._sizes = [150]

   

    plt.savefig(f'fig_{x_axis_name}_vs_{y_axis_name}.png');

 

 

# Call function

 

### PLOT: MEAN VS 2ND SMALLEST EIGENVALUE

 

# dicts_scatter_plot(

#                     {' L_second_small_nor': L_second_small_nor, 'L_aver_nor': L_aver_nor},

#                     "class 0 (not a mutagen)",

#                     {'L_second_small_mut': L_second_small_mut, 'L_aver_mut': L_aver_mut},

#                     "class 1 (mutagen)",

#                     "2nd smallest eigenvalue",

#                     "mean"

#                   )

 

# PLOT: MEAN VS VARIANCE OF L MATRIX

dicts_scatter_plot(

                    {' L_var_nor': L_var_nor, 'L_aver_nor': L_aver_nor},

                    "class 0 (not a mutagen)",

                    {'L_var_mut': L_var_mut, 'L_aver_mut': L_aver_mut},

                    "class 1 (mutagen)",

                    "variance",

                    "mean"

                  )




