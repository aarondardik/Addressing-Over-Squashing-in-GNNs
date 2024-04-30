##Note that while this is based generally off the stochastoic descent algorithm from the Over-Squashing paper there are substantial differences:
#1) that paper doesn't include code anyways, 2) this algorithm doesn't follow theirs step for step...when I didn't know something
#or something was slow I just fiddled with it. 

import math
import random
import numpy as np 
import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from GraphRicciCurvature.FormanRicci import FormanRicci
import matplotlib.pyplot as plt



#This function computes the Forman curvature for a single edge. While this function is not a 'theoretical breakthrough'
#the GraphRicciCurvature library (and to our knowledge any other library implementing similar techniques) compute the
#Forman curvature for all edges at once - which is not always necessary. The option to calculate it for a given edge
#is more efficient than recomputing the curvature for the entire graph whenever an edge is rewired.
def compute_forman_single_edge(G, vertex1, vertex2):
    v1_v2_edge = ((vertex1, vertex2) in G.edges())
    
    forman = 2
    if v1_v2_edge:
        return forman - G.degree[vertex1] - G.degree[vertex2] + 2
    else:
        return forman - G.degree[vertex1] - G.degree[vertex2]

    
#Given an edge represented by vertices vertex1 and vertex2 (v1 and v2) the function determines the edge that should be added 
#between a vertex in N(v1) and a vertex in N(v2) that maximally increases the curvature - i.e. the local change that would,
#amongst single edge changes, change the local regime 'most' from hyperbolic to spherical.
def formanImprovement(G, vertex1, vertex2):
    try:
        v1_neighbors = list(G.neighbors(vertex1))
    except nx.exception.NetworkXError:
        print("In the error case the vertices were vertex 1: {} and vertex 2: {}. Their types are {} and {} respectively".format(vertex1, vertex2, type(vertex1), type(vertex2)))
    try:
        v1_neighbors = list(G.neighbors(vertex1))
        v1_neighbors.remove(vertex2)
    except ValueError:
        print("{} is not a neighbor of {} and therefore cannot be removed.".format(vertex2, vertex1))
    v2_neighbors = list(G.neighbors(vertex2))
    try:
        v2_neighbors.remove(vertex1)
    except ValueError:
        print("{} is not a neighbor of {} and therefore cannot be removed.".format(vertex1, vertex2))
    
    
    if (len(v1_neighbors) == 0 or len(v2_neighbors) == 0 ):
        return vertex1, vertex2 
    try:
        w, h = len(v2_neighbors), len(v1_neighbors)
        #improvementArr = [len(v1_neighbors)][len(v2_neighbors)]
        improvementArr = [[0 for x in range(w)] for y in range(h)]
    except IndexError:
        print("The index causing the problem is either {} or {}. And the items of issue are {} and {} respectively".format(len(v1_neighbors), len(v2_neighbors), v1_neighbors, v2_neighbors))
    for i in range(len(v1_neighbors)):
        for j in range(len(v2_neighbors)):
            #Instantiate each entry in the matrix to -Ric(vertex1, vertex2)
            try:
                improvementArr[i][j] = - compute_forman_single_edge((G, vertex1, vertex2))
            except TypeError:
                print("vertex 1 and vertex 2 are {} and {}. Their types are {} and {}".format(vertex1, vertex2, type(vertex1), type(vertex2)))
    
    for index1, neighbor1 in enumerate(v1_neighbors):
        for index2, neighbor2 in enumerate(v2_neighbors):
            G.add_edge(neighbor1, neighbor2)
            improvementArr[index1][index2] += compute_forman_single_edge(G, vertex1, vertex2)
            G.remove_edge(neighbor1, neighbor2)
            
    
    ind = np.unravel_index(np.argmax(np.array(improvementArr), axis=None), np.array(improvementArr).shape)
    print("Ind is {} and its type is: {}".format(ind, type(ind)))
    return v1_neighbors[ind[0]], v2_neighbors[ind[1]]
    
    
def addMostImprovingEdgeForman(G, vertex1, vertex2):
    e1, e2 = formanImprovement(G, vertex1, vertex2)
    G.add_edge(e1, e2)  
    
    
    
def softmax(v : list, temp : float) -> list:
    probs = []
    total = 0
    if len(v)==0:
        return []
    else:
        for item in v:
            probs.append(math.exp(temp * item))
            total = total + math.exp(temp * item)
        return [x / total for x in probs]



def ollivierRicciImprovement(G, vertex1, vertex2, ALPHA=0.0001):
    v1_neighbors = list(G.neighbors(vertex1))
    try:
        v1_neighbors.remove(vertex2)
    except ValueError:
        print("{} is not a neighbor of {} and therefore cannot be removed.".format(vertex2, vertex1))
    v2_neighbors = list(G.neighbors(vertex2))
    try:
        v2_neighbors.remove(vertex1)
    except ValueError:
        print("{} is not a neighbor of {} and therefore cannot be removed.".format(vertex1, vertex2))

    
    #frc = FormanRicci(G, verbose="INFO")
    #frc.compute_ricci_curvature()
    orc = OllivierRicci(G, alpha=ALPHA, verbose="INFO")
    improvementArr = [len(v1_neighbors)][len(v2_neighbors)]
    for i in range(len(v1_neighbors)):
        for j in range(len(v2_neighbors)):
            #Instantiate each entry in the matrix to -Ric(vertex1, vertex2)
            improvementArr[i][j] = - orc.compute_ricci_curvature_edges((vertex1, vertex2))
    
    for index1, neighbor1 in enumerate(v1_neighbors):
        for index2, neighbor2 in enumerate(v2_neighbors):
            G.add_edge(neighbor1, neighbor2)
            improvementArr[index1][index2] += orc.compute_ricci_curvature_edges((vertex1, vertex2))
            G.remove_edge(neighbor1, neighbor2)
            
    
    ind = np.unravel_index(np.argmax(improvementArr, axis=None), improvementArr.shape)
    return v1_neighbors[ind[0]], v2_neighbors[ind[1]]



def addMostImprovingEdgeOllivier(G, vertex1, vertex2):
    e1, e2 = ollivierRicciImprovement(G, vertex1, vertex2, ALPHA=0.0001)
    G.add_edge(e1, e2)
    

def naiveRewiring(G):
    
    ALPHA = 0.0001
    orc = OllivierRicci(G, alpha=ALPHA, verbose="INFO")
    orc.compute_ricci_curvature()
    G = orc.G


    ## calculate FRC
    ##for G in graphs:
    frc = FormanRicci(G, verbose="INFO")
    frc.compute_ricci_curvature()
    G = frc.G

    # graph edge features
    #for G in graphs:
    negative_edge_list = []
    for edge in G.edges():
        if G.edges[edge]['ricciCurvature'] < 0:
            negative_edge_list.append([edge, G.edges[edge]])
    
    
    #ALPHA = 0.0001
    CPLUS = 25
    #product is a list of tuples. Each tuple in product is composed of I) a tuple and II) the 'ricciCurvature value associated 
    #with the edge in (I). Which brings us back to (I), the tuple is comprised of two numbers: the labels of each vertex. 
    #For example, we may find that product[2] = ((0, 19), -0.865195632561) where this represents an edge between vertices labeled
    #0 and 19, and with Ricci curvature of approximately -0.87
    l = []
    for edge in G.edges():
        l.append(edge)
    l.sort(key=lambda edge : G.edges[edge]['ricciCurvature'])
    product = []
    for item in l:
        product.append((item, G.edges[item]['ricciCurvature']))
        #print(item, G.edges[item]['ricciCurvature'])

    improvement = -math.inf 

    numBack = -1 



    for item in product:
        point1 = item[0][0]
        point2 = item[0][1]
    
    #print(type(G.neighbors(point1)))
    
        numTries1 = 0
        numTries2 = 0
        MAX = 25
        k = random.choice(list(G.neighbors(point1)))
        while k in G.neighbors(point2) and numTries1 < MAX:
            k = random.choice(list(G.neighbors(point1)))
            numTries1 +=1
        l = random.choice(list(G.neighbors(point2)))
        while l in G.neighbors(point1) and numTries2 < MAX:
            l = random.choice(list(G.neighbors(point2)))
            numTries2 += 1
        
        G.add_edge(k, l)
        if product[numBack][1] > CPLUS:
            G.remove_edge(product[numBack][0], product[numBack][1])
            numBack -= 1
    
    nx.draw(G, with_labels=True)
    plt.show()
    #print("\n\nWhy no draw wahhh\n\n")
    return


def rewiring_algorithm_one(G, max_iterations, upper_bound=2, ALPHA=0.0001):
    orc = OllivierRicci(G, alpha=ALPHA, verbose="INFO")
    orc.compute_ricci_curvature()
    
    # graph edge features
    #for G in graphs:
    negative_edge_list = []
    for edge in G.edges():
        if orc.G.edges[edge]['ricciCurvature'] < 0:
            negative_edge_list.append([edge, G.edges[edge]])
    
    
    iterations = 0
    #negCounter = len(negative_edge_list)
    while iterations < max_iterations:
        for edge in negative_edge_list:
            #if negCounter <= 0:
            #    break
            addMostImprovingEdgeForman(G, edge[0][0], edge[0][1])
            if orc.compute_ricci_curvature_edges([edge[0]])[edge[0]] > upper_bound:
                G.remove_edge(edge[0])
            #negCounter -=1
        iterations +=1
    
            
    nx.draw(G, with_labels=True)
    plt.show()
    return
    
    
if __name__=="main":
    #G = nx.karate_club_graph()
    #G = nx.complete_graph(5)
    G = nx.full_rary_tree(4, 80)


    print(G)
    nx.draw(G, with_labels=True)
    plt.show()
    rewiring_algorithm_one(G, max_iterations=2)