import math 
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from GraphRicciCurvature.FormanRicci import FormanRicci
from sklearn.metrics import mutual_info_score

'''
Misc. graphs, and function stubs to use in 1) wrapper to loop over the graphs below and input them into the funcs in this doc
and 2) eventually draw these graphs
#Load network
graphs = []
graphs.append(nx.karate_club_graph())
graphs.append(nx.complete_graph(10))
graphs.append(nx.full_rary_tree(4, 80))
G = nx.karate_club_graph()
G = nx.complete_graph(3)
G = nx.full_rary_tree(4, 80)
G = nx.caveman_graph(20, 7)
G = nx.random_regular_graph(10, 150, seed=42)
G = nx.binomial_tree(10)
G = nx.star_graph(100)
G = nx.hypercube_graph(8)
G = nx.random_internet_as_graph(5000)
print(G)
nx.draw(G, with_labels=True)
for G in graphs:
print(G)
nx.draw(G, with_labels=True)
'''

def graph_to_ollivier_numpy_array(G, ALPHA=0.0001):
    orc = OllivierRicci(G, alpha=ALPHA, verbose="INFO")
    orc.compute_ricci_curvature()
    return nx.to_numpy_array(orc.G)


def graph_to_forman_numpy_array(G, ALPHA=0.0001):
    frc = FormanRicci(G, verbose="INFO")
    frc.compute_ricci_curvature()
    return nx.to_numpy_array(frc.G)

def get_edge_ollivier_curvature(G, e, ALPHA=0.0001):
    orc = OllivierRicci(G, alpha=ALPHA, verbose="INFO")
    orc.compute_ricci_curvature()
    return orc.G.edges[e]['ricciCurvature']


def get_edge_forman_curvature(G, e, ALPHA=0.0001):
    frc = FormanRicci(G, verbose="INFO")
    frc.compute_ricci_curvature()
    return frc.G.edges[e]['ricciCurvature']
    
    


def graph_mutual_info(G, ALPHA=0.0001):
    orc = OllivierRicci(G, alpha=ALPHA, verbose="INFO")
    orc.compute_ricci_curvature()
        
    frc = FormanRicci(G, verbose="INFO")
    frc.compute_ricci_curvature()
    
    orcArr = nx.to_numpy_array(orc.G)
    frcArr = nx.to_numpy_array(frc.G)
    
    formanList = []
    ollivierList = []
    for edge in orc.G.edges():
        ollivierList.append(orc.G.edges[edge]['ricciCurvature'])
        formanList.append(frc.G.edges[edge]['formanCurvature'])
    
    return orc, frc, mutual_info_score(np.asarray(formanList), np.asarray(ollivierList))


# Compute graph ricci curvatures (per edge)
def orc_frc_correl(G):
    ALPHA = 0.0001
    orc = OllivierRicci(G, alpha=ALPHA, verbose="INFO")
    orc.compute_ricci_curvature()
    G = orc.G
    
    ## calculate FRC
    ##for G in graphs:
    frc = FormanRicci(G, verbose="INFO")
    frc.compute_ricci_curvature()
    G = frc.G
    ##FRC_list.append(frc.G)
    
    #Here we create two dictionaries orcValues and frcValues. The keys in each dictionary are tuples representing the edges in the graph. 
    #I.e. if the graph has an edge from node 2 to node 5, then there will be a key of type tuple = (2, 5) and whose value is the
    #Olivier curvature in orcValues and formanCurvature in frcValues

    orcValues = {}
    frcValues = {}
    #We need the means for both to calculate the correlation
    orcMean = 0
    frcMean = 0
    for edge in G.edges():
        #Create dictionaries
        orcValues[(edge[0], edge[1])] = G.edges[edge]['ricciCurvature']
        frcValues[(edge[0], edge[1])] = G.edges[edge]['formanCurvature']
        #Calculate means
        orcMean = orcMean + G.edges[edge]['ricciCurvature']
        frcMean = frcMean + G.edges[edge]['ricciCurvature']

    #Complete calculation of means by normalization
    orcMean = orcMean / len(list(orcValues.keys()))
    frcMean = frcMean / len(list(orcValues.keys()))

    #Find the correlation between FRC and ORC on the edges of the graph
    correlation = 0
    orcVariance = 0
    frcVariance = 0
    #print("ORC Mean is {}, ORC Variance is {}, Forman mean is {} and Forman variance is {}".format(orcMean, orcVariance, frcMean, frcVariance))

    for key in orcValues.keys():
        correlation = correlation + (orcValues[key] - orcMean)*(frcValues[key]-frcMean)
        orcVariance = orcVariance + math.pow(orcValues[key]-orcMean, 2)
        frcVariance = frcVariance + math.pow(frcValues[key]-frcMean, 2)

    correlation = correlation / (math.sqrt(orcVariance)*math.sqrt(frcVariance))
    #print("Correlation between ORC and FRC values is: {}".format(correlation))
    
    return orcValues, frcValues
    
    
    
def orc_frc_negative_correl(G):
    ALPHA = 0.0001
    orc = OllivierRicci(G, alpha=ALPHA, verbose="INFO")
    orc.compute_ricci_curvature()
    G = orc.G


    ## calculate FRC
    ##for G in graphs:
    frc = FormanRicci(G, verbose="INFO")
    frc.compute_ricci_curvature()
    G = frc.G
    ##FRC_list.append(frc.G)
    
    orcNegativeMean = 0
    frcNegativeMean = 0
    orcNegativeValues = {}
    frcNegativeValues = {}
    numNegative = 0 
    orcNegativeVariance = 0
    frcNegativeVariance = 0
    correl = 0

    for edge in G.edges():
        if G.edges()[edge]['ricciCurvature'] < 0:
            orcNegativeValues[(edge[0], edge[1])] = G.edges()[edge]['ricciCurvature']
            frcNegativeValues[(edge[0], edge[1])] = G.edges()[edge]['formanCurvature']
            numNegative += 1 
            orcNegativeMean += G.edges()[edge]['ricciCurvature']
            frcNegativeMean += G.edges()[edge]['formanCurvature']


    if numNegative > 0:
        orcNegativeMean = orcNegativeMean / numNegative
        frcNegativeMean = frcNegativeMean / numNegative
        for key in orcNegativeValues.keys():
            correl += (orcNegativeValues[key]-orcNegativeMean)*(frcNegativeValues[key]-frcNegativeMean)
            orcNegativeVariance += math.pow(orcNegativeValues[key]-orcNegativeMean, 2)
            frcNegativeVariance += math.pow(frcNegativeValues[key]- frcNegativeMean, 2)
        correl = correl / (math.sqrt(orcNegativeVariance)*math.sqrt(frcNegativeVariance))
    
        #print("Correlation when restriced to edges which have negative ORC curvature is: {}".format(correl))  
        
    return orcNegativeValues, frcNegativeValues



        
    
if __name__== "__main__":
    #G = nx.random_regular_graph(10, 150, seed=42)
    #G = nx.full_rary_tree(4, 80)
    G = nx.complete_graph(4)
    #G = nx.complete_graph(10)
    #G = nx.random_internet_as_graph(5000)
    d11, d12 = orc_frc_correl(G)
    d21, d22 = orc_frc_negative_correl(G)
    
    #print("mutual info is")
    o1, f1, mutual1 = graph_mutual_info(G)
    print(type(o1))
    for edge in o1.G.edges():
        print(o1.G.edges[edge]['ricciCurvature'])
    #o2, f2, mutual2 = negative_graph_mutual_info(G)
    #print(graph_mutual_info(G))
    #print("negative mutual info is")
    #print(negative_graph_mutual_info(G))
    #print("__________________")
    
    print("HI")
    print(o1.G.edges[(0, 1)]['ricciCurvature'])
    print(get_edge_ollivier_curvature(G, (0, 1)))
    
    '''
    for i in range(len(o1)):
        print(" For the {}th edge ORC is: {} and FRC is: {}".format(i, o1[i], f1[i]))
        #print(o1[i])
        #print("FRC is:")
        #print(f1[i])
    print("Mutual info")
    print(mutual1)
    print(____________________________)
    
    
    c = 1
    for item in list(d11.keys()):
        print("For edge {} we have ORC {}, we have FRC {}".format(item, d11[item], d12[item]))
        #print(item)
        #print(d21[item])
        print("\n\n")
        #if c > 30:
        #    break
        c +=1
    '''
    nx.draw_networkx(G)
    plt.show()
    
    #curv21 = d21.values()
    #curv11 = d11.values()
    
    #print(d11[(123, 131)])
    #print("hello")
    #s11 = pd.Series(d11)
    #s12 = pd.Series(d12)
    #print(s11.loc[(12, 123)])
    #print(s11.keys())
    
    #df1 = pd.DataFrame.from_dict(d12)
    #print(df1.info())
    #print(df1.head())
    
    #print(type(curv21))
    
    #for item in list(curv21)[:20]:
    #    print(item)
    #    print(type(item))
    
    #print("we have exited the last function call\n\n\n")

        
    
     
    
    
    
