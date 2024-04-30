# Milestone4
Milestone4


Misc Notes:


Things we’ve done since Milestone 3:

Instead of only considering the correlation between Forman and Ollivier Curvature, we considered the ‘Mutual Information’ metric which provides a different view of the relationship between the two curvature types. 

For some reason, the GraphRicciCurvature library has the functions to calculate Ollivier and Forman curvature for every edge, and can calculate Ollivier on a subset of edges (or even a single edge). Yet, this has not been implemented before for Forman curvature, so we implemented this. 

From here we describe our Forman-Improvement, Ollivier-Improvement and hybrid algorithms in parallel (not in the sense that they run literally in parallel, but they are similar). The algorithms each begin with calculating the curvature across the entire graph: both the Ollivier and hybrid algorithms use the Ollivier curvature here, while the Forman-Improvement algorithm uses the Forman curvature. From here, each algorithm looks at the most negatively curved edges and in the Forman-Improvement algorithm we then find the ‘bridge’ that would maximally increase Forman curvature. The same process is done in the hybrid (this is why we call it hybrid, because it considers both) and the ‘standard’ Ollivier algorithm finds the bridge that best improves Ollivier curvature. Then, all algorithms prune the edge with maximal curvature between v1 and v2, assuming its curvature exceeds a pre-set threshold. 

In pseudocode:

Forman Improvement / Forman Rewiring Algo(G, max_iterations, upper_bound){
	-Calculate Ricci curvature for every edge in the graph. 
	-We construct a list L of edges with negative curvature, and order the list in ascending order.
	-While iterations are < max_iterations{
		-for each edge e(v_i, v_j) in L {
			-We examine the neighbors N(v_i) and N(v_j) and find the edge amongst these 			neighbors (excluding v_i and v_j themselves) that would most improve the Forman 			curvature of the edge e(v_i, v_j).
			-Then, if there is an edge between v_i and v_j -either e(v_i, v_j) itself, or another edge - 			this case happens in the instance of multiple edges,  and the edge has curvature 			greater than upper_bound we prune that edge.
		}
		-Increase iterations counter.
	}

}


The Ollivier curvature algorithm works similarly, except every instance of Forman Curvature in the above algorithm is replaced with the ORC. The hybrid-curvature rewiring uses Ollivier curvature in step 1 of the algorithm to calculate the Ricci curvature for every edge. From then on, it uses the Forman curvature.

Unfortunately, we did not have time to do this for the Milestone but hope to include in the report a comparison of how long each rewiring takes on sample graphs (possibly run on gem5). 


We then experimented with the ‘Planetoid’  dataset from PyTorch Geometric. This dataset contains nodes that represent articles published either in PubMed, Cora or SiteSeer. Each vertex is represented by a 1433-dimensional bag-of-words feature vector, while the edges represent citations between articles. We engaged in a node-level prediction task to determine how well we could predict the ‘category / genre’ of each document. We noted that a completely untrained model had an ‘accuracy’ (almost embarrassing to even call it that) of 0.09. A multi-layer perceptron (per py-torch documentation on the same dataset had test accuracy of 0.5900. We then trained a GNN and our test accuracy improved to between 0.78 and 0.80 (we achieved a highest test accuracy of 0.80, but this involved fine-tuning some hyper parameters. The 0.78 test score was achieved with the ‘standard’ PyTorch parameters. We then applied our rewiring algorithm and again tackled the same problem. The test accuracy here was ___________ still waiting for this to finish running. (THE _____ here will clearly be ready before tomorrow, but I wanted to send this asap)

Going forward, for the paper submission we aim to apply the algorithms to more datasets, apply them not just to node prediction tasks but also link prediction and graph prediction to see how the different problems are affected by rewiring. Also, we aim to calculate the comparative runtimes between the three different rewiring algorithms and comparing them. And, as always, we hope to further improve the algorithms. Hopefully most of this can be fit into the paper next week. We also aim to collect statistics on how curvature (a local property) can be related to graph width / depth (a global property).

There is much other work to do - and this is less likely to make it into the paper due to time constraints but this includes: constructing time series for how the curvature of edges changes during the rewiring process. While this is already intrinsically calculated, we must store the data and then create a visualization for it. We also aim to compare Node-Based Multifractal Analysis methods for a rewiring improvement algorithm. 

Note: for this milestone we ended up referencing far more sources than the previous milestones (around 30-40 sources). Unfortunately these cannot all be put in the ‘previous work’ section for length reasons but of course they will be cited in our paper. 

Code is currently private on GitHub but can be made public if anyone is interested. 
