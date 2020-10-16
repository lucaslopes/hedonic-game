# coding=utf-8
import numpy as np
import igraph

def calc_score_diff(u, v):
    x = [0 for _ in range(len(u))]
    y = [0 for _ in range(len(v))]
    # print('u:', type(u), u, '\n'*2)
    # print('v:', type(v), v, '\n'*2)
    for node, label in u.items():
        x[node] = label
    for node, label in v.items():
        y[node] = label
    # print('x:', type(x), x, '\n'*2)
    # print('y:', type(y), y, '\n'*2)
    x, y = np.array(x), np.array(y)
    if len(x) != len(y):
        print(len(x), len(y))
        raise ValueError('x and y must be arrays of the same size')
    matches1 = sum(x == y)
    matches2 = sum(x == 1-y)
    score = max([matches1, matches2]) / len(x)
    return score

## Graph-aware measures (igraph version)
def gam(self, u, v, method="rand", adjusted=True):
    """
    Compute one of 11 graph-aware measures to compare graph partitions.
    
    Parameters
    ----------
    self: Graph of type 'igraph.Graph' on which the partitions are defined.

    u: Partition of type 'igraph.clustering.VertexClustering' on 'self', or a dictionary of node:community.

    v: Partition of type 'igraph.clustering.VertexClustering' on 'self', or a dictionary of node:community.

    method: 'str'
      one of 'rand', 'jaccard', 'mn', 'gmn', 'min' or 'max'

    adjusted: 'bool'
      if True, return adjusted measure (preferred). All measures can be adjusted except 'jaccard'.
      
    Returns
    -------
    A graph-aware similarity measure between vertex partitions u and v.
    
    Examples
    --------
    >>> g = ig.Graph.Famous('Zachary')
    >>> part1 = g.community_multilevel()
    >>> part2 = g.community_label_propagation()
    >>> print(g.GAM(part1, part2))
    
     Reference
    ---------
    Valérie Poulin and François Théberge, "Comparing Graph Clusterings: Set partition measures vs. Graph-aware measures", https://arxiv.org/abs/1806.11494.
    """
    if(type(u) is dict):
        d1 = u
    else:
        d1 = {val:idx for idx,part in enumerate(u) for val in part}
    if(type(v) is dict):
        d2 = v
    else:
        d2 = {val:idx for idx,part in enumerate(v) for val in part}    
    bu = np.array([(d1[x.tuple[0]]==d1[x.tuple[1]]) for x in self.es])
    bv = np.array([(d2[x.tuple[0]]==d2[x.tuple[1]]) for x in self.es])
    su = np.sum(bu) # qnty of intra conections in answer
    sv = np.sum(bv) # qnty of intra conections in gt
    suv = np.sum(bu*bv) # 
    m = len(bu) # len(edges)
    # print(f'bu={bu}, bv={bv}, su={su}, sv={sv}, suv={suv}, m={m}')

    ## all adjusted measures
    if adjusted:
        if method=="jaccard":
            print("no adjusted jaccard measure, set adjusted=False")
            return None
        if method=="rand" or method=="mn":
            if (np.average([su,sv])-su*sv/m) == 0: return 0
            else: return((suv-su*sv/m)/(np.average([su,sv])-su*sv/m))
        if method=="gmn":
            if (np.sqrt(su*sv)-su*sv/m) == 0: return 0
            else: return((suv-su*sv/m)/(np.sqrt(su*sv)-su*sv/m))
        if method=="min":
            if (np.min([su,sv])-su*sv/m) == 0: return 0
            else: return((suv-su*sv/m)/(np.min([su,sv])-su*sv/m))
        if method=="max":
            if (np.max([su,sv])-su*sv/m) == 0: return 0
            else: return((suv-su*sv/m)/(np.max([su,sv])-su*sv/m))
        if method=="diff":
            return calc_score_diff(d1, d2)
        else:
            print('Wrong method!')

    ## all non-adjusted measures
    else:
        if method=="jaccard":
            union_b = sum((bu+bv)>0)
            return(suv/union_b)
        if method=="rand":
            return(1-(su+sv)/m+2*suv/m)
        if method=="mn":
            return(suv/np.average([su,sv]))
        if method=="gmn":
            return(suv/np.sqrt(su*sv))
        if method=="min":
            return(suv/np.min([su,sv]))
        if method=="max":
            return(suv/np.max([su,sv]))
        if method=="diff":
            return calc_score_diff(d1, d2)
        else:
            print('Wrong method!')
        
    return None

igraph.Graph.gam = gam

import igraph
import numpy as np

def community_ecg(self, weights=None, ens_size = 16, min_weight = 0.05):
    """
    Stable ensemble-based graph clustering;
    the ensemble consists of single-level randomized Louvain; 
    each member of the ensemble gets a "vote" to determine if the edges 
    are intra-community or not;
    the votes are aggregated into ECG edge-weights in range [0,1]; 
    a final (full depth) Louvain is run using those edge weights;
    
    Parameters
    ----------
    self: graph of type 'igraph.Graph'
      Graph to define the partition on.
    weights: list of double, optional 
      the edge weights
    ens_size: int 
      the size of the ensemble of single-level Louvain
    min_weight: double in range [0,1] 
      the ECG edge weight for edges with zero votes from the ensemble

    Returns
    -------
    partition
      The final partition, of type 'igraph.clustering.VertexClustering'
    partition.W
      The ECG edge weights
    partition.CSI
      The community strength index

    Notes
    -----
    The ECG edge weight function is defined as:
      
      min_weight + ( 1 - min_weight ) x (#votes_in_ensemble) / ens_size
      
    Edges outside the 2-core are assigned 'min_weight'.
    
    Examples
    --------
    >>> g = igraph.Graph.Famous('Zachary')
    >>> part = g.community_ecg(ens_size=25, min_weight = .1)
    >>> print(part.CSI)
    
    Reference
    ---------
    Valérie Poulin and François Théberge, "Ensemble clustering for graphs: comparisons and applications", Appl Netw Sci 4, 51 (2019). 
    https://doi.org/10.1007/s41109-019-0162-z
    """
    W = [0]*self.ecount()
    ## Ensemble of level-1 Louvain
    for i in range(ens_size):
        p = np.random.permutation(self.vcount()).tolist()
        g = self.permute_vertices(p)
        l1 = g.community_multilevel(weights=weights, return_levels=True)[0].membership
        b = [l1[p[x.tuple[0]]]==l1[p[x.tuple[1]]] for x in self.es]
        W = [W[i]+b[i] for i in range(len(W))]
    W = [min_weight + (1-min_weight)*W[i]/ens_size for i in range(len(W))]
    ## Force min_weight outside 2-core
    core = self.shell_index()
    ecore = [min(core[x.tuple[0]],core[x.tuple[1]]) for x in self.es]
    w = [W[i] if ecore[i]>1 else min_weight for i in range(len(ecore))]
    part = self.community_multilevel(weights=w)
    part.W = w
    part.CSI = 1-2*np.sum([min(1-i,i) for i in w])/len(w)
    return part

igraph.Graph.community_ecg = community_ecg