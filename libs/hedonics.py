from .operations import add, move
from .export     import save_values

graph = alpha = gamma = None

def set_params(g, a, c):
    global graph, alpha, gamma
    graph = g
    alpha = a
    gamma = c

########################### Perspectives ###########################

# Individual

def hedonic(node, cluster):
    if cluster == {}:
        return -gamma
    elif cluster.get(node) == None:
        c = add(node, cluster)
        return len(c[node]) - alpha * len(cluster)
    else:
        return len(cluster[node]) - alpha * (len(cluster) - 1)

def hedonic_full(node, cluster):
    if cluster.get(node) == None:
        cluster = add(node, cluster)
    neighbors = len(cluster[node])
    strangers = len(cluster) - neighbors - 1
    a = (1 - alpha) * neighbors
    b =      alpha  * strangers
    return a - b

# Local

def sum_hedonics(cluster, hed):
    total = 0
    for key in cluster:
        total += hed(key, cluster)
    return total / 2

def potential(cluster):
    verts = len(cluster)
    edges = 0
    for key in cluster:
        edges += len(cluster[key])
    edges /= 2
    num  = verts * (verts - 1)
    num /= 2
    return edges - alpha * num

# General

def general(here, node, there, pot, hed):
    if not node in here.keys() and node in there.keys():
        there, here = move(node, there, here)
    if hed:
        return pot(here, hed) + pot(there, hed)
    else:
        return pot(here) + pot(there)

####################################################################

def global_potential(clusters):
    pot = 0
    for c in clusters:
        pot += potential(c)
    return pot

def perspectives(here, node, there, clusters):
    h_1 = hedonic_full(node, there) - hedonic_full(node, here)
    h_2 = hedonic(node, there) - hedonic(node, here)

    g_1  = general(there, node, here, potential, None)
    g_1 -= general(here, node, there, potential, None)

    g_2  = general(there, node, here, sum_hedonics, hedonic_full)
    g_2 -= general(here, node, there, sum_hedonics, hedonic_full)

    g_3  = general(there, node, here, sum_hedonics, hedonic)
    g_3 -= general(here, node, there, sum_hedonics, hedonic)

    save_values(node, clusters.index(here), clusters.index(there), (h_1, h_2, g_1, g_2, g_3))

    if (round(h_1, 2) == round(h_2, 2) == round(g_1, 2) == round(g_2, 2) == round(g_3, 2)):
        return h_1
    else:
        print("Divergence")
