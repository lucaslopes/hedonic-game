from copy import deepcopy

graph = None

def set_graph(g):
    global graph
    graph = g

def add(node, original):
    g = deepcopy(original)
    g[node] = []
    for key in g:
        if node in graph[key]:
            g[key].append(node)
            g[node].append(key)
    return g

def remove(node, original):
    g = deepcopy(original)
    for v in g[node]:
        g[v].remove(node)
    g.pop(node)
    return g

def move(node, From, To):
    From = remove(node, From)
    To   = add(node, To)
    return From, To
