from random import random, shuffle
from time import time, sleep
import os
import asyncio
import numpy as np
import pandas as pd
import networkx as nx
from networkx.generators.community import planted_partition_graph as PPG
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
import scipy.sparse as sprs
import scipy.spatial
import scipy.sparse.linalg

#################################################################################################
## Algoritms #################################################################################################

def almost_robust(G): # Hedonic (almost) Robust
	moved, nodes_list = True, list(G.nodes)
	while moved is True:
		moved = False
		shuffle(nodes_list)
		for node in nodes_list:
			if check_status(*get_node_atributes(G, node)) in ['move_ambos', 'move_alpha0', 'move_alpha1', 'depende_alpha1']:
				G = move(G, node)
				moved = True
	for node in G.nodes:
		if not satisfied(G, node):
			print('>>> ROBUST NOT FOUND!', node, get_node_atributes(G, node), G.nodes[node]['cluster'], G.clusters_nodes, check_status(*get_node_atributes(G, node)))
	return G

def find_equilibrium_shuffle(G, alpha=0): # Hedonic Naive
	moved, nodes_list = True, list(G.nodes)
	while moved is True:
		moved = False
		shuffle(nodes_list)
		for node in nodes_list:
			if not satisfied_for_alpha(G, node, alpha):
				G = move(G, node)
				moved = True
	return G

def hedonic_weighted(G, W, alpha=0):
	moved, nodes_list = True, list(G.nodes)
	while moved is True:
		moved = False
		shuffle(nodes_list)
		for node in nodes_list:
			if not satisfied_weighted(G, node, W[node]*alpha):
				G = move(G, node)
				moved = True
	return G

def spectral(G):
	A = nx.adjacency_matrix(G).toarray()
	w, v = np.linalg.eigh(A)
	ew_2 = v[:,-2]
	labels = np.array([1 if x > 0 else 0 for x in ew_2], dtype='int')
	# print('Spectral acuracy:', accuracy(extract_labels(G), labels))
	return labels

def one_pass_neighbors(G):
	want_move = []
	for node in G.nodes:
		here, there, dis_here, dis_there = get_node_atributes(G, node)
		if here < there:
			want_move.append(node)
	for node in want_move:
		move(G, node)
	return G

#################################################################################################
## Functions #################################################################################################

def satisfied_for_alpha(G, node, alpha=.5, return_profit=False):
	here, there, dis_here, dis_there = get_node_atributes(G, node)
	value_here  = here  * (1-alpha) - dis_here  * alpha
	value_there = there * (1-alpha) - dis_there * alpha
	return value_there - value_here if return_profit else value_here >= value_there

def satisfied_weighted(G, node, weights, return_profit=False):
	here, there = [0,0], [0,0] # [pros, cons]
	my_cluster = G.nodes[node]['cluster']
	not_friend = list(np.arange(0,len(weights),1))
	for friend in G.neighbors(node):
		not_friend[friend] = None
	not_friend = [node for node in not_friend if node is not None]
	for i in G.neighbors(node):
		if my_cluster == G.nodes[i]['cluster']:
			here[0] += weights[i]
		else:
			there[0] += weights[i]
	for i in not_friend:
		if my_cluster == G.nodes[i]['cluster']:
			here[1] += weights[i]
		else:
			there[1] += weights[i]
	value_here, value_there = (here[0]-here[1]), (there[0]-there[1])
	return value_there - value_here if return_profit else value_here >= value_there

def load_ground_truth(G, values, label='gt', replace={'0':0,'1':1}):
	for node, cluster in zip(G.nodes, values):
		G.nodes[node][label] = replace[cluster]
	return G

def initialize(G, labels=[], prob=.5):
	if len(labels) != len(G.nodes):
		for _ in range(len(G.nodes)):
			labels.append(0 if random() < prob else 1)
	on_c0 = labels.count(0)
	heres = np.zeros(len(G.nodes))
	edges_c0, edges_c1 = 0, 0
	for edge in G.edges:
		n0, n1 = edge[0], edge[1]
		c_n0 = labels[n0]
		c_n1 = labels[n1]
		if c_n0 == c_n1:
			heres[n0] += 1
			heres[n1] += 1
			if c_n0 == 0: edges_c0 += 1
			else:         edges_c1 += 1
	for node, i, cluster, here in zip(G.nodes, range(len(G.nodes)), labels, heres):
		if node != i:
			print(f'ERROR: index of node {node} is different from index {i}.')
		G.nodes[node]['cluster'] = cluster
		G.nodes[node]['here']    = here
	G.clusters_nodes = [on_c0, len(G.nodes)-on_c0]
	G.clusters_edges = [edges_c0, edges_c1]
	return G

def initialize_old(G, labels=[], prob=.5):
	G.clusters_nodes = [0, 0] # c1 = total_nodes - nodes_c0
	G.clusters_edges = [0, 0]
	for node in G.nodes:
		G.nodes[node]['here'] = 0
		if len(labels) == len(G.nodes):
			G.nodes[node]['cluster'] = int(labels[node])
		else:
			G.nodes[node]['cluster'] = 0 if random() < prob else 1
		G.clusters_nodes[G.nodes[node]['cluster']] += 1
	node_cluster = [c for n,c in list(G.nodes.data('cluster'))]
	for edge in G.edges:
		n1, n2 = edge[0], edge[1]
		c_n1 = node_cluster[n1] # G.nodes[n1]['cluster']
		c_n2 = node_cluster[n2] # G.nodes[n2]['cluster']
		if c_n1 == c_n2:
			G.nodes[n1]['here'] += 1
			G.nodes[n2]['here'] += 1
			if c_n1 == 0: G.clusters_edges[0] += 1
			else:         G.clusters_edges[1] += 1
	return G

def move(G, node):
	G.nodes[node]['cluster'] = 1 - G.nodes[node]['cluster']
	G.nodes[node]['here'] = get_node_atributes(G, node)[1] # there
	for friend in G.neighbors(node):
		if G.nodes[node]['cluster'] == G.nodes[friend]['cluster']:
			G.nodes[friend]['here'] += 1
		else:
			G.nodes[friend]['here'] -= 1
	if G.nodes[node]['cluster'] == 0:
		G.clusters_nodes[0] += 1
		G.clusters_nodes[1] -= 1
		G.clusters_edges[0] += G.nodes[node]['here']
		G.clusters_edges[1] -= get_node_atributes(G, node)[1] # there
	else:
		G.clusters_nodes[0] -= 1
		G.clusters_nodes[1] += 1
		G.clusters_edges[0] -= get_node_atributes(G, node)[1] # there
		G.clusters_edges[1] += G.nodes[node]['here']
	# print(node, get_node_atributes(G, node))
	return G

def check_status(here, there, dis_here, dis_there):
	if   (here <  there) and (dis_here >  dis_there):
		return 'move_ambos'
	elif (here <  there) and (dis_here == dis_there):
		return 'move_alpha0'
	elif (here == there) and (dis_here >  dis_there):
		return 'move_alpha1'
	elif (here <  there) and (dis_here <  dis_there):
		return 'depende_alpha0'
	elif (here >  there) and (dis_here >  dis_there):
		return 'depende_alpha1'
	elif (here == there) and (dis_here == dis_there):
		return 'independe'
	elif (here >  there) and (dis_here <  dis_there):
		return 'fica_ambos'
	elif (here >  there) and (dis_here == dis_there):
		return 'fica_alpha0'
	elif (here == there) and (dis_here <  dis_there):
		return 'fica_alpha1'

def max_edges_possible(nodes):
	return nodes * (nodes - 1) / 2

def extract_labels(G, label='block'):
	return [l for n,l in list(G.nodes.data(label))]
	# return [G.nodes[node][label] for node in G.nodes]

def get_node_atributes(G, node):
	here  = G.nodes[node]['here']
	there = len(list(G.neighbors(node))) - here
	dis_here  = G.clusters_nodes[G.nodes[node]['cluster']]-1 - here
	dis_there = G.clusters_nodes[1-G.nodes[node]['cluster']] - there
	return here, there, dis_here, dis_there

def satisfied(G, node):
	here, there, dis_here, dis_there = get_node_atributes(G, node)
	return here >= there and dis_here <= dis_there

def add_noise(labels, noise=.5):
	new_labels = []
	for label in labels:
		new_labels.append(1 - label if random() < noise else label)
	return new_labels

def separate_clusters(G):
	clusters = [[], []]
	for node in G.nodes:
		clusters[G.nodes[node]['cluster']].append(node)
	return clusters

def measure_robustness(G):
	nodes_satisfied = 0
	for node in G.nodes:
		if satisfied(G, node):
			nodes_satisfied += 1
	return nodes_satisfied / len(G.nodes)

def accuracy(x,y): # WARNING: x and y must be only 0 or 1
	x, y = np.array(x), np.array(y)
	if len(x) != len(y):
		print(len(x), len(y))
		raise ValueError('x and y must be arrays of the same size')
	matches1 = sum(x == y)
	matches2 = sum(x == 1-y)
	score = max([matches1, matches2]) / len(x)
	return score

def run_algorithm(G, GT, noise=.5, alpha=1, W=None, alg='naive'):
	G = initialize(G, labels=add_noise(GT, noise))
	seconds = time()
	if alg == 'hedonic' : answer = almost_robust(G)
	if alg == 'naive'   : answer = find_equilibrium_shuffle(G, alpha)
	if alg == 'weighted': answer = hedonic_weighted(G, W, alpha)
	if alg == 'onepass' : answer = one_pass_neighbors(G)
	if alg == 'spectral': answer = initialize(G, labels=spectral(G))
	seconds = time() - seconds
	acc = accuracy(GT, extract_labels(answer, label='cluster'))
	robustness = measure_robustness(answer)
	infos = (answer.clusters_nodes, answer.clusters_edges, len(answer.edges))
	return seconds, acc, robustness, infos, answer

#################################################################################################

def get_file_name(folder='', outname='no_name'):
	outdir = './outputs/'
	if not os.path.exists(outdir):
		os.mkdir(outdir)
	outdir = f'./outputs/{folder}/'
	if not os.path.exists(outdir):
		os.mkdir(outdir)
	fullname = os.path.join(outdir, outname) 
	return fullname

#################################################################################################

def get_real_nets(nets=['karate', 'dolphins', 'pol_blogs', 'pol_books']):
	real_nets = {}
	for net in nets:
		csv = pd.read_csv(f'real_nets/csv/{net}.csv', names=['A', 'B'])
		gt  = pd.read_csv(f'real_nets/gt/{net}.csv',  names=['N', 'GT'])
		clusters = gt['GT'].unique()
		G = nx.from_pandas_edgelist(csv, 'A', 'B')
		G = load_ground_truth(G, gt['GT'].values, replace={clusters[0]:0, clusters[1]:1})
		real_nets[net] = G
		# colors = ['red' if G.nodes[node]['gt'] == 0 else 'blue' for node in G.nodes]
		# nx.draw(G, pos=nx.spring_layout(G), node_color=colors) # , node_color=colors, alpha=alpha, width=width, node_size=sizes, edge_color=edge_color
		# plt.show()
	return real_nets

#################################################################################################
## Experiment 1: synthetic Networks with Noise #############################################################

def exp_1(
	noises=np.linspace(0,.5,11), multipliers=np.concatenate(([.001], np.linspace(0,1,11)[1:])), ps=np.linspace(.01,.1,10),
	instances=25, repetitions=25, communities=2, nodes_per_cluster=500):

	total = len(noises) * len(multipliers) * len(ps) * instances * repetitions
	went  = 0

	begin = time()
	print(f'\n\nExp 1 begin at:{begin} -- TOTAL = {total}\n\n')
	for noi in noises:
		df_results = pd.DataFrame(columns=['noise','mult','p_in','p_out','instance','repetition'])
		for mult in multipliers:
			for p in ps:
				for i in range(instances):
					G = PPG(communities, nodes_per_cluster, p, p*mult)
					G = nx.convert_node_labels_to_integers(G.subgraph(max(nx.connected_components(G), key=len)))
					GT = extract_labels(G)
					spectral_answered, spectral_answer = False, None
					for r in range(repetitions):
						print(f'% = {round(went/total*100, 2)}%\tNoise = {np.where(noises==noi)[0][0]+1}/{len(noises)}\tMult = {np.where(multipliers==mult)[0][0]+1}/{len(multipliers)}\tP = {np.where(ps==p)[0][0]+1}/{len(ps)}\tInst = {i+1}/{instances}\tRep = {r+1}/{repetitions}')
						
						if not spectral_answered:
							spectral_answer = run_algorithm(G, GT, alg='spectral')
						onepass_answer = run_algorithm(G, GT, noise=noi, alg='onepass')
						hedonic_answer = run_algorithm(G, GT, noise=noi, alg='naive') # here

						went += 1
						df_results = df_results.append({
							'noise' : noi,
							'mult'  : mult,
							'p_in'  : p,
							'p_out' : p*mult,
							'instance' : i,
							'repetition' : r,

							'spectral_time': spectral_answer[0],
							'spectral_accuracy': spectral_answer[1],
							'spectral_robustness': spectral_answer[2],
							'spectral_infos': spectral_answer[3],

							'onepass_time': onepass_answer[0],
							'onepass_accuracy': onepass_answer[1],
							'onepass_robustness': onepass_answer[2],
							'onepass_infos': onepass_answer[3],

							'hedonic_time': hedonic_answer[0],
							'hedonic_accuracy': hedonic_answer[1],
							'hedonic_robustness': hedonic_answer[2],
							'hedonic_infos': hedonic_answer[3] },
							ignore_index=True)
		df_results.to_csv(get_file_name('noises', f'noise={round(noi,3)}.csv'), index=False)
	print('\n\n\nFINISHED EXP 1!', time()-begin)

#################################################################################################
## Experiment 2: Real Networks with Noise #############################################################

def exp_2(
	algorithms = ['onepass', 'naive'], noises=np.linspace(0,.5,11), # here
	networks=get_real_nets(), repetitions = 1000):

	df = pd.DataFrame(columns=['network', 'algorithm'])
	went, total = 0, repetitions*len(networks)

	begin = time()
	print(f'\n\nExp 2 begin at:{begin} -- TOTAL = {total}\n\n')
	for net in list(networks):
		G  = networks[net]
		GT = extract_labels(G, label='gt')
		spectral_answered, spectral_answer = False, None
		for r in range(repetitions):
			print(f'% = {round(went/total*100, 2)}%\tNet = {net}\tRep = {r}')
			
			if not spectral_answered:
				spectral_answer = run_algorithm(G, GT, alg='spectral')
			df = df.append({
				'network':net, 'algorithm':f'spectral',
				'accuracy':spectral_answer[1], 'time':spectral_answer[0],
				'robustness':spectral_answer[2], 'infos':spectral_answer[3]}, ignore_index=True)
			for alg in algorithms:
				for noi in noises:
					answer = run_algorithm(G, GT, noise=noi, alg=alg)
					df = df.append({
						'network':net, 'algorithm':f'{alg}_n{round(noi,2)}',
						'accuracy':answer[1], 'time':answer[0],
						'robustness':answer[2], 'infos':answer[3]}, ignore_index=True)
			went += 1
	df.to_csv(get_file_name('real_nets', f'{len(networks)}_networks_{repetitions}_repts.csv'), index=False)
	print('\n\n\nFINISHED EXP 2!', time()-begin)

#################################################################################################
## Experiment 3: Consensus #############################################################

def exp_cons(communities=2, nodes_per_cluster=500, p=.05, mult=.585, repts=20):
	G = PPG(communities, nodes_per_cluster, p, p*mult)
	# G = nx.convert_node_labels_to_integers(G.subgraph(max(nx.connected_components(G), key=len)))
	GT = extract_labels(G)
	W = np.zeros((len(G.nodes), len(G.nodes)))
	accs = 0
	for r in range(repts):
		time, acc, robst, infos, graph = run_algorithm(G, GT, alg='naive', alpha=nx.density(G)) # here
		accs += acc
		c0, c1 = separate_clusters(graph)
		for i in range(len(c0)):
			for j in range(i+1, len(c0)):
				n1, n2 = c0[i], c0[j]
				W[n1][n2] += robst
				W[n2][n1] += robst
		for i in range(len(c1)):
			for j in range(i+1, len(c1)):
				n1, n2 = c1[i], c1[j]
				W[n1][n2] += robst
				W[n2][n1] += robst
	W /= repts
	accs /= repts
	time, acc, robst, infos, graph = run_algorithm(G, GT, alpha=nx.density(G), W=W, alg='weighted')
	print('\n'*3,acc, 'media =', accs)

#################################################################################################
## Experiment 4: Markov #############################################################

# https://asajadi.github.io/fast-pagerank/
def pagerank(A, p=0.85, personalize=None, reverse=False):
    """ Calculates PageRank given a csr graph
    Inputs:
    -------
    G: a csr graph.
    p: damping factor
    personlize: if not None, should be an array with the size of the nodes
                containing probability distributions.
                It will be normalized automatically
    reverse: If true, returns the reversed-PageRank

    outputs
    -------
    PageRank Scores for the nodes
    """
	# In Moler's algorithm, $$A_{ij}$$ represents the existences of an edge
    # from node $$j$$ to $$i$$, while we have assumed the opposite!
    if reverse:
        A = A.T
    n, _ = A.shape
    r = sp.asarray(A.sum(axis=1)).reshape(-1)
    k = r.nonzero()[0]
    D_1 = sprs.csr_matrix((1 / r[k], (k, k)), shape=(n, n))
    if personalize is None:
        personalize = sp.ones(n)
    personalize = personalize.reshape(n, 1)
    s = (personalize / personalize.sum()) * n
    I = sprs.eye(n)
    x = sprs.linalg.spsolve((I - p * A.T @ D_1), s)
    x = x / x.sum()
    return x

def nxgraph(edgelist):
	G = nx.Graph()
	G.add_edges_from(edgelist)
	return G

def exp_markov(G):
	# print(len(G.nodes), 2 ** (len(G.nodes)-1), len(G.edges), nx.density(G))

	possible_states = np.arange(0, 2 ** (len(G.nodes)-1), 1)
	bins = [('0'*len(G.nodes)+'{0:b}'.format(state))[-len(G.nodes):] for state in possible_states]
	# print(int(bins[-4],2))

	W = np.zeros((len(possible_states), len(possible_states)))
	for current in range(len(possible_states)):
		state = bins[current]
		G = initialize(G, labels=[int(value) for value in state])
		for node in G.nodes:
			change = state[:node]+str(1-int(state[node]))+state[node+1:]
			reverse = ''
			for v in change:
				reverse += str(1-int(v))
			next_state = min(int(change,2), int(reverse,2))
			profit = satisfied_for_alpha(G, node, alpha=nx.density(G), return_profit=True)
			if profit > 0:
				W[current][next_state] = profit
			else:
				W[next_state][current] = profit * -1
	
	for i, row in enumerate(W): # normalize
		total = sum(row)
		for j, elem in enumerate(row):
			if total > 0:
				W[i][j] = elem / total

	pgrk = pagerank(W)
	best = np.argmax(pgrk)
	best = ('0'*len(G.nodes)+'{0:b}'.format(best))[-len(G.nodes):]
	# print(sum(pgrk), max(pgrk), best)

	return best

def show_net(G, labels):
	pos = nx.spectral_layout(G) # spring_layout planar_layout
	# nx.draw(G, pos=pos)
	# plt.savefig(f'{labels}_grey.png')
	# plt.clf()
	colors = ['red' if int(v) == 0 else 'blue' for v in labels]
	nx.draw(G, pos=pos,node_color=colors)
	plt.savefig(f'{labels}.png')

#################################################################################################
## Experiment 5: Loop State: Scatter InitPror x Alpha ###########################################

# local improvement: 
# 1) condition = one trial
# 2) K = number of nodes

# hedonic game:
# 1) condition=equilibrium not reached
# 2) K =1

# Important point: 	
# Generalized local-improvement
# Condition = equilibrium not reached
# K = number of nodes
# 6. Can we show a concrete example where this generalized local-improvement does not converge?

# def gen(K=1, condition='eq', C='hedonic'):
# 	while (condition):
# 		nodes = network.nodes[:K]
# 		want_move = []
# 		for node in nodes:
# 			if node crierion(C):
# 				want_move.append(node)
# 		for node in want_move:
# 			G = move(G, node)

# def run_algorithm(G, GT, noise=.5, alpha=1, W=None, alg='naive'):
# 	G = initialize(G, labels=add_noise(GT, noise))
# 	seconds = time()
# 	if alg == 'naive'   : answer = find_equilibrium_shuffle(G, alpha)
# 	seconds = time() - seconds
# 	acc = accuracy(GT, extract_labels(answer, label='cluster'))
# 	robustness = measure_robustness(answer)
# 	infos = (answer.clusters_nodes, answer.clusters_edges, len(answer.edges))
# 	return seconds, acc, robustness, infos, answer

def label2int(label):
	state, reverse = '', ''
	for s in label:
		state   += str(s)
		reverse += str(1-int(s))
	return min(int(state,2), int(reverse,2))

def converge(G, K=1, alpha=1, init=.5, tolerance=10):
	G = initialize(G, prob=init)
	moved, nodes_list, repeted = True, list(G.nodes), 0
	visited_states = []
	while moved is True:
		moved = False
		shuffle(nodes_list)
		want_move, simul = [], nodes_list[:K]
		for node in simul:
			if not satisfied_for_alpha(G, node, alpha):
				want_move.append(node)
		for node in want_move:
			G = move(G, node)
			moved = True
		if moved:
			representation = label2int(extract_labels(G, 'cluster'))
			if representation not in visited_states:
				visited_states.append(representation)
				repeted = 0
			else:
				repeted += 1
				if repeted >= tolerance:
					return False
	return True

def exp_5(alphas=5, inits=5, ps=5, insts=5, reps=5, in_cluster=50):
	states = pd.DataFrame(columns=['alpha','init_prop','avg'])
	went, total = 0, (alphas * inits * ps * insts * reps)
	for alpha in np.linspace(0,1,alphas):
		for init in np.linspace(0,.5,inits):
			values = []
			for p in np.linspace(.5, 1, ps):
				for inst in range(insts):
					G = PPG(2, in_cluster, p, 1-p)
					for rep in range(reps):
						print(f'{round(went/total*100,2)}%')
						if converge(G, K=len(G.nodes), alpha=alpha, init=init):
							values.append(0)
						else:
							values.append(1)
						went += 1
			states = states.append({
				'alpha' : alpha,
				'init_prop' : init,
				'avg' : np.mean(values)
			}, ignore_index=True)
	states.to_csv(f'exp5_a{alphas}_i{inits}_p{ps}_in{inst}_r{reps}_n{in_cluster}.csv')
	# sns.scatterplot(data=states, x="alpha", y="init_prop", hue="avg", palette='coolwarm', s=1000, marker='s') # sizes=[100]*len(states), markers=['s']*len(states)
	# plt.show()

# await sleep()

## Main ###############################################################################################

# Spell command:
# spell run --pip-req requirements.txt 'python run_experiments.py' # --machine-type cpu

if __name__ == "__main__":
	# exp_1()
	# exp_2()

	# exp_1(
	# 	noises=np.linspace(0,.5,3), multipliers=np.linspace(0.001,1,5), ps=np.linspace(.01,.1,3),
	# 	instances=3, repetitions=3, communities=2, nodes_per_cluster=50)

	# exp_2(
	# 	algorithms = ['onepass', 'hedonic'], noises=np.linspace(0,.5,5),
	# 	networks=get_real_nets(), repetitions = 10)

	# exp_cons()

	################################################

	# edges1 = [(0,1),(1,2),(2,3),(2,6),(4,5),(5,6),(6,7)]
	# G = nxgraph(edges1)
	# show_net(G, exp_markov(G))

	# p = .6
	# G = PPG(2, 5, .5, 1-p)
	# show_net(G, exp_markov(G))

	################################################

	equal = 11
	# exp_5(alphas=equal, inits=equal, ps=equal, insts=equal, reps=equal)
	exp_5(alphas=equal, inits=equal, ps=5, insts=5, reps=5)