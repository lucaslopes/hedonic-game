from random import random, shuffle
from time import time
import os
import numpy as np
import pandas as pd
import networkx as nx
from networkx.generators.community import planted_partition_graph as PPG

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

def load_ground_truth(G, values, label='gt', replace={'0':0,'1':1}):
	for node, cluster in zip(G.nodes, values):
		G.nodes[node][label] = replace[cluster]
	return G

def initialize(G, labels=[], prob=.5):
	G.clusters_nodes = [0, 0] # c1 = total_nodes - nodes_c0
	G.clusters_edges = [0, 0]
	for node in G.nodes:
		G.nodes[node]['here'] = 0
		if len(labels) == len(G.nodes):
			G.nodes[node]['cluster'] = int(labels[node])
		else:
			G.nodes[node]['cluster'] = 0 if random() < prob else 1
		G.clusters_nodes[G.nodes[node]['cluster']] += 1
	for edge in G.edges:
		n1, n2 = edge[0], edge[1]
		c_n1 = G.nodes[n1]['cluster']
		c_n2 = G.nodes[n2]['cluster']
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
	return [G.nodes[node][label] for node in G.nodes]

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

def run_algorithm(G, GT, noise=.5, alpha=1, alg='naive'):
	G = initialize(G, labels=add_noise(GT, noise))
	seconds = time()
	if alg == 'hedonic' : answer = almost_robust(G)
	if alg == 'naive'   : answer = find_equilibrium_shuffle(G, alpha)
	if alg == 'onepass' : answer = one_pass_neighbors(G)
	if alg == 'spectral': answer = initialize(G, labels=spectral(G))
	seconds = time() - seconds
	acc = accuracy(GT, extract_labels(answer, label='cluster'))
	robustness = measure_robustness(answer)
	infos = (answer.clusters_nodes, answer.clusters_edges, len(answer.edges))
	return seconds, acc, robustness, infos

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
	noises=np.linspace(0,.5,11), multipliers=np.linspace(0.001,1,11), ps=np.linspace(.01,.1,10),
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
					edge_density = len(G.edges) / max_edges_possible(len(G.nodes))
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

## Main ###############################################################################################

# Spell command:
# spell run --pip-req requirements.txt 'python planted_partition.py' # --machine-type cpu

if __name__ == "__main__":
	exp_1()
	exp_2()

	# exp_1(
	# 	noises=np.linspace(0,.5,3), multipliers=np.linspace(0.001,1,3), ps=np.linspace(.01,.1,3),
	# 	instances=3, repetitions=3, communities=2, nodes_per_cluster=50)

	# exp_2(
	# 	algorithms = ['onepass', 'hedonic'], noises=np.linspace(0,.5,2),
	# 	networks=get_real_nets(), repetitions = 10)