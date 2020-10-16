import os
import networkx as nx
import numpy as np
import igraph as ig
import partition_igraph
import pandas as pd
from networkx.generators.community import planted_partition_graph as PPG
from hedonic import Game
from time import time
from random import random, choice, shuffle, sample

#################################################################################################
## Functions #################################################################################################

def from_label_to_dict(labels):
	# funcoes de acc do gam recebem lista de sets, enquanto que nossa recebe lista com labels
	num_clusters = len(set(labels))
	clusters = [set() for _ in range(num_clusters)]
	for i, label in enumerate(labels):
		clusters[label].add(i)
	return {val:idx for idx, part in enumerate(clusters) for val in part}

def accuracies(G, answers=[], gt=[{}], methods=['rand','jaccard','mn','gmn','min','max','diff']):
	algs=['ml','ecg','hedonic','spectral']
	result = []
	for i, ans in enumerate(answers):
		acc = {}
		for mthd in methods:
			adj = False if mthd=="jaccard" else True # print('alg', algs[i])
			acc[mthd] = G.gam(ans,gt,method=mthd,adjusted=adj)
		result.append(acc)
	return result

# 'rand': the RAND index # Graph-Aware Rand
# 'jaccard': the Jaccard index # Jaccard Graph-Aware
# 'mn': pairwise similarity normalized with the mean function
# 'gmn': pairwise similarity normalized with the geometric mean function
# 'min': pairwise similarity normalized with the minimum function
# 'max': pairwise similarity normalized with the maximum function
# Each measure can be adjusted (recommended) 
def get_answers(g, algs=['ml','ecg','hedonic','spectral']):
	answers = {}
	for alg in algs:
		answers[alg], duration = {}, None
		begin = time()
		if alg is 'ml': answers[alg]['ans'] = g.community_multilevel()
		if alg is 'ecg': answers[alg]['ans'] = g.community_ecg(ens_size=32)
		if alg is 'hedonic': answers[alg]['ans'], duration = hedonic_solve_igrpah(g)
		if alg is 'spectral': answers[alg]['ans'], duration = spectral(g)
		answers[alg]['sec'] = duration if duration else time() - begin
	return answers

def get_file_name(folder='', outname='no_name'): # to output csv result
	outdir = './outputs/'
	if not os.path.exists(outdir):
		os.mkdir(outdir)
	outdir = f'./outputs/{folder}/'
	if not os.path.exists(outdir):
		os.mkdir(outdir)
	fullname = os.path.join(outdir, outname) 
	return fullname

# todo: nem sempre converge
def get_ppg_fully_connected(numComm, commSize, p, q):
	g = PPG(numComm, commSize, p, q) # g = ig.Graph.Famous('Zachary')
	nodes2connect = [set() for _ in range(len(set([g.nodes[node]['block'] for node in g.nodes()])))]
	for node in g.nodes(): # print(node, [friend for friend in g.neighbors(node)])
		if len(list(g.neighbors(node))) == 0:
			nodes2connect[g.nodes[node]['block']].add(node)
	sizes = [len(cluster) for cluster in nodes2connect] # print(nodes2connect)
	if sum(sizes) % 2 != 0: # even number of nodes
		has_friend = False
		while not has_friend:
			node = choice(range(len(g.nodes)))
			if len(list(g.neighbors(node))) > 0:
				has_friend = True
		nodes2connect[g.nodes[node]['block']].add(node)
		sizes = [len(cluster) for cluster in nodes2connect]
	while sum(sizes) > 0: # min argmax
		for i, size in enumerate(sizes):
			if size > 0:
				node = sample(nodes2connect[i],1)[0]
				others = [pos for pos, size in enumerate(sizes) if size > 0 and pos != i]
				if sizes[i] == 1 or (random() < q/(p+q) and len(others) > 0):
					j = choice(others)
					friend = sample(nodes2connect[j],1)[0]
					g.add_edge(node,friend)
					nodes2connect[i].remove(node)
					nodes2connect[j].remove(friend)
				else:
					friend = node
					while friend == node:
						friend = sample(nodes2connect[i],1)[0]
					g.add_edge(node,friend)
					nodes2connect[i].remove(node)
					nodes2connect[i].remove(friend)
				sizes = [len(cluster) for cluster in nodes2connect]
				break
	true_comm = [set(list(range(commSize*i, commSize*(i+1)))) for i in range(numComm)]
	GT = {val:idx for idx, part in enumerate(true_comm) for val in part}
	# convert graph 'G' from nx to igraph format:
	G = ig.Graph(directed=False)
	G.add_vertices(g.nodes())
	G.add_edges(g.edges())
	return G, GT

#################################################################################################
## Algoritms #################################################################################################

# todo: ver se ultimo level tem apenas 2 comunidades:
# g.community_multilevel(weights=weights, return_levels=True)[0].membership

# def solve_from_edgelist(edgelist):
# 	clusters = [{},{}]
# 	for n0, n1 in edgelist:
# 		try:    nodes[n0].append(n1)
# 		except: nodes[n0] = [n1]
# 		try:    nodes[n1].append(n0)
# 		except: nodes[n1] = [n0]
	
# 		# [{n0,n1},{}]
# 		# [{},{n0,n1}]
# 		# [{n0},{n1}]
# 		# [{n1},{n0}]

def hedonic_solve_igrpah(g):
	game = Game(g.get_edgelist())
	duration = time()
	game.play()
	duration = time() - duration
	return from_label_to_dict(game.labels), duration

def spectral(G):
	# A = nx.adjacency_matrix(G).toarray() # networkx
	A = np.array(list(G.get_adjacency())) # igraph
	duration = time()
	# w, v = np.linalg.eigh(A)
	# ew_2 = v[:,-2]
	# labels = np.array([1 if x > 0 else 0 for x in ew_2], dtype='int')
	labels = np.array([1 if x > 0 else 0 for x in np.linalg.eigh(A)[1][:,-2]], dtype='int')
	duration = time() - duration
	return from_label_to_dict(labels), duration

# def local_improvement(G):
# 	want_move = []
# 	for node in G.nodes:
# 		here, there, dis_here, dis_there = get_node_atributes(G, node)
# 		if here < there:
# 			want_move.append(node)
# 	for node in want_move:
# 		move(G, node)
# 	return G

#################################################################################################
## Compare Time and Accuracy: Hedonic vs Spectral vs Louvain vs ECG #############################

def compare(noises=np.linspace(.5,.5,1),
	multipliers=np.concatenate(([.001], np.linspace(0,1,11)[1:])),
	ps=np.linspace(.01,.1,11), instances=11, repetitions=11, numComm=2, commSize=111):

	total = len(noises) * len(multipliers) * len(ps) * instances * repetitions
	went  = 0
	begin = time()
	print(f'\n\nComparison Experiment - begin at:{begin} -- TOTAL = {total}\n\n')
	for noi in noises:
		# columns = x=q/p, y=accuracy, hue=algorithm, method(each plot)
		df_results = pd.DataFrame(columns=['p_in','mult','instance','repetition','algorithm','accuracy','method','seconds'])
		for mult in multipliers:
			for p in ps:
				for i in range(instances):
					G, GT = get_ppg_fully_connected(numComm, commSize, p, p*mult)
					# spectral_answered, spectral_answer = False, None
					for r in range(repetitions):
						print(f'% = {round(went/total*100, 2)}%\tNoise = {np.where(noises==noi)[0][0]+1}/{len(noises)}\tMult = {np.where(multipliers==mult)[0][0]+1}/{len(multipliers)}\tP = {np.where(ps==p)[0][0]+1}/{len(ps)}\tInst = {i+1}/{instances}\tRep = {r+1}/{repetitions}')
						# if not spectral_answered:
						# 	spectral_answer = run_algorithm(G, GT, alg='spectral')
						# onepass_answer = run_algorithm(G, GT, noise=noi, alg='onepass')
						# hedonic_answer = run_algorithm(G, GT, noise=noi, alg='naive') # here
						answers = get_answers(G)
						ans_order = list(answers)
						seconds = [answers[alg]['sec'] for alg in ans_order]
						answers = [answers[alg]['ans'] for alg in ans_order]
						scores = accuracies(G, answers, GT)
						for alg, score, sec in zip(ans_order, scores, seconds):
							for mthd, acc in score.items():
								df_results = df_results.append({ # todo: colocar segundos, infos e robustez?
									'p_in':p,
									'mult':mult,
									'instance':i,
									'repetition':r,
									'algorithm':alg,
									'accuracy':acc,
									'method': mthd,
									'seconds':sec }, ignore_index=True)
						went += 1
						# infos and robustness
		# df_results['q_over_p'] = df_results['p_out'] / df_results['p_in']
		df_results.to_csv(get_file_name('comparisons', f'comparison_commSize={commSize}.csv'), index=False)
	print('\n\n\nFINISHED EXP COMPARISON!', time()-begin)

#################################################################################################
## Main #########################################################################################

# spell run --pip-req requirements.txt 'python compare.py'

if __name__ == "__main__":
	compare()

	# G, GT = get_ppg_fully_connected(2, 50, .2, .05)
	# print(list(G.get_edgelist()))
	# solve_from_edgelist(list(G.get_edgelist()))