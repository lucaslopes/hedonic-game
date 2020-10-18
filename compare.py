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

def accuracies(G, answers=[], gt=[{}], methods=['rand','jaccard','mn','gmn','min','max','dist']):
	# algs=['ml','ecg','hedonic','spectral']
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
def get_answers(g, algs=['ml','ecg','hedonic','spectral'], spectral_ans=None): # naive
	answers = {}
	for alg in algs:
		answers[alg], duration = {}, None
		begin = time()
		if alg is 'ml':       answers[alg]['ans'] = g.community_multilevel()
		if alg is 'ecg':      answers[alg]['ans'] = g.community_ecg() # ens_size=32
		if alg is 'hedonic':  answers[alg]['ans'], duration, answers[alg]['rob'] = hedonic_solve_igrpah(g)
		if alg is 'naive':    answers[alg]['ans'], duration, answers[alg]['rob'] = hedonic_solve_igrpah(g, naive=True)
		if alg is 'spectral': answers[alg]['ans'], duration = spectral_ans['ans'], spectral_ans['time']
		answers[alg]['sec'] = duration if duration else time() - begin
	for alg in [a for a in algs if a == 'ml' or a == 'ecg']:
		answers[alg]['ans'] = two_communities(g, answers[alg]['ans'])
	# for alg in ['ml_s1','ml_s2','ecg_s1','ecg_s2']: # [a for a in algs if a == 'ml' or a == 'ecg']:
	# 	ref = alg.split('_')[0]
	# 	answers[alg] = {'ans':None, 'sec':answers[ref]['sec']}
	# 	if alg[-1] == '1': answers[alg]['ans'] = two_communities_old(g, answers[ref]['ans'])
	# 	if alg[-1] == '2': answers[alg]['ans'] = two_communities(g, answers[ref]['ans'])
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

def get_ppg_max_components(numComm, commSize, p, q):
	g = PPG(numComm, commSize, p, q)
	g = nx.convert_node_labels_to_integers(g.subgraph(max(nx.connected_components(g), key=len)))
	GT = {}
	for node in g.nodes:
		GT[node] = (g.nodes[node]['block'])
	# convert graph 'G' from nx to igraph format:
	G = ig.Graph(directed=False)
	G.add_vertices(g.nodes())
	G.add_edges(g.edges())
	return G, GT

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

def subset_sum(numbers, target, partial=[]):
	s = sum(partial)

	# check if the partial sum is equals to target
	if s == target:
		print("sum(%s)=%s" % (partial, target))
	if s >= target:
		return  # if we reach the number why bother to continue

	for i in range(len(numbers)):
		n = numbers[i]
		remaining = numbers[i + 1:]
		subset_sum(remaining, target, partial + [n])

def two_communities_reciprocity(G, clusters):
	now = time()
	while len(clusters) > 2:
		modul_pairs = [[0 for _ in range(len(clusters))] for _ in range(len(clusters))]
		for i in range(len(clusters)-1):
			for j in range(i+1, len(clusters)):
				new_membership = [i if g == j else g for g in clusters.membership]
				max_cluster = max(set(new_membership))
				if j < max_cluster:
					new_membership = [j if g == max_cluster else g for g in new_membership]
				modul = ig.clustering.VertexClustering(G, new_membership).modularity
				modul_pairs[i][j] = modul
				modul_pairs[j][i] = modul
		joins = []
		for i, row in enumerate(modul_pairs):
			want = np.argmax(row)
			if np.argmax(modul_pairs[want]) == i:
				pair = (min(i,want), max(i,want))
				if pair not in joins:
					joins.append(pair)
		best_membership = []
		for i, j in joins:
			best_membership = [i if g == j else g for g in clusters.membership]
			max_cluster = max(set(best_membership))
			if j < max_cluster:
				best_membership = [j if g == max_cluster else g for g in best_membership]
		clusters = ig.clustering.VertexClustering(G, best_membership)
	print(time()-now)
	return clusters

def two_communities(G, clusters):
	# now = time()
	while len(clusters) > 2:
		best_modularity, best_membership = 0, None
		for i in range(len(clusters)-1):
			for j in range(i, len(clusters)):
				new_membership = [i if g == j else g for g in clusters.membership]
				max_cluster = max(set(new_membership))
				if j < max_cluster:
					new_membership = [j if g == max_cluster else g for g in new_membership]
				ans = ig.clustering.VertexClustering(G, new_membership)
				if ans.modularity > best_modularity:
					best_modularity = ans.modularity
					best_membership = new_membership
		clusters = ig.clustering.VertexClustering(G, best_membership)
	# print(time()-now)
	return clusters

def two_communities_old(G, clusters): # compara todas as possiveis combinações de 2 grupos
	now = time()
	best_modularity, answer = 0, None
	if len(clusters) > 2:
		combinations = []
		for i in range(2**(len(clusters)-1)): # fica inviavel pra muitos clusters, então unir o par que resulta na maior modularidade até ter apenas 2 clusters
			combinations.append(('0'*len(clusters)+f'{i:b}')[-len(clusters):])
		combinations = combinations[1:]
		for comb in combinations:
			groups = [[],[]]
			for i, g in enumerate(comb):
				groups[int(g)] += clusters[i]
			membership = [0 for _ in range(len(clusters.membership))]
			g = 0 if len(groups[0]) < len(groups[1]) else 1
			for node in groups[g]:
				membership[node] = 1
			ans = ig.clustering.VertexClustering(G, membership)
			if ans.modularity > best_modularity:
				best_modularity = ans.modularity
				answer = ans
	# print(time()-now)
	return answer if answer else clusters

#################################################################################################
## Algoritms #################################################################################################

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

def hedonic_solve_igrpah(g, naive=False):
	game = Game(g.get_edgelist())
	duration = time()
	game.play(naive=naive)
	duration = time() - duration
	return from_label_to_dict(game.labels), duration, game.calc_robustness()

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

def compare(multipliers=np.concatenate(([.05], np.linspace(0,1,6)[1:])),
	ps=5, instances=30, repetitions=30, numComm=2, commSize=50): # noises=, #np.linspace(.5,.5,1)

	total = len(multipliers) * ps * instances * repetitions # len(noises) 
	went  = 0
	begin = time()
	print(f'\n\nComparison Experiment - begin at:{begin} -- TOTAL = {total}\n\n')
# for noi in noises:
	# columns = x=q/p, y=accuracy, hue=algorithm, method(each plot)
	columns={'p_in':[], 'mult':[], 'instance':[], 'repetition':[], 'algorithm':[], 'accuracy':[], 'robustness':[], 'method':[], 'seconds':[]}
	for mult in multipliers:
		for i_p, p in enumerate(np.linspace(.1,1,ps)): # default: .01 to .1
			for i in range(instances):
				# G, GT = get_ppg_fully_connected(numComm, commSize, p, p*mult)
				G, GT = get_ppg_max_components(numComm, commSize, p, p*mult)
				spectral_ans = None
				for r in range(repetitions):
					print(f'% = {round(went/total*100, 2)}%\tMult = {np.where(multipliers==mult)[0][0]+1}/{len(multipliers)}\tP = {i_p+1}/{ps}\tInst = {i+1}/{instances}\tRep = {r+1}/{repetitions}')
					if not spectral_ans:
						spectral_ans = {}
						spectral_ans['ans'], spectral_ans['time'] = spectral(G)
					answers = get_answers(G, spectral_ans=spectral_ans) #  algs=['hedonic','naive'] , algs=['ml','ecg']
					ans_order = list(answers)
					seconds = [answers[alg]['sec'] for alg in ans_order]
					robust  = [answers[alg]['rob'] if 'rob' in answers[alg] else 0 for alg in ans_order] # only hedonics
					answers = [answers[alg]['ans'] for alg in ans_order]
					scores = accuracies(G, answers, GT) # , methods=['dist']
					# print(scores)
					for alg, score, rob, sec in zip(ans_order, scores, robust, seconds):
						for mthd, acc in score.items():
							columns['p_in'].append(p)
							columns['mult'].append(mult)
							columns['instance'].append(i)
							columns['repetition'].append(r)
							columns['algorithm'].append(alg)
							columns['accuracy'].append(acc)
							columns['robustness'].append(rob)
							columns['method'].append(mthd)
							columns['seconds'].append(sec)
					went += 1
	df_results = pd.DataFrame()
	for col, values in columns.items():
		df_results[col] = values
	df_results.to_csv(f'max_components__ps={ps}_mults={len(multipliers)}_inst={instances}_reps={repetitions}_nComm={numComm}_commSize={commSize}.csv', index=False) # get_file_name('comparisons', f'comparison_commSize={commSize}.csv'
	print('\n\n\nFINISHED EXP COMPARISON!', time()-begin)

#################################################################################################
## Main #########################################################################################

# spell run --pip-req requirements.txt 'python compare.py'

if __name__ == "__main__":
	compare()

	# G, GT = get_ppg_fully_connected(2, 111, .05, .01)
	# print(list(G.get_edgelist()))
	# solve_from_edgelist(list(G.get_edgelist()))

	# a = G.community_multilevel()
	# a = two_communities(G, a)
	# print(accuracies(G, [a], GT))
