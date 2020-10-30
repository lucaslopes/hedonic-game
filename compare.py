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

from run_experiments import run_algorithm

#################################################################################################
## Functions #################################################################################################

def from_label_to_dict(labels):
	# funcoes de acc do gam recebem lista de sets, enquanto que nossa recebe lista com labels
	dict_labels = {}
	for node, label in enumerate(labels):
		dict_labels[node] = label
	return dict_labels

def from_dict_to_label(dict_label):
	labels = [0] * len(dict_label)
	for node, label in dict_label.items():
		labels[node] = label
	return labels

def accuracy(x,y): # WARNING: x and y must be only 0 or 1
	x, y = np.array(x), np.array(y)
	if len(x) != len(y):
		print(len(x), len(y))
		raise ValueError('x and y must be arrays of the same size')
	matches1 = sum(x == y)
	matches2 = sum(x == 1-y)
	score = max([matches1, matches2]) / len(x)
	return score

# 'rand': the RAND index # Graph-Aware Rand
# 'jaccard': the Jaccard index # Jaccard Graph-Aware
# 'mn': pairwise similarity normalized with the mean function
# 'gmn': pairwise similarity normalized with the geometric mean function
# 'min': pairwise similarity normalized with the minimum function
# 'max': pairwise similarity normalized with the maximum function
# Each measure can be adjusted (recommended) 
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

# spectral, hedonic, ml
# hed_ecg_hed, hed_ecg_spec, hed_ecg_ml
# ml_ecg_ml, ml_ecg_hed, ml_ecg_spec
def get_answers(g, algs=['spectral', 'hedonic', 'ml', 'ecg', 'local improve'], spectral_ans=None, init_labels=None): # naive algs=['ml','ecg','hedonic','spectral']    'hed_ecg_hed', 'hed_ecg_spec', 'hed_ecg_ml', 'ml_ecg_ml', 'ml_ecg_hed', 'ml_ecg_spec'
	answers = {}
	for alg in algs:
		answers[alg], duration = {}, None
		if alg == 'hedonic': answers[alg]['ans'], duration, answers[alg]['rob'] = hedonic_solve_igraph(g, init_labels)
		if alg == 'naive': answers[alg]['ans'], duration, answers[alg]['rob'] = hedonic_solve_igraph(g, init_labels, naive=True)
		if alg == 'local improve': answers[alg]['ans'], duration = local_improvement(g, init_labels)
		if alg == 'spectral': answers[alg]['ans'], duration = spectral_ans['ans'], spectral_ans['time']
		begin = time()
		if alg == 'ml': answers[alg]['ans'] = g.community_multilevel()
		if alg == 'ecg': answers[alg]['ans'] = g.community_ecg(ens_size=32) # ens_size=32 default=16
		answers[alg]['sec'] = duration if duration else time() - begin
	for alg in [a for a in algs if a == 'ml' or a == 'ecg']:
		answers[alg]['ans'] = two_communities(g, answers[alg]['ans'])
	# for alg in ['ml_s1','ml_s2','ecg_s1','ecg_s2']: # [a for a in algs if a == 'ml' or a == 'ecg']:
	# 	ref = alg.split('_')[0]
	# 	answers[alg] = {'ans':None, 'sec':answers[ref]['sec']}
	# 	if alg[-1] == '1': answers[alg]['ans'] = two_communities_old(g, answers[ref]['ans'])
	# 	if alg[-1] == '2': answers[alg]['ans'] = two_communities(g, answers[ref]['ans'])
	return answers

def load_ground_truth(G, values, label='gt', replace={'0':0,'1':1}):
	for node, cluster in zip(G.nodes, values):
		G.nodes[node][label] = replace[cluster]
	return G

def get_file_name(folder='', outname='no_name'): # to output csv result
	outdir = './outputs/'
	if not os.path.exists(outdir):
		os.mkdir(outdir)
	outdir = f'./outputs/{folder}/'
	if not os.path.exists(outdir):
		os.mkdir(outdir)
	fullname = os.path.join(outdir, outname) 
	return fullname

def convert_from_netx_to_igraph(netx_G, label='block'): # convert graph 'G' from nx to igraph format:
	GT = {}
	for node in netx_G.nodes:
		GT[node] = (netx_G.nodes[node][label])
	G = ig.Graph(directed=False)
	G.add_vertices(netx_G.nodes())
	G.add_edges(netx_G.edges())
	return G, GT

def get_ppg_max_components(numComm, commSize, p, q, netx=False):
	g = PPG(numComm, commSize, p, q)
	infos = {'nodes':len(g.nodes), 'edges':len(g.edges)}
	g = nx.convert_node_labels_to_integers(g.subgraph(max(nx.connected_components(g), key=len)))
	infos['max_comp'] = len(g.nodes)
	infos['gt_balance'] = [g.nodes[node]['block'] for node in g.nodes].count(0)/len(g.nodes)
	if netx: # networkx
		return (g, [g.nodes[node]['block'] for node in g.nodes]), infos
	else: # igraph
		return convert_from_netx_to_igraph(g), infos

def get_ppg_fully_connected(numComm, commSize, p, q, netx=False):
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
	
	infos = {'nodes':len(g.nodes), 'max_comp':len(g.nodes), 'edges':len(g.edges), 'gt_balance':[g.nodes[node]['block'] for node in g.nodes].count(0)/len(g.nodes)}
	if netx: # networkx
		return (g, [g.nodes[node]['block'] for node in g.nodes]), infos
	else: # igraph
		return convert_from_netx_to_igraph(g), infos

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
	print('clusters', clusters)
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

def apply_noise(labels, noise=.5):
	membership = [0] * len(labels)
	for n, l in labels.items():
		membership[n] = l 
	return [1 - label if random() < noise else label for label in membership]

#################################################################################################
## Algoritms #################################################################################################

def edge_weights_to_AW(g, w):
	Aw = np.zeros((g.vcount(),g.vcount()))
	for (n0, n1), w in zip(g.get_edgelist(), w):
		Aw[n0][n1] = w
		Aw[n1][n0] = w
	return Aw

def get_memberships(g, alg='ml', ens_size=16):
	memberships = []
	for _ in range(ens_size):
		if alg == 'hedonic':
			memberships.append(hedonic_solve_igraph(g, only_membership=True))
		else:
			memberships.append(g.community_multilevel(return_levels=True)[0].membership) # default
			# memberships.append(two_communities(g, g.community_multilevel(return_levels=True)[0]).membership)
	return memberships

def get_ecg_matrix(g, memberships, min_weight=0.05):
	W = [0] * g.ecount()
	for ms in memberships:
		b = [ms[n0]==ms[n1] for n0, n1 in g.get_edgelist()]
		W = [W[i]+b[i] for i in range(len(W))]
	W = [min_weight + (1-min_weight)*W[i]/len(memberships) for i in range(len(W))]
	core = g.shell_index()
	ecore = [min(core[x.tuple[0]],core[x.tuple[1]]) for x in g.es] ## Force min_weight outside 2-core
	return [W[i] if ecore[i]>1 else min_weight for i in range(len(ecore))]

def get_answer_from_weights(g, w, alg='ml'):
	if alg == 'hedonic':
		return hedonic_solve_weighted(g, edge_weights_to_AW(g, w))[0]
	elif alg == 'spectral':
		return spectral(g, edge_weights_to_AW(g, w))[0]
	else:
		return two_communities(g, g.community_multilevel(weights=w)) # default

def alg_with_ecg(g, alg1='ml', alg2='ml', size=16):
	return get_answer_from_weights(g, get_ecg_matrix(g, get_memberships(g, alg=alg1, ens_size=size)), alg=alg2)

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

def hedonic_solve_networkx(g, naive=False):
	game = Game(g.edges())
	duration = time()
	game.play(naive=naive)
	duration = time() - duration
	return game.labels, duration, game.calc_robustness()

def hedonic_solve_weighted(g, W):
	game = Game(g.get_edgelist())
	duration = time()
	game.hedonic_weighted(W)
	duration = time() - duration
	return from_label_to_dict(game.labels), duration, game.calc_robustness()

def hedonic_solve_igraph(g, init_labels=[], naive=False, only_membership=False):
	game = Game(g.get_edgelist())
	game.set_labels(init_labels)
	duration = time()
	game.play(naive=naive)
	duration = time() - duration
	eq = game.in_equilibrium_for(inspect=True)
	if not eq: # Walrus Operator :=
		print(f'game is not in equilibrium for alpha=edge density ({eq}')
	if only_membership:
		return game.labels
	else:
		return from_label_to_dict(game.labels), duration, game.calc_robustness()

def spectral(G, A=None, netx=False):
	if A is None:
		if netx:
			A = nx.adjacency_matrix(G).toarray() # networkx
		else:
			A = np.array(list(G.get_adjacency())) # igraph
	duration = time()
	# w, v = np.linalg.eigh(A)
	# ew_2 = v[:,-2]
	# labels = np.array([1 if x > 0 else 0 for x in ew_2], dtype='int')
	labels = np.array([1 if x > 0 else 0 for x in np.linalg.eigh(A)[1][:,-2]], dtype='int')
	duration = time() - duration
	return from_label_to_dict(labels), duration

def local_improvement(G, labels, prob=.5, only_membership=False):
	if len(labels) != len(G.vs):
		print('local_improvement: labels has different size of number of vertices.')
		labels = [0 if prob > random() else 1 for _ in range(len(G.vs))]
	# init = [l if labels[0] == 0 else 1 - l for l in labels]
	duration = time()
	want_move = []
	for node in G.vs.indices:
		here, there = 0, 0
		for friend in G.neighbors(node):
			if labels[friend] == labels[node]:
				here += 1
			else:
				there += 1
		if there > here:
			want_move.append(node)
	# print(want_move)
	for node in want_move:
		labels[node] = 1 - labels[node]
	# final = [l if labels[0] == 0 else 1 - l for l in labels]
	duration = time() - duration
	if only_membership:
		return labels
	else:
		return from_label_to_dict(labels), duration#, (init,want_move,final)

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
## Compare speed: Networkx vs Python naive ######################################################

def speed_test(multipliers=np.concatenate(([.05], np.linspace(0,1,6)[1:])),
	ps=5, instances=10, repetitions=10, numComm=2, commSize=250):
	
	total = len(multipliers) * ps * instances * repetitions # len(noises) 
	went  = 0
	begin = time()
	print(f'\n\nComparison Experiment - begin at:{begin} -- TOTAL = {total}\n\n')

	cols = {'ps':[], 'mults':[], 'secs':[], 'alg':[]}
	for mult in multipliers:
		for i_p, p in enumerate(np.linspace(.01,.1,ps)): # ps
			for i in range(instances):
				G, GT = get_ppg_fully_connected(numComm, commSize, p, p*mult, netx=True)
				for r in range(repetitions):
					print(f'% = {round(went/total*100, 2)}%\tMult = {np.where(multipliers==mult)[0][0]+1}/{len(multipliers)}\tP = {i_p+1}/{ps}\tInst = {i+1}/{instances}\tRep = {r+1}/{repetitions}')
					sec0 = run_algorithm(G, GT, alg='hedonic')[0]
					sec1 = hedonic_solve_networkx(G)[1]
					for alg, sec in zip(['netx','python'], [sec0,sec1]):
						cols['ps'].append(p)
						cols['mults'].append(mult)
						cols['secs'].append(sec)
						cols['alg'].append(alg)
					went += 1
	df_results = pd.DataFrame()
	for col, values in cols.items():
		df_results[col] = values
	df_results.to_csv(f'speed_test__ps={ps}_mults={len(multipliers)}_inst={instances}_reps={repetitions}_nComm={numComm}_commSize={commSize}.csv', index=False) # get_file_name('comparisons', f'comparison_commSize={commSize}.csv'

#################################################################################################
## Compare Time and Accuracy: Hedonic vs Spectral vs Louvain vs ECG #############################

def compare(with_noise=True, multipliers=np.concatenate(([.05], np.linspace(0,1,11)[1:])),
	ps=np.linspace(.01,.1,5), instances=20, repetitions=20, numComm=2, commSize=250, output_name='with_noises_fix'): # noises=, #np.linspace(.5,.5,1)

	if with_noise:
		noises = [0,.025]+list(np.linspace(0,.5,11))[1:-1]+[.475,.5]
		noises = [round(noi, 3) for noi in noises]
	else:
		noises = [.5]

	total = len(noises) * len(multipliers) * len(ps) * instances * repetitions # len(noises) 
	went  = 0
	begin = time()
	print(f'\n\nComparison Experiment - begin at:{begin} -- TOTAL = {total}\n\n')
	# columns = x=q/p, y=accuracy, hue=algorithm, method(each plot)
	columns={'nodes':[], 'max_comp':[], 'edges':[], 'gt_balance':[], 'p_in':[], 'mult':[], 'instance':[], 'repetition':[], 'noise':[], 'algorithm':[], 'accuracy':[], 'robustness':[], 'method':[], 'seconds':[]}
	for mult in multipliers:
		for i_p, p in enumerate(ps): # default: .01 to .1
			for i in range(instances):
				# (G, GT), infos = get_ppg_fully_connected(numComm, commSize, p, p*mult)
				(G, GT), infos = get_ppg_max_components(numComm, commSize, p, p*mult)
				spectral_ans = spectral(G)
				ml_anss, ecg_anss = [], []
				for noi, noise in enumerate(noises):
					for r in range(repetitions):
						print(f'% = {round(went/total*100, 2)}%\tMult = {np.where(multipliers==mult)[0][0]+1}/{len(multipliers)}\tP = {i_p+1}/{len(ps)}\tInst = {i+1}/{instances}\tRep = {r+1}/{repetitions}\tNoi = {noi+1}/{len(noises)}')
						if noi == 0: # algs=['spectral', 'hedonic', 'ml', 'ecg', 'local improve']
							answers = get_answers(G, algs=['hedonic', 'ml', 'ecg', 'local improve'], init_labels=apply_noise(GT, noise)) # algs=['hedonic','naive'] , algs=['ml','ecg']
							ml_anss.append((answers['ml']['ans'],answers['ml']['sec']))
							ecg_anss.append((answers['ecg']['ans'],answers['ecg']['sec']))
						else:
							answers = get_answers(G, algs=['hedonic', 'local improve'], spectral_ans=spectral_ans, init_labels=apply_noise(GT, noise)) # algs=['hedonic','naive'] , algs=['ml','ecg']
							answers['ml'] = {'ans':ml_anss[r][0], 'sec':ml_anss[r][1]}
							answers['ecg'] = {'ans':ecg_anss[r][0], 'sec':ecg_anss[r][1]}
						answers['spectral'] = {'ans':spectral_ans[0], 'sec':spectral_ans[1]}
						answers[f'hedonic_n{noise}'] = answers.pop('hedonic')
						answers[f'local improve_n{noise}'] = answers.pop('local improve')
						ans_order = list(answers)
						seconds = [answers[alg]['sec'] for alg in ans_order]
						robust  = [answers[alg]['rob'] if 'rob' in answers[alg] else 0 for alg in ans_order] # only hedonics
						answers = [answers[alg]['ans'] for alg in ans_order]
						scores = accuracies(G, answers, GT, methods=['dist','jaccard','rand']) # , methods=['dist']
						# print(scores)
						for alg, score, rob, sec in zip(ans_order, scores, robust, seconds):
							for mthd, acc in score.items():
								columns['nodes'].append(infos['nodes'])
								columns['max_comp'].append(infos['max_comp'])
								columns['edges'].append(infos['edges'])
								columns['gt_balance'].append(infos['gt_balance'])
								columns['p_in'].append(p)
								columns['mult'].append(mult)
								columns['instance'].append(i)
								columns['repetition'].append(r)
								columns['noise'].append(noise)
								columns['algorithm'].append(alg)
								columns['accuracy'].append(acc)
								columns['robustness'].append(rob)
								columns['method'].append(mthd)
								columns['seconds'].append(sec)
						went += 1
	df_results = pd.DataFrame()
	for col, values in columns.items():
		df_results[col] = values
	df_results.to_csv(f'{output_name}__ps={len(ps)}_mults={len(multipliers)}_inst={instances}_reps={repetitions}_noises={len(noises)}_nComm={numComm}_commSize={commSize}.csv', index=False) # get_file_name('comparisons', f'comparison_commSize={commSize}.csv'
	print('\n\n\nFINISHED EXP COMPARISON!', time()-begin)

#################################################################################################
## Real Nets #############################

def get_real_nets(nets=['karate', 'dolphins', 'pol_blogs', 'pol_books']):
	real_nets = {}
	for net in nets:
		csv = pd.read_csv(f'real_nets/csv/{net}.csv', names=['A', 'B'])
		gt  = pd.read_csv(f'real_nets/gt/{net}.csv',  names=['N', 'GT'])
		clusters = gt['GT'].unique()
		G = nx.from_pandas_edgelist(csv, 'A', 'B')
		G = load_ground_truth(G, gt['GT'].values, label='GT', replace={clusters[0]:0, clusters[1]:1})
		# colors = ['red' if G.nodes[node]['gt'] == 0 else 'blue' for node in G.nodes]
		# nx.draw(G, pos=nx.spring_layout(G), node_color=colors) # , node_color=colors, alpha=alpha, width=width, node_size=sizes, edge_color=edge_color
		# plt.show()
		G, GT = convert_from_netx_to_igraph(G, label='GT')
		real_nets[net] = (G, GT)
	return real_nets

def compare_real_nets(networks=get_real_nets(), repetitions=100, with_noise=True, output_name='real_nets_fix'): # noises=, #np.linspace(.5,.5,1)

	if with_noise:
		noises = [0,.025]+list(np.linspace(0,.5,11))[1:-1]+[.475,.5]
		noises = [round(noi, 3) for noi in noises]
	else:
		noises = [.5]

	total = len(networks) * repetitions * len(noises) # len(noises) 
	went  = 0
	begin = time()
	print(f'\n\nComparison Experiment - begin at:{begin} -- TOTAL = {total}\n\n')
	# columns = x=q/p, y=accuracy, hue=algorithm, method(each plot)
	columns={'network':[], 'repetition':[], 'noise':[], 'algorithm':[], 'accuracy':[], 'robustness':[], 'method':[], 'seconds':[]}  # 'nodes':[], 'max_comp':[], 'edges':[], 'gt_balance':[],

	for i, net in enumerate(list(networks)):
		G, GT = networks[net]
		spectral_ans = spectral(G)
		ml_anss, ecg_anss = [], []
		for noi, noise in enumerate(noises):
			for r in range(repetitions):
				print(f'% = {round(went/total*100, 2)}%\tNet = {i+1}/{len(networks)}\tRep = {r+1}/{repetitions}\tNoi = {noi+1}/{len(noises)}')
				if noi == 0: # algs=['spectral', 'hedonic', 'ml', 'ecg', 'local improve']
					answers = get_answers(G, algs=['hedonic', 'ml', 'ecg', 'local improve'], init_labels=apply_noise(GT, noise)) # algs=['hedonic','naive'] , algs=['ml','ecg']
					ml_anss.append((answers['ml']['ans'],answers['ml']['sec']))
					ecg_anss.append((answers['ecg']['ans'],answers['ecg']['sec']))
				else:
					answers = get_answers(G, algs=['hedonic', 'local improve'], spectral_ans=spectral_ans, init_labels=apply_noise(GT, noise)) # algs=['hedonic','naive'] , algs=['ml','ecg']
					answers['ml'] = {'ans':ml_anss[r][0], 'sec':ml_anss[r][1]}
					answers['ecg'] = {'ans':ecg_anss[r][0], 'sec':ecg_anss[r][1]}
				answers['spectral'] = {'ans':spectral_ans[0], 'sec':spectral_ans[1]}
				answers[f'hedonic_n{noise}'] = answers.pop('hedonic')
				answers[f'local improve_n{noise}'] = answers.pop('local improve')
				ans_order = list(answers)
				seconds = [answers[alg]['sec'] for alg in ans_order]
				robust  = [answers[alg]['rob'] if 'rob' in answers[alg] else 0 for alg in ans_order] # only hedonics
				answers = [answers[alg]['ans'] for alg in ans_order]
				scores = accuracies(G, answers, GT, methods=['dist','jaccard','rand']) # , methods=['dist']
				# print(scores)
				for alg, score, rob, sec in zip(ans_order, scores, robust, seconds):
					for mthd, acc in score.items():
						# columns['nodes'].append(infos['nodes'])
						# columns['max_comp'].append(infos['max_comp'])
						# columns['edges'].append(infos['edges'])
						# columns['gt_balance'].append(infos['gt_balance'])
						columns['network'].append(net)
						columns['repetition'].append(r)
						columns['noise'].append(noise)
						columns['algorithm'].append(alg)
						columns['accuracy'].append(acc)
						columns['robustness'].append(rob)
						columns['method'].append(mthd)
						columns['seconds'].append(sec)
				went += 1
	df_results = pd.DataFrame()
	for col, values in columns.items():
		df_results[col] = values
	df_results.to_csv(f'{output_name}__networks={len(networks)}_reps={repetitions}_noises={len(noises)}.csv', index=False) # get_file_name('comparisons', f'comparison_commSize={commSize}.csv'
	print('\n\n\nFINISHED EXP COMPARISON!', time()-begin)

#################################################################################################
## Main #########################################################################################

def test_acc_realnet():
	G, GT = get_real_nets(nets=['karate'])['karate']
	# (G, GT), infos = get_ppg_max_components(2, 500, .02, .01)
	game = Game(G.get_edgelist())
	noise = .5
	results, res_local = [], []
	for _ in range(1000):
		init_labels = apply_noise(GT, noise)
		init_labels = [l if init_labels[0] == 0 else 1 - l for l in init_labels]
		# init_labels_copy = [l for l in init_labels]
		# print('0 obvio:', init_labels_copy == init_labels)
		# print('init_labels:', init_labels)
		# print('noise:', accuracy(from_dict_to_label(GT), init_labels), G.gam(from_label_to_dict(init_labels),GT,method='dist',adjusted=True))
		game.set_labels(init_labels)
		# labls = [l if labls[0] == 0 else 1 - l for l in labls]
		# print('init eq:', labls == init_labels)
		# print('labls:', labls)
		game.play(naive=True)
		eq = game.in_equilibrium_for(inspect=True)
		if not eq: # Walrus Operator :=
			print(f'game is not in equilibrium for alpha=edge density ({eq}')
		ans_hedonic = from_label_to_dict(game.labels)
		results.append(accuracies(G, [ans_hedonic], GT, methods=['dist'])[0]['dist'])
		# print('hed init', hed_infos[0])
		# print('hed eq:', labls == hed_infos[0])
		# print('2 init eq:', labls == init_labels)
		# print('o obvio:', init_labels_copy == init_labels)
	
		ans_local, _ = local_improvement(G, init_labels)
		res_local.append(accuracies(G, [ans_local], GT, methods=['dist'])[0]['dist'])

		# hed_infos[1].sort()
		# local_infos[1].sort()
		# print('inits equals?', hed_infos[0] == local_infos[0])
		# print('noves moved equals?', hed_infos[1] == local_infos[1])
		# print('final equals?', hed_infos[2] == local_infos[2])

		# print(ans_hedonic == ans_local)
		# print(accuracy(from_dict_to_label(ans_hedonic), from_dict_to_label(ans_local)))

		# scores = accuracies(G, [ans_hedonic, ans_local], GT, methods=['dist']) # , methods=['dist']
		# print(scores)
	print('hedonic:', max(results), np.mean(results))
	print('local:', max(res_local), np.mean(res_local))

# spell run --pip-req requirements.txt 'python compare.py'

if __name__ == "__main__":
	# test_acc_realnet()

	compare()
	# compare(output_name='dict_label_fix__max_components')
	# compare(multipliers=np.array([1]), ps=np.array([.1]), instances=100, repetitions=100, numComm=2, commSize=250, output_name='tttest') # noises=, #np.linspace(.5,.5,1)

	# G, GT = get_ppg_fully_connected(2, 500, .05, .01, netx=True)
	# print(list(G.get_edgelist()))
	# solve_from_edgelist(list(G.get_edgelist()))

	# a = G.community_multilevel()
	# a = two_communities(G, a)
	# print(accuracies(G, [a], GT))

	# speed_test()

	# (g, GT), infos = get_ppg_max_components(2, 500, .02, .01)
	# ans = alg_with_ecg(g, alg1='hedonic', alg2='hedonic', size=8)
	# print(g.gam(ans, GT, method='dist'))

	# A = np.array(list(G.get_adjacency())) # igraph

	# labels, duration = spectral(G, A)
	# part = G.community_ecg()
	# AW = np.zeros((len(A),len(A)))
	# for (n0, n1), w in zip(G.get_edgelist(), part.W):
	# 	AW[n0][n1] = w
	# 	AW[n1][n0] = w
	# Wlabels, Wduration = spectral(G, AW)

	## Real Nets ############

	# compare_real_nets(repetitions=100) # 