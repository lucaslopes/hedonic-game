import numpy as np
from random import random, shuffle
from time import time
import seaborn as sns
import matplotlib.pyplot as plt
# from compare import get_real_nets

class Game:

	def __init__(self, edge_list=None, gt=None, alpha=None):
		self.load_from_edge_list(edge_list, gt)
		self.set_labels()
		self.set_alpha(alpha)
	
	def load_from_edge_list(self, edge_list=None, gt=None):
		edge_list = self.ppg() if edge_list is None else edge_list
		nodes = {}
		for edge in edge_list:
			n0, n1 = edge[0], edge[1] # verify redudant
			try:    nodes[n0].append(n1)
			except: nodes[n0] = [n1]
			try:    nodes[n1].append(n0)
			except: nodes[n1] = [n0]
		# verify if int list(nodes) sorted is contious (1,2,3...++,)
		for i in range(max(list(nodes))):
			if i not in nodes:
				print('ISOLATED NODE:', i)
				nodes[i] = []
		neighbors = [[] for _ in range(len(nodes))]
		for node, friends in nodes.items():
			neighbors[node] = [friend for friend in friends] 
		self.edge_list = list(edge_list) # [e for e in edge_list]
		self.labels = [0] * len(nodes)
		self.neighbors = neighbors
		self.GT = dict(gt) # [0] * int(len(self.labels)/2) + [1] * int(len(self.labels)/2) # todo: load GT

	def set_labels(self, labels=[], prob=.5):
		if len(labels) != len(self.labels):
			labels = [0 if prob > random() else 1 for _ in range(len(self.labels))]
			# if len(self.labels) != self.labels.count(0):
			# 	print('hedonic: labels has different size of number of vertices.')
		on_c0 = labels.count(0)
		with_me = [0] * len(self.labels)
		edges = [0, 0]
		for edge in self.edge_list:
			n0, n1 = edge[0], edge[1]
			c_n0 = labels[n0]
			c_n1 = labels[n1]
			if c_n0 == c_n1:
				with_me[n0] += 1
				with_me[n1] += 1
				edges[c_n0] += 1
		self.labels = [l for l in labels]
		self.with_me = with_me
		self.clusters = {
			'nodes' : [on_c0, len(self.labels)-on_c0],
			'edges' : edges }
		if labels != self.labels:
			print('ta errado tbm!!!!')
		# return labels

	def set_alpha(self, alpha=None):
		if alpha is None:
			alpha = self.edge_density()
		self.alpha = alpha

	def ppg(k=50, p=.5, q=.5):
		labels = [1] * k + [0] * k
		edge_list = []
		for i in range(k*2):
			for j in range(i+1, k*2):
				if labels[i] == labels[j] and p > random():
					edge_list.append((i,j))
				elif labels[i] != labels[j] and q > random():
					edge_list.append((i,j))
		return edge_list

	def edge_density(self, clusters=False):
		if clusters:
			n0, n1 = self.clusters['nodes'][0], self.clusters['nodes'][1]
			e0, e1 = self.clusters['edges'][0], self.clusters['edges'][1]
			max_edg_c0, max_edg_c1 = n0 * (n0-1) / 2, n1 * (n1-1) / 2
			return (e0 + e1) / (max_edg_c0 + max_edg_c1)
		else:
			edges = len(self.edge_list)
			n = len(self.labels)
			max_edges_possible = n * (n-1) / 2
			return edges / max_edges_possible

	def hedonic_value(self, have=0, not_have=0, alpha=None):
		alpha = self.alpha if alpha is None else alpha
		pros =     have * (1-alpha)
		cons = not_have *    alpha
		return pros - cons

	def potential(self, alpha=None, separated=False):
		nodes_c0, edges_c0 = self.clusters['nodes'][0], self.clusters['edges'][0]
		nodes_c1, edges_c1 = self.clusters['nodes'][1], self.clusters['edges'][1]
		pot_c0 = self.hedonic_value(have=edges_c0, not_have=(nodes_c0*(nodes_c0-1)/2)-edges_c0, alpha=alpha)
		pot_c1 = self.hedonic_value(have=edges_c1, not_have=(nodes_c1*(nodes_c1-1)/2)-edges_c1, alpha=alpha)
		if separated: return pot_c0, pot_c1
		else: return pot_c0 + pot_c1

	def get_node_atributes(self, node):
		friends_here  = self.with_me[node]
		friends_there = len(self.neighbors[node]) - friends_here
		strangers_here  = self.clusters['nodes'][self.labels[node]]-1 - friends_here
		strangers_there = self.clusters['nodes'][1-self.labels[node]] - friends_there
		return friends_here, friends_there, strangers_here, strangers_there

	def in_equilibrium_for(self, alpha=None, inspect=False):
		alpha = self.alpha if alpha is None else alpha
		dissatisfied = 0
		for node in range(len(self.labels)):
			if not self.satisfied(node, alpha):
				if inspect:
					dissatisfied += 1
				else:
					return False
		if inspect:
			return 1 - dissatisfied / len(self.labels)
		else:
			return True

	def satisfied(self, node, alpha=None, profit=False):
		friends_here, friends_there, strangers_here, strangers_there = self.get_node_atributes(node)
		value_here  = self.hedonic_value(friends_here, strangers_here, alpha)
		value_there = self.hedonic_value(friends_there, strangers_there, alpha)
		return value_there - value_here if profit else value_here >= value_there
	
	def move(self, node):
		there, here, _, _ = self.get_node_atributes(node)
		self.labels[node] = 1 - self.labels[node]
		self.with_me[node] = here
		for friend in self.neighbors[node]:
			if self.labels[node] == self.labels[friend]:
				self.with_me[friend] += 1
			else:
				self.with_me[friend] -= 1
		l = self.labels[node]
		self.clusters['nodes'][l]   += 1
		self.clusters['nodes'][1-l] -= 1
		self.clusters['edges'][l]   += here
		self.clusters['edges'][1-l] -= there
		# print(node, self.get_node_atributes(node), l)

	def play(self, alpha=None, naive=False):
		# now = time()
		# init = [l if self.labels[0] == 0 else 1 - l for l in self.labels]
		moved, nodes_list = True, list(range(len(self.labels)))
		nodes_moved = []
		while moved is True:
			moved = False
			shuffle(nodes_list)
			for node in nodes_list:
				want_move = False
				if naive:
					want_move = not self.satisfied(node, alpha)
				else: # almost robust (95%)
					pft_A0 = self.satisfied(node, alpha=0, profit=True)
					pft_A1 = self.satisfied(node, alpha=1, profit=True)
					want_move = ((pft_A0  > 0 and pft_A1 >= 0) or (pft_A0 <= 0 and pft_A1  > 0))
				if want_move:
					self.move(node)
					moved = True
					nodes_moved.append(node)
					# print('moved', node)
		# final = [l if self.labels[0] == 0 else 1 - l for l in self.labels]
		# return init, nodes_moved, final
		# print('python naive:', time()-now)

	def hedonic_weighted(self, W): # , alpha=0
		moved, nodes_list = True, list(range(len(self.labels)))
		while moved is True:
			print('newloop')
			moved = False
			shuffle(nodes_list)
			for node in nodes_list:
				if not self.satisfied_weighted(node, W[node]):
					self.move(node)
					moved = True

	def satisfied_weighted(self, node, weights, alpha=1, return_profit=False):
		here, there = [0,0], [0,0] # [pros, cons]
		for other, w in enumerate(weights):
			if self.labels[node] == self.labels[other]:
				if other in self.neighbors[node]:
					here[0] += w
				else:
					here[1] += alpha
			else:
				if other in self.neighbors[node]:
					there[0] += w
				else:
					there[1] += alpha
		value_here, value_there = (here[0]-here[1]), (there[0]-there[1])
		return value_there - value_here if return_profit else value_here >= value_there

	# def satisfied_weighted(self, node, weights, alpha=1, return_profit=False):
	# 	here, there = [0,0], [0,0] # [friends, strangers]
	# 	my_cluster  = self.labels[node]
	# 	strangers   = list(range(len(weights))) # list(np.arange(0,len(weights),1))
	# 	for friend in self.neighbors[node]:
	# 		strangers[friend] = None
	# 	strangers = [node for node in strangers if node is not None]
	# 	for friend in self.neighbors[node]: # friends
	# 		if my_cluster == self.labels[friend]:
	# 			here[0]  += weights[friend]
	# 		else:
	# 			there[0] += weights[friend]
	# 	for stranger in strangers: # not friends
	# 		if my_cluster == self.labels[stranger]:
	# 			here[1]  += alpha
	# 		else:
	# 			there[1] += alpha
	# 	value_here, value_there = (here[0]-here[1]), (there[0]-there[1])
	# 	return value_there - value_here if return_profit else value_here >= value_there

	def accuracy(self, x=None,y=None): # WARNING: x and y must be only 0 or 1
		if x is None:
			x = self.labels
			gt = [0 for i in range(len(self.GT))] # max(list(self.GT))+1
			for node, cluster in self.GT.items():
				gt[node] = cluster
			y = gt
		x, y = np.array(x), np.array(y)
		if len(x) != len(y):
			print(len(x), len(y))
			raise ValueError('x and y must be arrays of the same size')
		matches1 = sum(x == y)
		matches2 = sum(x == 1-y)
		score = max([matches1, matches2]) / len(x)
		return score
	
	def find_route(self, from_state, to_state):
		x, y = np.array(from_state), np.array(to_state)
		if len(x) != len(y):
			print(len(x), len(y))
			raise ValueError('x and y must be arrays of the same size')
		route1 = [i for i in range(len(x)) if x[i] != y[i]]
		route2 = [i for i in range(len(x)) if x[i] != 1-y[i]]
		return route1 if len(route1) < len(route2) else route2

	def all_possible_states(self):
		possible_states = range(2 ** (len(self.labels)-1))
		return [('0'*len(self.labels)+f'{state:b}')[-len(self.labels):] for state in possible_states]
	
	def potential_robustness_accuracy(self, possible_states=None):

		# histogramas cruzados:
		# https://seaborn.pydata.org/examples/hexbin_marginals.html
		#
		# Com acurácia
		# - prop verts X acuracia
		# - prop intra cluster X acuracia
		# - prop inter cluster X acuracia
		# - prop potential X acuracia
		# - potencial X acurácia
		# - robustez X acurácia
		# - density_gain X acuracia
		#
		# Entre si
		# - prop verts X prop intra cluster
		# - prop potential X prop intra cluster
		# - prop inter cluster X density_gain
		# - potencial (histogram: b and m (y=mx+b) where x = alpha)

		# propoções: histogramas -- cor = 5 níveis de acurácia (.5, .6, .7, .8, .9)
		# https://seaborn.pydata.org/examples/histogram_stacked.html
		
		#   - satisfeitos (robustez) -- é equilíbrio para quais alphas?
		#       - quais alphas estavam em equilibrio: histog (x=valor de alpha) [0,0,3,...,7,1,2]
		#       - quantos alphas estavam em equilíbrio: histog (x=qntd de alphas) [0, 1, 2, ...]

		columns = { # todo: accuracy is 'clusters' (.7<.9<1)
				'vertices' : [],
				'intra'    : [],
				'inter'    : [],
				'pot_prop': [],
				'density_gain' : [],
				'alphas'   : [],
				'potential': [],
				'accuracy' : [] }
		possible_states = self.all_possible_states() if possible_states is None else possible_states
		for state in possible_states:
			state = [int(label) for label in state]
			route = self.find_route(self.labels, state)
			for node in route: self.move(node)
			pots = [self.potential(alpha) for alpha in [0,1]]
			pot_c0, pot_c1 = self.potential(alpha=1, separated=True)
			pot_c0, pot_c1 = abs(pot_c0), abs(pot_c1)
			pot_prop = min(pot_c0, pot_c1)/(pot_c0+pot_c1)
			satisfied = [self.in_equilibrium_for(alpha, inspect=True) for alpha in np.linspace(0,1,11)]
			zeros = state.count(0)
			columns['vertices'].append(min(zeros,len(state)-zeros)/len(state))
			columns['intra'].append(min(self.clusters['edges'][0],self.clusters['edges'][1])/(self.clusters['edges'][0]+self.clusters['edges'][1]))
			columns['inter'].append((len(self.edge_list)-(self.clusters['edges'][0]+self.clusters['edges'][1])) / len(self.edge_list))
			columns['pot_prop'].append(pot_prop)
			columns['density_gain'].append(self.edge_density(clusters=True) / self.edge_density())
			columns['alphas'].append(satisfied)
			columns['potential'].append([pots[0],pots[1]-pots[0]]) # b and m (y=mx+b) where x = alpha
			columns['accuracy'].append(self.accuracy())
		return columns

	def calc_robustness(self):
		return np.mean([self.in_equilibrium_for(alpha, inspect=True) for alpha in np.linspace(0,1,11)])

	def find_eq_in_O_Edges(self):
		# percorre a lista de edges e vai decidindo se ela será intra ou inter cluster
		# se a partir de uma lista de edges, saber-se se cada uma delas será 'intra' OU 'inter' conectada, é possível montar tal particionamento facilmente?
		# # atentar-se aos casos impossíveis (A junto com B, A separado de C, C junto com B)
		# não preocupar-se com quantas passadas terá em O(Edges), mas buscar que qualquer operação seja O(Edges)
		# ao adicionar uma aresta, há 4 possibilidades: (AB,_) (_,AB) (A,B) (B,A) -> escolher a que possui maior potencial
		for edge in self.edge_list:
			break

	def save_game(self):
		pass # dict in .txt

	def load_game(self):
		pass # txt to load attributes

if __name__ == "__main__":
	game = Game(edge_list=Game.ppg(6, .6, .4))
	game.play()

	# now = time()
	# for p in np.linspace(.5, 1, 11):
	# 	for r in range(10):
	# 		game = Game(edge_list=Game.ppg(50, p, 1-p))
	# 		game.play()
	# print(time()-now)
	
	# real_nets = get_real_nets() # nao esquecer de voltar com import hedonic em compare e partition_graph
	# for net, (g, gt) in real_nets.items():
	# 	game = Game(g.get_edgelist(), gt)
	# 	best = 0
	# 	for i in range(100):
	# 		game.set_labels()
	# 		game.play(naive=False)
	# 		acc = game.accuracy()
	# 		if acc > best:
	# 			best = acc
	# 	print(net, best)
	# eq = game.in_equilibrium_for(inspect=True)
	# if not eq: # Walrus Operator :=
	# 	print(f'game is not in equilibrium for alpha=edge density ({eq}')
	# if only_membership:
	# 	return game.labels
	# else:
	# 	return from_label_to_dict(game.labels), duration, game.calc_robustness()

	# pot, robs, accs = game.potential_robustness_accuracy()
	# robs = [x for _, x in sorted(zip(accs, robs))] # sort by first elem of tupple: accs or pot
	# pot  = [x for _, x in sorted(zip(accs, pot))]
	# # pot.sort()
	
	# # todo: distance (100%) between states, in the order of accuracy ascending
	# print('min and max of robustness:', min(robs), max(robs))
	# sns.barplot(x=list(range(len(pot))), y=pot, hue=robs,palette='Greens', dodge=False) # , legend=False 
	# # plt.bar(x=list(range(len(pot))), height=pot, color=robs) # palette='coolwarm'
	# plt.show()
	# # print([1])

	# tentativa: quantos % foram equilibrio para tal alpha
	# realidade: media de robustez de tal alpha
	# col = game.potential_robustness_accuracy()
	# als = col['alphas']
	# hist = [0] * 11
	# for lst in als:
	#     for i, a in enumerate(lst):
	#         hist[i] += a
	# hist = [h/len(als) for h in hist]
	# print(hist)
	# plt.plot(hist)
	# plt.show()

	# for label in game.all_possible_states():
	#     game.set_labels(labels=[int(l) for l in label])
	#     pots = []
	#     for alpha in [0,1]: # np.linspace(0,1,11)
	#         pots.append(game.potential(alpha))
	#     b, m = pots[0], pots[1]-pots[0]
	#     x = np.linspace(0,1,11)
	#     y = m * x
	#     sns.lineplot(x=x, y=m*x+b)#, hue=game.accuracy(),palette='Greens')
	# plt.show()

	# columns = game.potential_robustness_accuracy()
   
	# Com acurácia:
	# - prop verts X acuracia
	# sns.jointplot(x=columns['vertices'], y=columns['accuracy'], color='#4CB391', kind='kde') # kde hex

	# # - prop intra cluster X acuracia
	# sns.jointplot(x=columns['intra'], y=columns['accuracy'], color='#4CB391', kind='kde') # kde hex
	# print(list(set(columns['accuracy'])))

	# # - prop inter cluster X acuracia
	# sns.jointplot(x=columns['inter'], y=columns['accuracy'], color='#4CB391', kind='kde') # kde hex

	# # - prop potential X acuracia
	# sns.jointplot(x=columns['pot_prop'], y=columns['accuracy'], color='#4CB391', kind='kde') # kde hex

	# # - potencial X acurácia
	# # sns.jointplot(x=columns['potential'], y=columns['accuracy'], color='#4CB391', kind='kde') # kde hex

	# # - robustez X acurácia
	# robustez = [np.mean(robs) for robs in columns['alphas']]
	# sns.jointplot(x=columns['vertices'], y=columns['accuracy'], color='#4CB391', kind='kde') # kde hex

	# # - density_gain X acuracia
	# sns.jointplot(x=columns['density_gain'], y=columns['accuracy'], color='#4CB391', kind='kde') # kde hex

	# # Entre si:
	# # - prop verts X prop intra cluster
	# sns.jointplot(x=columns['vertices'], y=columns['intra'], color='#4CB391', kind='kde') # kde hex

	# # - prop potential X prop intra cluster
	# sns.jointplot(x=columns['pot_prop'], y=columns['intra'], color='#4CB391', kind='kde') # kde hex

	# # - prop inter cluster X density_gain
	# sns.jointplot(x=columns['inter'], y=columns['density_gain'], color='#4CB391', kind='kde') # kde hex

	# # - potencial (histogram: b and m (y=mx+b) where x = alpha)
	# bs = [pot[0] for pot in columns['potential']]
	# ms = [pot[1] for pot in columns['potential']]
	# sns.jointplot(x=bs, y=ms, color='#4CB391', kind='kde') # kde hex

	# plt.show()