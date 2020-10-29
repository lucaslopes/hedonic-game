import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns; sns.set(style="whitegrid")
from PIL import Image
from tqdm import tqdm

##############################################################################
## Load Data ############################################################################

# old_rows = {'noise':[],'p_in':[],'mult':[],'instance':[],'repetition':[],'algorithm':[],'accuracy':[],'robustness':[],'seconds':[],'method':[]}
	# print('entrou')
	# print(df.columns)
	# print(df2.columns)

	# ['noise', 'mult', 'p_in', 'p_out', 'instance', 'repetition',
    #    'hedonic_accuracy', 'hedonic_infos', 'hedonic_robustness',
    #    'hedonic_time', 'onepass_accuracy', 'onepass_infos',
    #    'onepass_robustness', 'onepass_time', 'spectral_accuracy',
    #    'spectral_infos', 'spectral_robustness', 'spectral_time', 'algorithm',
    #    'accuracy', 'robustness', 'method', 'seconds'],
	# ['p_in', 'mult', 'instance', 'repetition', 'algorithm', 'accuracy',
    #    'robustness', 'method', 'seconds', 'noise'],

# print(df3.columns)
	# print(df2.columns)
	# print(df3.head(10))

	# df3 = df.melt('mult', var_name='algorithm', value_name='noise')
	# for index, row in tqdm(df.iterrows()):
	# 	print(index, len(df), index/len(df)*100)
	# 	for alg in ['hedonic', 'onepass', 'spectral']:
	# 		pass
			# old_rows['noise'].append(row['noise'])
			# old_rows['p_in'].append(row['p_in'])
			# old_rows['mult'].append(row['mult'])
			# old_rows['instance'].append(row['instance'])
			# old_rows['repetition'].append(row['repetition'])
			# old_rows['algorithm'].append(alg)
			# old_rows['accuracy'].append(row[f'{alg}_accuracy'])
			# old_rows['robustness'].append(row[f'{alg}_robustness'])
			# old_rows['seconds'].append(row[f'{alg}_time'])
			# old_rows['method'].append('dist')
			# df2 = df2.append({
			# 	'noise': row['noise'],
			# 	'p_in': row['p_in'],
			# 	'mult': row['mult'],
			# 	'instance': row['instance'],
			# 	'repetition': row['repetition'],
			# 	'algorithm': alg,
			# 	'accuracy': row[f'{alg}_accuracy'],
			# 	'robustness': row[f'{alg}_robustness'],
			# 	'seconds': row[f'{alg}_time'],
			# 	'method': 'dist'
			# }, ignore_index=True)

# clustering / improvement / robust / onepass == local improve
algorithms = { 'spectral':'spectral', 'local improve':'local improve', 'hedonic':'hedonic', 'ml':'louvain', 'ecg':'ecg' }
alg_colors = { 'spectral':'#C44E53', 'local improve':'#DD8452', 'hedonic':'#55A869', 'louvain':'#4C72B0', 'ecg':'#8172B3' }
file_name_prefix = ''

def adaptd_df_to_new_exp(df, prefix='outputs/noises/'):
	df1 = pd.read_csv(f'{prefix}ps=10_mults=11_inst=10_reps=10_nComm=2_commSize=500.csv')
	df1 = df1.loc[df1['method'] == 'dist']
	df1 = df1.loc[df1['algorithm'] != 'hedonic']
	df1 = df1.loc[df1['algorithm'] != 'spectral']
	df1.drop(columns=['p_in','instance','repetition','robustness','method'], inplace=True)

	df2, temp = pd.DataFrame(), df1.copy()
	for noise in df['noise'].unique():
		temp['noise'] = [noise] * len(temp['mult'])
		df2 = pd.concat([df2, temp])
	
	fix_cols = ['noise', 'p_in', 'mult', 'instance', 'repetition']
	dfs_melted = {}
	for col in ['accuracy', 'time', 'robustness']:
		dfs_melted[col] = df.melt(id_vars=fix_cols, value_vars=[f'hedonic_{col}', f'onepass_{col}', f'spectral_{col}'], var_name='algorithm', value_name=col)
	df3 = dfs_melted['accuracy'].loc[:, fix_cols+['algorithm']]
	df3['method'] = ['dist'] * len(df3)
	df3['algorithm'] = [alg.split('_')[0] for alg in df3['algorithm']]
	df3['accuracy'] = dfs_melted['accuracy']['accuracy']
	df3['robustness'] = dfs_melted['robustness']['robustness']
	df3['seconds'] = dfs_melted['time']['time']

	df2 = pd.concat([df2, df3])
	df2['algorithm'] = [algorithms[alg] for alg in df2['algorithm'].values]
	return df2

# def load_df():
# 	prefix = 'outputs/noises/'
# 	df = pd.concat([pd.read_csv(f'{prefix}{csv}') for csv in os.listdir(prefix) if csv[-3:] == 'csv' and csv[:5] == 'noise'])
# 	# df['p_in-p_out'] = df['p_in'] - df['p_out']
# 	return adaptd_df_to_new_exp(df, prefix)

def load_df():
	df = pd.read_csv('test_noise__ps=5_mults=11_inst=5_reps=5_noises=13_nComm=2_commSize=50.csv')
	df = df.loc[df['method'] == 'dist']
	df.drop(columns=['p_in','instance','repetition','robustness','method'], inplace=True)
	df['algorithm'] = [alg.split('_')[0] for alg in df['algorithm']]
	df['algorithm'] = [algorithms[alg] for alg in df['algorithm'].values]
	return df

##############################################################################

def load_dfs_noise(df):
	dfs_noise = dict(tuple(df.groupby('noise')))
	# went, total = 0, len(dfs_noise) * 11 * 25 * 25 * 3
	for noise, data in dfs_noise.items():
		df_noise = data.loc[:, ['mult', 'algorithm', 'accuracy']]
		# df_noise = pd.DataFrame(columns=['mult', 'algorithm', 'accuracy']) # +[f'{alg}_accuracy' for alg in algorithms]
		# for mult in data['mult'].unique():
		# 	# print('noise =', noise, '-- mult =', mult)
		# 	for inst in data['instance'].unique():
		# 		for alg in data['algorithm'].unique(): # [f'{alg}_accuracy' for alg in algorithms]
		# 			went += 1
		# 			print(went, total, went/total*100)
		# 			df_noise = df_noise.append({
		# 				'mult'  : mult,
		# 				'algorithm': alg,
		# 				'accuracy': np.mean(data.loc[(data['mult'] == mult) & (data['instance'] == inst) & (data['algorithm'] == alg), 'accuracy'].values),
		# 				# algs[0] : np.mean(data.loc[(data['mult'] == mult) & (data['instance'] == inst), [algs[0]]].values),
		# 				# algs[1] : np.mean(data.loc[(data['mult'] == mult) & (data['instance'] == inst), [algs[1]]].values),
		# 				# algs[2] : np.mean(data.loc[(data['mult'] == mult) & (data['instance'] == inst), [algs[2]]].values),
		# 			}, ignore_index=True)
		# df_noise = data.loc[:, ['mult']+[f'{alg}_accuracy' for alg in algorithms]]
		# df_noise = df_noise.melt('mult', var_name='algorithm', value_name='accuracy')
		# df_noise['algorithm'] = [alg.split('_')[0] for alg in df_noise['algorithm'].values]
		# df_noise['algorithm'] = [algorithms[alg] for alg in df_noise['algorithm'].values]
		df_noise.rename(columns = {'mult':'q/p'}, inplace = True)
		dfs_noise[noise] = df_noise
	return dfs_noise

##############################################################################

def load_speed(df):
	speed = df.loc[:, ['noise', 'algorithm', 'seconds']] # +[f'{alg}_time' for alg in algorithms]
	# speed = speed.melt('noise', var_name='algorithm', value_name='seconds')
	# speed['algorithm'] = [alg.split('_')[0] for alg in speed['algorithm'].values]
	speed['algorithm'] = [alg.replace(' ','\n') for alg in speed['algorithm'].values]
	for alg in speed['algorithm'].unique():
		alg_df = speed[speed['algorithm'] == alg]
	return speed#.loc[speed['algorithm'] != 'ecg']

##############################################################################

def calc_r(df_accs, noise=.5):
	mults = df_accs['q/p'].unique()
	mults.sort()
	accuracies = []
	for mult in mults:
		accuracies.append(np.mean(df_accs.loc[(df_accs['q/p'] == mult), 'accuracy'].values))
	for i, acc in enumerate(accuracies):
		if acc < 1-noise:
			if i == 0:
				return 0
			else:
				ab = accuracies[i-1] - acc
				bc = mults[i] - mults[i-1]
				ef = 1 - noise - acc
				fc = (bc * ef) / ab
				return mults[i] - fc
	return 1

def load_df_A(dfs_noise):
	# noises = [round(x, 3) for x in [0, 0.025] + list(np.linspace(0, .5, 11))[1:-1] + [0.475, .5]]
	df_A = pd.DataFrame(columns=['noise', 'algorithm', 'r'])
	for noi in list(dfs_noise):
		df_noi = dfs_noise[noi]
		for alg in df_noi['algorithm'].unique():
			df_A = df_A.append({
				'noise'  : noi,
				'algorithm': alg,
				'r': calc_r(df_noi[df_noi['algorithm'] == alg].loc[:,['q/p','accuracy']],noi)
			}, ignore_index=True)
	df_A['r'] = df_A['r'].apply(pd.to_numeric)
	df_A['noise'] = df_A['noise'].apply(pd.to_numeric)
	df_A.rename(columns={'r':'r = maximum tolerated q/p'}, inplace=True)
	# denom = 1468.49
	# df_A['spectral'] = [0,(denom-1028.25)/denom,(denom-900.48)/denom,(denom-710.51)/denom,(denom-600.57)/denom,(denom-536.37)/denom,(denom-474.91)/denom,(denom-414.02)/denom,(denom-370.85)/denom,(denom-325.62)/denom,(denom-245.38)/denom,(denom-170.39)/denom,1]
	# df_A['onepass']  = [0,(denom-824.52)/denom,(denom-700.02)/denom,(denom-591.54)/denom,(denom-543.63)/denom,(denom-512.04)/denom,(denom-487.50)/denom,(denom-470.42)/denom,(denom-462.15)/denom,(denom-453.79)/denom,(denom-448.22)/denom,(denom-389.49)/denom,1]
	# df_A['hedonic']  = [0,(denom-857.10)/denom,(denom-715.39)/denom,(denom-575.51)/denom,(denom-502.74)/denom,(denom-438.50)/denom,(denom-397.13)/denom,(denom-362.14)/denom,(denom-338.83)/denom,(denom-302.52)/denom,(denom-303.45)/denom,(denom-217.39)/denom,1]
	# df_A = df_A.melt('noise', var_name='algorithm', value_name='r = maximum tolerated q/p') # r is the maximum value of q/p for which we observe gains in the corresponding method when contrasted against simple replication of the input as the output, i.e., maximum value of q/p for which the accuracy of the method is greater than 1-noise
	# df_A['algorithm'] = [algorithms[alg] for alg in df_A['algorithm'].values]
	return df_A

##############################################################################

def load_realnets_df():
	prefix = 'outputs/real_nets'
	csv_name = '4_networks_10_repts'
	realnets_df = pd.read_csv(f'{prefix}/{csv_name}.csv')
	realnets_df.drop(columns=['infos','robustness'], inplace=True)
	realnets_df.rename(columns={'time':'seconds'}, inplace=True)

	new_realnets_df = pd.read_csv(f'{prefix}/real_nets__networks=4_reps=10.csv')
	new_realnets_df = new_realnets_df.loc[new_realnets_df['method'] == 'dist']
	new_realnets_df.drop(columns=['repetition','robustness','method'], inplace=True)
	new_realnets_df = new_realnets_df.loc[new_realnets_df['algorithm'] != 'hedonic']
	new_realnets_df = new_realnets_df.loc[new_realnets_df['algorithm'] != 'spectral']

	realnets_df = pd.concat([realnets_df, new_realnets_df])
	# print(realnets_df['algorithm'].unique())
	# algs = []
	# for alg in realnets_df['algorithm']:
	# 	if alg != [algorithms['spectral'], algorithms['ml'], algorithms['ecg']]:
	# 		alg = alg.split('_')
	# 		alg[0] = algorithms[alg[0]]
	# 		alg = f'{alg[0]} (noi={alg[1][2:]})'
	# 	algs.append(alg)
	# realnets_df['algorithm'] = algs
	
	# print(realnets_df.loc[(realnets_df['algorithm'] == 'ml') & (realnets_df['network'] == 'pol_blogs'), 'accuracy'].max())
	# print(realnets_df.loc[(realnets_df['algorithm'] == 'ecg') & (realnets_df['network'] == 'pol_blogs'), 'accuracy'].max())

	return realnets_df

##############################################################################

def load_datas():
	df = load_df()
	dfs_noise = load_dfs_noise(df)
	speed = load_speed(df)
	df_A = load_df_A(dfs_noise)
	realnets_df = load_realnets_df()
	return df, dfs_noise, speed, df_A, realnets_df

##############################################################################
## Functions ############################################################################

def create_noises_plot_dir(outname='no_name'):
	outdir = './outputs/noises/plots'
	if not os.path.exists(outdir):
		os.mkdir(outdir)
	fullname = os.path.join(outdir, outname) 
	return fullname

def get_ax(font_size=45, n_cols=4, width=50):
	plt.clf()
	matplotlib.rc('xtick', labelsize=font_size*.9)
	matplotlib.rc('ytick', labelsize=font_size*.9)
	return plt.subplots(nrows=1, ncols=n_cols, figsize=(width,10)) # num=None , dpi=250 , figsize=(7.8*4, 6.3)

def save_plot(ax, title='no title', x_label='', y_label='', f_name='',
		font_size=45, n_col=1, labels_handles=None, has_legend=True,
		font_scale=.8, leg_loc=0, title_loc='center', title_y=-.275):

	handles, labels = ax.get_legend_handles_labels()
	# handles = []
	# for alg in labels:
	# 	handles.append(mpatches.Patch(color=alg_colors[alg], label=alg))
	# handles = [mpatches.Patch(color=c, label=key) for key, c in alg_colors.items()]
	# labels  = list(alg_colors)
	if len(labels) == 5:
		order = [3,1,2,0,4]
		handles = [handles[idx] for idx in order]
		labels  = [labels[idx] for idx in order]
	if labels_handles is not None:
		handles = [mpatches.Patch(color=c, label=key) for key, c in labels_handles.items()]
		labels  = list(labels_handles)
	if has_legend:
		for h in handles:
			h.set_linewidth(5)
		leg = ax.legend(fontsize=font_size*font_scale, ncol=n_col, handles=handles, labels=labels, loc=leg_loc)
	ax.set_title(title, fontsize=font_size, loc=title_loc, y=title_y)
	ax.set_xlabel(x_label, fontsize=font_size)
	ax.set_ylabel(y_label, fontsize=font_size)
	ax.grid(True)

##############################################################################
## Generate Plots ############################################################################

def generate_noises_plot():
	for noise_graph, data in dfs_noise.items():
		fig, ax = get_ax(n_cols=1, width=12.5)
		ax.set_ylim(.499,1.01)
		g = sns.lineplot(x="q/p", y="accuracy", hue="algorithm", data=data, marker='o', linewidth=4.5, ax=ax, ci=95) # , style="event"
		ax.axhline(1-noise_graph, c='red', alpha=.75, ls='--', lw=2.5)
		# ax.annotate('$1$-noise', (.85, 1.005-noise_graph), fontsize=40)
		noise_name = str(round(noise_graph,3))+'00000000'
		save_plot(ax, title=f'noise={noise_name[:5]}', x_label='q/p', y_label='accuracy', f_name=f'{file_name_prefix}_n{round(noise_graph,3)}', leg_loc='lower left', title_loc='right', title_y=.9)
		plt.tight_layout()
		fig.savefig(create_noises_plot_dir(f'noise_{noise_graph}.png'))
		# break

##############################################################################

def generate_gif():
	frames = []
	folder = './outputs/noises/plots/'
	imgs = [png for png in os.listdir(folder) if png[-3:] == 'png'] # get_all_files()#glob.glob('.outputs/noises/*.png')
	imgs = [(img, eval(img[6:-4])) for img in imgs]
	imgs.sort(key= lambda x: x[1])
	imgs = [img[0] for img in imgs]
	for i in imgs:
		new_frame = Image.open(folder+i)
		frames.append(new_frame)
	frames[0].save('noises.gif', format='GIF',
				append_images=frames[1:],
				save_all=True,
				duration=500, loop=0)

##############################################################################

def generate_fig_1():
	fig, axes = get_ax()

	noise_graph = .35 # Choose one noise to be in Fig 1 (a)
	axes[0].set_ylim(.499,1.01)
	g = sns.lineplot(x="q/p", y="accuracy", hue="algorithm", data=dfs_noise[noise_graph], marker='o', linewidth=4.5, ci=95, hue_order=['louvain','local improve','hedonic','spectral','ecg'], ax=axes[0]) # , style="event"
	axes[0].axhline(1-noise_graph, c='red', alpha=.75, ls='--', lw=2.5)
	axes[0].annotate('$1$-noise', (.85, 1.005-noise_graph), fontsize=40)
	plt.text(0.6, 0.65, '', bbox=dict(boxstyle="round", fc="white", ec="red", pad=0.2))
	noise_name = str(round(noise_graph,3))+'00000000'
	save_plot(axes[0], title=f'(a) accuracy vs q/p (noise={noise_name[:5]})', x_label='q/p', y_label='accuracy', f_name=f'{file_name_prefix}_n{round(noise_graph,3)}')
	print('plot 1')

	##############################################################################

	g = sns.lineplot(x="noise", y="r = maximum tolerated q/p", hue="algorithm", data=df_A, marker='o', linewidth=4.5, hue_order=['louvain','local improve','hedonic','spectral','ecg'], ax=axes[1])
	save_plot(axes[1], title='(b) max q/p such that accuracy$\geq1$-noise', x_label='noise', y_label='r = maximum tolerated q/p', f_name='cross')
	print('plot 2')

	##############################################################################

	# ['louvain','local improve','hedonic','spectral','ecg']
	# order=['spectral','local\nimprov','hedonic','louvain','ecg']
	# ['louvain' 'ecg' 'hedonic' 'local\nimprov' 'spectral']
	# print('ki', speed['algorithm'].unique())
	g = sns.violinplot(x="algorithm", y="seconds", data=speed, fontsize=27.5, hue_order=['spectral','local\nimprov','hedonic','louvain','ecg'], ax=axes[2])
	# g = sns.boxplot(x="algorithm", y="seconds", data=speed, ax=axes[2]) # , fontsize=27.5
	# g = sns.swarmplot(x="algorithm", y="seconds", data=speed, color=".25", ax=axes[2])
	axes[2].set(yscale="log") # xscale="log", 
	save_plot(axes[2], title='(c) time efficiency', x_label='', y_label='seconds', f_name='time', has_legend=False) # algorithm
	print('plot 3')

	##############################################################################

	alg_repetition = int((len(realnets_df['algorithm'].unique())-3)/2) # #5975A4 sns.color_palette("Blues", n_colors=1)
	cor = [(alg_colors['louvain']),(alg_colors['ecg']),(alg_colors['spectral'])]+sns.color_palette("Oranges", n_colors=alg_repetition)+sns.color_palette("Greens", n_colors=alg_repetition) # #5975A4 #CC8964 #5F9E6E
	realnets_order = ['ml','ecg','spectral','onepass_n0.0','onepass_n0.05','onepass_n0.1','onepass_n0.15','onepass_n0.2','onepass_n0.25','onepass_n0.3','onepass_n0.35','onepass_n0.4','onepass_n0.45','onepass_n0.5','hedonic_n0.0','hedonic_n0.05','hedonic_n0.1','hedonic_n0.15','hedonic_n0.2','hedonic_n0.25','hedonic_n0.3','hedonic_n0.35','hedonic_n0.4','hedonic_n0.45','hedonic_n0.5']
	g = sns.barplot(x="network", y="accuracy", hue="algorithm", data=realnets_df, palette=cor, hue_order=realnets_order, ax=axes[3]) # order=realnets_order
	max_acc = [] # [0] * len(realnets_order) * 4
	for net in ['karate', 'dolphins', 'pol_blogs', 'pol_books']:
		nets_df = realnets_df[realnets_df['network'] == net]
		for alg in realnets_order: # nets_df['algorithm'].unique():
			nets_alg_df = nets_df[nets_df['algorithm'] == alg] # .max()
			max_acc.append(nets_alg_df['accuracy'].max())
			# max_acc[realnets_order.index(alg)] = nets_alg_df['accuracy'].max()
	x_pos = [p.get_x() for p in axes[3].patches]
	x_pos.sort()
	colors = [alg_colors['louvain'],alg_colors['ecg'],alg_colors['spectral']]+[alg_colors['local improve'] for _ in range(alg_repetition)]+[alg_colors['hedonic'] for _ in range(alg_repetition)]
	for pos, maxx, c in zip(x_pos, max_acc, colors*4):
		axes[3].annotate('_', (pos, maxx), c=c, weight=1000) # p.get_height() * 1.005)
	axes[3].set_ylim(0.499,1.01)
	# labls = {'spectral clustering':'#5975A4', 'local improvement\n(0$\leq$noise$\leq$0.5)':'#CC8964', 'hedonic robust\n(0$\leq$noise$\leq$0.5)':'#5F9E6E'}
	labls = {'spectral':alg_colors['spectral'], 'louvain':alg_colors['louvain'], 'ecg':alg_colors['ecg'], 'local improve':alg_colors['local improve'], 'hedonic':alg_colors['hedonic']} # \n(0$\leq$noise$\leq$0.5)
	save_plot(axes[3], title='(d) real networks', x_label='network', y_label='accuracy', f_name='real_nets_bar_plot', n_col=2, labels_handles=labls, font_scale=.7)
	print('plot 4')

	##############################################################################

	plt.tight_layout()
	fig.savefig('fig1.png')
	print('save fig')

##############################################################################
## Plot Comparison ############################################################################

def filter_gt_balanced(df):
	print(df['max_comp'].max(), df['max_comp'].min())
	# print(df.columns)
	adasdadas()
	# df = df[df['gt_balance'] ]

def generate_plot_comparion(filename=''):
	df = pd.read_csv(filename)
	# df = filter_gt_balanced(df)
	plot_hists(df)
	asdas()

	# print(len(df))
	# df['q'] = df['p_in'] * df['mult']
	# df = df[(df['p_in']+df['q']) / 2 > np.log2(500*2)] # ( p+q ) / 2 > log n
	# print(len(df))

	font_size = 45
	plt.clf()
	matplotlib.rc('xtick', labelsize=font_size*.8)
	matplotlib.rc('ytick', labelsize=font_size*.8)
	fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(60,30)) # num=None , dpi=250 , figsize=(7.8*4, 6.3)
	i, j = 0, 0
	methods = df['method'].unique()
	for method in methods:
		df_method = df[df['method']==method]
		ax = axes[i][j]
		sns.lineplot(x='mult', y='accuracy', hue='algorithm', data=df_method, marker='o', linewidth=4, ax=ax)
		ax.set_title(method, fontsize=font_size)
		ax.set_xlabel('q/p', fontsize=font_size*.9)
		ax.set_ylabel('accuracy', fontsize=font_size*.9)
		ax.grid(True)
		handles, labels = ax.get_legend_handles_labels()
		# handles, labels = handles[1:], labels[1:]
		for h in handles:
			h.set_linewidth(5)
		ax.legend(fontsize=font_size*.7, handles=handles, labels=labels, loc='upper right')
		j += 1
		if j == 4:
			j  = 0
			i += 1
	ax = axes[i][j]
	ax.set_title('speed', fontsize=font_size)
	ax.set_xlabel('q/p', fontsize=font_size*.9) # algorithms
	ax.set_ylabel('seconds', fontsize=font_size*.9)
	# ax. # log scale
	# ax.set(ylim=(0, .15))
	# sns.violinplot(x='algorithm', y='seconds', data=df, fontsize=27.5, ax=ax) # [df['method']=='rand']
	
	sns.lineplot(x='mult', y='seconds', hue='algorithm', data=df[df['method']=='dist'], marker='o', linewidth=4, ax=ax)
	ax.grid(True)
	handles, labels = ax.get_legend_handles_labels()
	for h in handles:
		h.set_linewidth(5)
	ax.legend(fontsize=font_size*.7, handles=handles, labels=labels, loc='upper left')

	plt.savefig(f'{filename[:-4]}_junto_filter.png')

##########################
## Hedonic Robust vs Naive

def plot_robust_vs_naive():
	df = pd.read_csv('outputs/comparisons/rob_vs_naive__ps=10_mults=11_inst=10_reps=10_nComm=2_commSize=50.csv')
	plt.clf()
	print(np.mean(df[df['algorithm']=='hedonic']['accuracy']))
	print(np.mean(df[df['algorithm']=='naive']['accuracy']))
	print(len(df))
	# sns.jointplot(data=df, x="robustness", y="accuracy", hue="algorithm", kind="scatter") # scatter kde hist hex reg resid
	# sns.violinplot(x='algorithm', y='seconds', data=df, fontsize=27.5) # [df['method']=='rand']
	sec0 = df[df['algorithm']=='naive']['seconds']
	sec0 = [s for s in sec0 if s > .01]
	sec1 = df[df['algorithm']=='hedonic']['seconds']
	sec1 = [s for s in sec1 if s > .01]
	# plt.hist(sec) # [df['method']=='rand']
	sns.violinplot(sec0, alpha=.1, color='green') # [df['method']=='rand']
	sns.violinplot(sec1, alpha=.1, color='red') # [df['method']=='rand']
	plt.show()

##########################
## Python Naive vs Networkx

def plot_python_vs_netx(filename=''):
	df = pd.read_csv(filename)
	plt.clf()
	# matplotlib.rc('xtick', labelsize=font_size*.8)
	# matplotlib.rc('ytick', labelsize=font_size*.8)
	fig, ax = plt.subplots(dpi=200) # nrows=2, ncols=4, figsize=(60,30)  num=None , dpi=250 , figsize=(7.8*4, 6.3)
	sns.lineplot(x='mults', y='secs', hue='alg', data=df, marker='o', linewidth=4, ax=ax) # , ax=ax
	ax.set_title('time comparison between implementations') # , fontsize=font_size
	ax.set_xlabel('q/p') # algorithms , fontsize=font_size*.9
	ax.set_ylabel('seconds') # , fontsize=font_size*.9
	handles, labels = ax.get_legend_handles_labels()
	labels = ['previous','current'] # 'implementations'
	for h in handles:
		h.set_linewidth(4)
	leg = ax.legend(handles=handles, labels=labels) # , loc=leg_loc
	plt.savefig(f'{filename[:-4]}_time.png')

################################################################################

def plot_hists(filename):
	# ['nodes', 'max_comp', 'edges', 'gt_balance', 'p_in', 'mult', 'instance', 'repetition', 'algorithm', 'accuracy', 'robustness', 'method', 'seconds']
	df = pd.read_csv(filename)

	plt.clf()
	plt.hist(df['gt_balance'])
	plt.savefig('gt_balance.png')

	plt.clf()
	plt.hist(1-(df['max_comp']/df['nodes']))
	plt.savefig('isolated_nodes.png')

	plt.clf()
	plt.hist(df['edges'] / (df['max_comp']*(df['max_comp']-1)/2))
	plt.savefig('edge_density.png')

##############################################################################
## Main ############################################################################

if __name__ == "__main__":
	# df = load_df()
	
	df, dfs_noise, speed, df_A, realnets_df = load_datas()
	# generate_noises_plot()
	# generate_gif()
	generate_fig_1()

	# name = 'max_components__ps=10_mults=11_inst=10_reps=10_nComm=2_commSize=500.csv'
	# fname = f'outputs/comparisons/news/{name}'
	
	# generate_plot_comparion(fname)
	# plot_hists(fname)

	# df = pd.read_csv('outputs/comparisons/comparison_commSize=111.csv')
	# for alg in df['algorithm'].unique():
	# 	plt.clf()
	# 	plt.violinplot(df[df['algorithm']==alg]['seconds'])
	# 	plt.savefig(f'{alg}.png')
	# sns.violinplot(x='algorithm', y='seconds', data=df[df['method']=='rand'], fontsize=27.5, ax=ax)

	# plot_robust_vs_naive()

	# plot_python_vs_netx('outputs/speed_python_vs_netx/speed_test__ps=5_mults=6_inst=10_reps=10_nComm=2_commSize=250.csv')