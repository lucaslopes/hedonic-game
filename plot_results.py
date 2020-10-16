import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns; sns.set(style="whitegrid")
from PIL import Image

##############################################################################
## Load Data ############################################################################

def load_df():
	algorithms = { 'spectral':'spectral clustering', 'onepass':'local improvement', 'hedonic':'hedonic robust' }
	prefix = 'outputs/noises/'
	file_name_prefix = ''
	df = pd.concat([pd.read_csv(f'{prefix}{csv}') for csv in os.listdir(prefix) if csv[-3:] == 'csv'])
	df['p_in-p_out'] = df['p_in'] - df['p_out']
	return df

##############################################################################

def load_dfs_noise():
	dfs_noise = dict(tuple(df.groupby('noise')))
	for noise, data in dfs_noise.items():
		df_noise = pd.DataFrame(columns=['mult']+[f'{alg}_accuracy' for alg in algorithms])
		algs = [f'{alg}_accuracy' for alg in algorithms]
		for mult in data['mult'].unique():
			# print('noise =', noise, '-- mult =', mult)
			for inst in data['instance'].unique():
				df_noise = df_noise.append({
					'mult'  : mult,
					algs[0] : np.mean(data.loc[(data['mult'] == mult) & (data['instance'] == inst), [algs[0]]].values),
					algs[1] : np.mean(data.loc[(data['mult'] == mult) & (data['instance'] == inst), [algs[1]]].values),
					algs[2] : np.mean(data.loc[(data['mult'] == mult) & (data['instance'] == inst), [algs[2]]].values),
				}, ignore_index=True)
		# df_noise = data.loc[:, ['mult']+[f'{alg}_accuracy' for alg in algorithms]]
		df_noise = df_noise.melt('mult', var_name='algorithm', value_name='accuracy')
		df_noise['algorithm'] = [alg.split('_')[0] for alg in df_noise['algorithm'].values]
		df_noise['algorithm'] = [algorithms[alg] for alg in df_noise['algorithm'].values]
		df_noise.rename(columns = {'mult':'q/p'}, inplace = True)
		dfs_noise[noise] = df_noise
	return dfs_noise

##############################################################################

def load_speed():
	speed = df.loc[:, ['noise']+[f'{alg}_time' for alg in algorithms]]
	speed = speed.melt('noise', var_name='algorithm', value_name='seconds')
	speed['algorithm'] = [alg.split('_')[0] for alg in speed['algorithm'].values]
	speed['algorithm'] = [algorithms[alg].replace(' ','\n') for alg in speed['algorithm'].values]
	return speed

##############################################################################

def load_df_A():
	denom = 1468.49
	df_A = pd.DataFrame()
	df_A['noise']    = [round(x, 3) for x in [0, 0.025] + list(np.linspace(0, .5, 11))[1:-1] + [0.475, .5]]
	df_A['spectral'] = [0,(denom-1028.25)/denom,(denom-900.48)/denom,(denom-710.51)/denom,(denom-600.57)/denom,(denom-536.37)/denom,(denom-474.91)/denom,(denom-414.02)/denom,(denom-370.85)/denom,(denom-325.62)/denom,(denom-245.38)/denom,(denom-170.39)/denom,1]
	df_A['onepass']  = [0,(denom-824.52)/denom,(denom-700.02)/denom,(denom-591.54)/denom,(denom-543.63)/denom,(denom-512.04)/denom,(denom-487.50)/denom,(denom-470.42)/denom,(denom-462.15)/denom,(denom-453.79)/denom,(denom-448.22)/denom,(denom-389.49)/denom,1]
	df_A['hedonic']  = [0,(denom-857.10)/denom,(denom-715.39)/denom,(denom-575.51)/denom,(denom-502.74)/denom,(denom-438.50)/denom,(denom-397.13)/denom,(denom-362.14)/denom,(denom-338.83)/denom,(denom-302.52)/denom,(denom-303.45)/denom,(denom-217.39)/denom,1]
	df_A = df_A.melt('noise', var_name='algorithm', value_name='r = maximum tolerated q/p') # r is the maximum value of q/p for which we observe gains in the corresponding method when contrasted against simple replication of the input as the output, i.e., maximum value of q/p for which the accuracy of the method is greater than 1-noise
	df_A['algorithm'] = [algorithms[alg] for alg in df_A['algorithm'].values]
	return df_A

##############################################################################

def load_realnets_df():
	prefix = 'outputs/real_nets'
	csv_name = '4_networks_10_repts'
	realnets_df = pd.read_csv(f'{prefix}/{csv_name}.csv')
	algs = []
	for alg in realnets_df['algorithm']:
		if alg != 'spectral':
			alg = alg.split('_')
			alg[0] = algorithms[alg[0]]
			alg = f'{alg[0]} (noi={alg[1][2:]})'
		algs.append(alg)
	realnets_df['algorithm'] = algs
	return realnets_df

##############################################################################

def load_datas():
	return load_df(), load_dfs_noise(), load_speed(), load_df_A(), load_realnets_df()

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
	if labels_handles is None:
		handles, labels = handles[1:], labels[1:]
	else:
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

	noise_graph = .35 # Choose one noise for be in Fig 1 (a)
	axes[0].set_ylim(.499,1.01)
	g = sns.lineplot(x="q/p", y="accuracy", hue="algorithm", data=dfs_noise[noise_graph], marker='o', linewidth=4.5, ci=95, ax=axes[0]) # , style="event"
	axes[0].axhline(1-noise_graph, c='red', alpha=.75, ls='--', lw=2.5)
	axes[0].annotate('$1$-noise', (.85, 1.005-noise_graph), fontsize=40)
	plt.text(0, 0.1, '', bbox=dict(boxstyle="round", fc="white", ec="red", pad=0.2))
	noise_name = str(round(noise_graph,3))+'00000000'
	save_plot(axes[0], title=f'(a) accuracy vs q/p (noise={noise_name[:5]})', x_label='q/p', y_label='accuracy', f_name=f'{file_name_prefix}_n{round(noise_graph,3)}')

	##############################################################################

	g = sns.lineplot(x="noise", y="r = maximum tolerated q/p", hue="algorithm", data=df_A, marker='o', linewidth=4.5, ax=axes[1])
	save_plot(axes[1], title='(b) max q/p such that accuracy$\geq1$-noise', x_label='noise', y_label='r = maximum tolerated q/p', f_name='cross')

	##############################################################################

	g = sns.violinplot(x="algorithm", y="seconds", data=speed, fontsize=27.5, ax=axes[2])
	save_plot(axes[2], title='(c) time efficiency', x_label='', y_label='seconds', f_name='time', has_legend=False) # algorithm

	##############################################################################

	alg_repetition = int((len(realnets_df['algorithm'].unique())-1)/2) # #5975A4 sns.color_palette("Blues", n_colors=1)
	cor = [('#5975A4')]+sns.color_palette("Oranges", n_colors=alg_repetition)+sns.color_palette("Greens", n_colors=alg_repetition) # #5975A4 #CC8964 #5F9E6E
	g = sns.barplot(x="network", y="accuracy", hue="algorithm", data=realnets_df, palette=cor, ax=axes[3])
	max_acc = []
	for net in ['karate', 'dolphins', 'pol_blogs', 'pol_books']:
		nets_df = realnets_df[realnets_df['network'] == net]
		for alg in nets_df['algorithm'].unique():
			nets_alg_df = nets_df[nets_df['algorithm'] == alg] # .max()
			max_acc.append(nets_alg_df['accuracy'].max())
	x_pos = [p.get_x() for p in axes[3].patches]
	x_pos.sort()
	colors = ['#5975A4']+['#CC8964' for _ in range(alg_repetition)]+['#5F9E6E' for _ in range(alg_repetition)]
	for pos, maxx, c in zip(x_pos, max_acc, colors*4):
		axes[3].annotate('_', (pos, maxx), c=c, weight=1000) # p.get_height() * 1.005)
	axes[3].set_ylim(0.499,1.01)
	labls = {'spectral clustering':'#5975A4', 'local improvement\n(0$\leq$noise$\leq$0.5)':'#CC8964', 'hedonic robust\n(0$\leq$noise$\leq$0.5)':'#5F9E6E'}
	save_plot(axes[3], title='(d) real networks', x_label='network', y_label='accuracy', f_name='real_nets_bar_plot', n_col=1, labels_handles=labls, font_scale=.7)
	
	##############################################################################

	plt.tight_layout()
	fig.savefig('fig1.png')

##############################################################################
## Plot Comparison ############################################################################

def generate_plot_comparion(filename=''):
	df = pd.read_csv(filename)
	methods = df['method'].unique()
	for method in methods:
		df_method = df[df['method']==method]
		plt.clf()
		sns.lineplot(x='mult', y='accuracy', hue='algorithm', data=df_method, marker='o', linewidth=4.5) # , ax=axes[1]
		plt.savefig(f'{filename[:-4]}_{method}.png')
	plt.clf()
	sns.violinplot(x='algorithm', y='seconds', data=df, fontsize=27.5) # , ax=axes[2]
	plt.savefig(f'{filename[:-4]}_time.png')

##############################################################################
## Main ############################################################################

if __name__ == "__main__":
	# df, dfs_noise, speed, df_A, realnets_df = load_datas()
	# generate_noises_plot()
	# generate_gif()
	# generate_fig_1()

	generate_plot_comparion('outputs/comparisons/comparison_commSize=50.csv')
 