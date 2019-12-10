# Hedonic Games for Clustering
- A research experiment of ^Giovanni, ^Kostya, ´Daniel, ´Lucas (ack ´Eduardo)
- ^INRIA ∣ ´Federal University of Rio de Janeiro
- [Demonstration](https://lucaslopes.github.io/hedonic/)

## To Run

1. run `python app.py` in the terminal

<!-- 1. Open `hedonic.py`
2. Set the parameters
3. Save the file
4. run `python hedonic.py` in the terminal -->

## Folders

### networks
Contains the data file for the network. You can check existing ones [here]('https://github.com/lucaslopes/hedonic/blob/master/networks/README.md'), or add yours. Requirements:
1. Must be `.csv`
2. All rows is a edge is the form: `node,node`
3. Nodes can be represented as `numbers` or `strings`
- Sample of `hedonic/networks/my_network.csv`:
```
0,1
1,2
1,3
2,3
```

#### `./ground_truth`
Ground Truths are the correct classification of each node in a network. Requirements:
1. Must be `.csv`
2. Each `column` represents a node
3. First `row` indicate the index/label of a node
4. Second `row` indicate the cluster where each node is in
- Sample of `hedonic/networks/ground_truth/my_network.csv`:
```
0,1,2,3
in,in,out,out
```

#### `./converted`
Datasets that was already read in `.csv` saved in JSON as `.txt` for fast load in future experiments. No need to touch this folder, let the algorithm populate it (unless you'd like to clear it).
- Sample of `hedonic/networks/converted/my_network.txt`:
```
{"0": [1], "1": [0, 2, 3], "2": [1, 3], "3": [2]}
```

#### `./details`
No need to edit, just read. Here are files containing informations about the datasets and/or its plot images.

### experiments
Here is where the raw data, exported by the algorithm, is saved. Each folder is a experiment where is saved in this format: `{name-of-the-network}_{timestamp-at-beginning}`. It contains:
- `infos.md`: Informations about the experiment
- `states.csv`: The state (classification of each node) at every iteration of the experiment
- `properties.csv`: Number of Vertices and Edges of each cluster
- `accuracy.csv`: Is obtained in respected with the Ground Truth:

  - **True Positive**: In *Cluster* **and** In *Ground Truth*
  - **True Negative**: Not in *Cluster* **and** Not in *Ground Truth*
  - **False Positive**: In *Cluster* **and** Not in *Ground Truth*
  - **False Negative**: Not in *Cluster* **and** In *Ground Truth*
- `iterations.csv`: Each row/iteration contains the following columns:

  - **move**: The node that were moved
  - **from**: The cluster from where it was
  - **to**: The cluster where it went
  - **increased**: How much the potential increased
  - **accumulated**: How much the potential has increased at this point
  - **global**: The current Global Potential
  - **accuracy**: The current accuracy compared to the Ground Truth

## Files

### `README.md`
You're reading it.

### `./plot.ipynb`
A Jupyter Notebook that plot visualizations of each experiment.

### `index.html`
The [Jupyter Notebook](#TO-DO) containing visualizations of the experiments, converted in `.html` and hosted at https://lucaslopes.github.io/hedonic/.

### `hedonic.py`

#### Parameters
- **Graph**: The name of the Network that will be clusterized
- **Alpha** The Fragmentation Factor $0 \le \alpha \le 1$
- **Init**'
  - Initial Modes:
    - Random `r`: Nodes will be randomly selected to start inside cluster
    - Select `s`: Chose which nodes will start inside cluster
    - Any other: Start with an empty cluster
  - Modes `params`:
    - Random `r`: Number of nodes - If is between 0 and 1 will be multiply by quantity of nodes
    - Select `s`: List of selected nodes `[node indice, ..., node indice]`
- **Print**: If `True` will print in the terminal the node moved at each iteration
- **Freq**: Probability that Accuracy will be computed at each iteration (because it costs time)
- **Note**: A string to comment about the experiment

#### Import Dependecies

- **datetime from datetime**: To print the time
- **Path from pathlib**: To check if a file exist
- **random**: To select nodes randomly in the Random`r` init mode
- **json**: To dump and import graphs
- **csv**: To and import graphs
- **os**: To create folders

#### A Hedonic Game

- `class Game`:
    - `__init__(self, p)`:
        - **graph**: A dictionary (vertices) of dictionaries (its connections)
        - **folder**: String that is the path of experiment directory
        - **g_truth**: The Ground Truth, the correct classification of nodes/clusters
        - **classes**: The
        - **clusters**: Stores the number of Vertices and Edges of each Cluster
        - **potential**: The Potential value of the partition
        - **accuracy**:
        - **iteration**: Stores
        - **score**: The accumulated gain of Potential
    - `start(self)`: Function for begin the experiment
      1. Check the node that it's more profitable to move (highest difference between its Hedonic Value in the other cluster compared the where is current in)
      2. If there's one, move it. If don't (all moves are unprofitable), game is over

#### Load the Network
- `load_network(file)`: Check if there's a already converted version of the dataset, is don't, import from `.csv`
- `csv2dict(file)`: Convert from `.csv` to a Python dictionary

#### Setters
- `set_game_path(graph)`: Set the path where the raw data of the experiment will be saved
- `set_ground_truth(file, graph)`: Loads the respective Ground Truth
- `set_classes(param, nodes)`: Determine the inital classification of nodes, based on the [init]() parameter;
- `set_accuracy(g_truth, clusters)`: For each combination between the clusters of the algorithm (In and Out) and the clusters of the Ground Truth, there's a *Confusion Matrix*, as show in [experiment/`accuracy.csv`](##experiment)

#### Counting
- `check_friends(node)`: Check how many connections are in the same cluster that node, and how many are in the other one
- `count_verts_edges(Class, graph)`: Count the number of vertices and edges of each cluster

#### Hedonic Functions
- `hedonic(neighbors, strangers, alpha=p['alpha'])`: Return the *Hedonic Value* of been in a cluster with `x` neighbors (connections), `y` strangers (not connected with), and with a weight determined by $\alpha$
- `global_potential(clusters, alpha=p['alpha'])`: Return the sum of the Potential of each cluster
- `profit(node, us=None, them=None, alpha=p['alpha'])`: Return the difference between the *Hedonic Value* in the other cluster, and the one which the node is currently part of

#### Accuracy
- `accuracy()`: Check how many `TP`, `TN`, `FP` and `FN` are in each combination of clusters (from Experiment and Ground Truth)

#### Move a Node
- `move(node, increased=None)`: Move a node from the cluster where it is currently in, to the other cluster

#### Export Results
- `create_files(path, cluster_combinations)`: Create the files from where the raw data will be saved
- `stringify_nodes()`: Convert the list of nodes in a String
- `stringify_state()`: Convert the list of classes in a String
- `print_parameters()`: Show the Parameters at the beginning of the experiment
- `timestamp(node, increased)`: Timestamp the current state of the experiment in the files
- `results(duration)`: Save experiment results

#### Initiate Experiment

```
game = Game(p) # 1.
create_files(game.folder, len(game.accuracy.keys())) # 2.
game.start() # 3.
results(datetime.now() - game.begin) # 4.
```
1. Create the Game based on the parameters defined
2. Create the files where the output of experiment will be saved
3. Initiate the experiment
4. Saved results

####  Legend (to-do)
- ¹Initial params:
  - Modes Commands:
    - Random (r): Nodes will be randomly selected to start inside cluster
    - Select (s): Chose which nodes will start inside cluster
    - Any other:  Start with an empty cluster
  - Modes Parameters:
    - Random (r): Number of nodes - If it is between 0 and 1 will be multiply by the number of nodes
    - Select (s): List of selected nodes. e.g. [node indice, ..., node indice]
- ²Potential Gain in %
  - Accumulated Gain / Initial Potential * 100"
