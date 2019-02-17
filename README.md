# Hedonic Clustering
> _Finding Clusters with Cooperative Game Theory._<br/>
> A research experiment of _Daniel Sadoc_$^\ast$ and _Lucas Lopes_$^\ast$.<br/>
> $^\ast$Federal University of Rio de Janeiro.<br/>
> December, 2018.

<!---
Todo:
1. Explain the Hedonic function
2. Explain how the learning processing works
-->


<!--------------------------------------------------------------->

## Parameters
For convenience and reachability, let's put the parameters on top.<br/>
To understand each parameter, click: [dataset](#converting-a-csv-file-to-a-python-dictionary), [initial](#creating-the-initial-condicions), [verbose](#printing-the-algorithm-steps), [alpha](#-main-function)

--------------------------------------------------------------------------------

# Helper Functions

### Converting a CSV file to a Python Dictionary

Since the graph network is represented as a dictonary in the form$^\ast$, and usually a dataset that stores a network is in the _.csv_ format, it is helpful to have a function that convert a dataset in a form that the algorithm can read and operate on it.

To do so, the _.csv_ file must be in the following way:

| From Node | To Node  |
| :-------: | :------: |
| _number_  | _number_ |

$^\ast$**Key** is the vertices; and **Value** is a list of its connections.

--------------------------------------------------------------------------------

##Doing graph operations in a Dictionary

It is possible to do many operations in a graph, such as:

- Add a new vertice;
- Remove an old one;
- Move a vertice from a graph to another.

So here are helper functions to do this above operations. But before that, we need to import the _copy_ library because of this$^\ast$.

$^\ast$`dict.copy()` method do a [shallow copy](https://docs.python.org/2/library/stdtypes.html#dict.copy). We need a [deep copy](https://docs.python.org/2/library/copy.html#copy.deepcopy) to avoid [this problem](https://stackoverflow.com/questions/3975376/understanding-dict-copy-shallow-or-deep).

--------------------------------------------------------------------------------

## Creating the initial condicions

We have 2 graphs, one is the _cluster_ that we're looking for, e the other one is the original graph without those nodes in the cluster (a.k.a _remain_).

These 2 graphs (_cluster_ and _remain_) could be initiate in different ways, varying by answer this questions:

1. How many nodes?
2. It will be manually selected or random choices?

To do this different combinations, just change the value of the variable `init`.<br/>
There are 3 modes and its respectives parameters:

1. **Random mode**<br/>
    - Select a specific number of vertices randomly.<br/>
    - `['r', num]` where `num` is the amount that will be selected.

2. **Select mode**<br/>
    - Inform the specific vertices that you want to initiate.<br/>
    - `['s', num]` where `num` could be just the of vertice or a list of them.

3. **Empty mode**<br/>
    - Initiate with no vertices.<br/>
    - `['e']` (It is the same that initiate with all node, just changes the reference)

--------------------------------------------------------------------------------

### Printing the algorithm steps

If the verbose mode is activate (_i.e._ `verbose == True`) it will print, at each iteration, this informations:

- The nodes and its hedonic value for move and stay where they are;
- Which nodes are in the _cluster_ and which is in _remain_;
- Who will move and what is its alpha.

--------------------------------------------------------------------------------

# Perspectives

The hedonic function tells how good a node is 'fit' in a given network.
It takes:

1. A $\alpha$ value, where $\alpha \in [0,1]$ (_closer to 1 is broader_);
2. The number of vertices in a particular graph;
3. The number of connections that a node (in this graph) has.

And returns the _"score"_ of that node.

1. Individual
2. Local
3. Global

--------------------------------------------------------------------------------

# Algorithms

1. Stochastic
2. Partial
3. Greedy

--------------------------------------------------------------------------------

# Trainning

### Set the original graph and initial condicions

- Convert the dataset that was selected to a dictonary;
- Read the initiate option.

## Export the Graphs

- Export the original graph, the result cluster, and the remain, each of them in its respective _.txt_ file.

### $\Delta$ Cost
### Cost
### $\Delta$ Cost Scatter
### In-Out Matrix
### Relation Matrix
### Ground-Truth Comparison

---------------------------------------------

# Datasets
> Folder containing the network graphs used in the experiment

To use a network dataset in the code, it should follow this requirements:

1. **File format**
It should be a _.csv_ file

2. **Collums**
It should have 2 collums:

| From Node | To Node  |
|-----------|----------|
| _number_  | _number_ |

3. **Put in Code**
Replace de _file_ variable with the path to _.csv_ file
