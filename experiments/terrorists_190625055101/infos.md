# Parameters
- Graph:                  terrorists
- Alpha:                  0.95
- ¹Initial Configuration: {'mode': 's', 'params': ['40', '42', '51', '54']}
- Verbose Mode:           True
- Accuracy Frequency:     1
- commentaries:           Begin with Cluster 2

# Results
- Finished at:            2019-06-25 05:51:01
- Converged in:           0:00:00.025195
- Initial Potential:      -1440.05
- Final Potential:        -747.45
- Accumulated Gain:       692.60
- ²Potential Gain in %:   92.66%
- Iterations:             38

# Legend
- ¹Initial Configuration
  - Modes Configurations:
    - Random (r): Nodes will be randomly selected to start inside cluster
    - Select (s): Chose which nodes will start inside cluster
    - Any other:  Start with an empty cluster
  - Modes Parameters:
    - Random (r): Number of nodes - If it is between 0 and 1 will be multiply by the number of nodes
    - Select (s): List of selected nodes. e.g. [node indice, ..., node indice]
- ²Potential Gain in %
  - Accumulated Gain / Initial Potential * 100