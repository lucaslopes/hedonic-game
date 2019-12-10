import csv

def create_gt_row(gt_col):
    with open(f'{gt_col}.csv', newline='') as f:
        table = csv.reader(f)
        row = next(table)
        nodes = []
        for node in row:
            nodes.append(node)
        row = next(table)
        clusters = []
        for cluster in row:
            clusters.append(cluster)
        new_gt = open(f'{gt_col}_row.csv', 'w+')
        for n, c in zip(nodes, clusters):
            new_gt.write(f'{n},{c}\n')
        new_gt.close()

def set_ground_truth(self, file): # old (cols)
    g_t, duration = {}, datetime.now()
    if Path(file).is_file():
        with open(file, newline='') as f:
            table = csv.reader(f)
            row = next(table)
            nodes = []
            for node in row:
                nodes.append(node)
            row = next(table)
            clusters = []
            for cluster in row:
                clusters.append(cluster)
            for i in range(len(nodes)):
                self.nodes[nodes[i]].ground_truth = clusters[i]
    else:
        for node in self.nodes:
            self.nodes[node].ground_truth = 'none'
    self.results['import_duration'] += datetime.now() - duration
