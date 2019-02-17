import csv

def insert(d, a, b):
    if a not in d:
        d[a] = [b]
    elif b not in d[a]:
        d[a].append(b)
    return d

def csv_2_dict(file):
    d = {}
    with open(file, 'r') as f:
        table = csv.reader(f)
        for row in table:
            a = int(row[0])
            b = int(row[1])
            d = insert(d, a, b)
            d = insert(d, b, a)
    return d
