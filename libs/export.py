from pprint import pformat
import os

graph = path = steps = tries = heds = None

def create_folder(p, g):
    os.makedirs(os.path.dirname(p), exist_ok = True)
    global graph, path, steps, tries
    graph = g
    path  = p
    steps = open(p + 'iterations.csv', 'w+')
    tries = open(p + 'tries.csv', 'w+')
    steps.write('move, from, to, increased')
    tries.write('valid, invalid')

def register(valid, invalid):
    r = '\n{}, {}'.format(valid, invalid)
    tries.write(r)

def save_step(n):
    if n[0]:
        s = '\n{:02d}, {:02d}, {:02d}, {:.2f}'.format(n[0], n[1], n[2], n[3])
        steps.write(s)

def save_values(n, f, t, v):
    global heds, path
    if heds == None:
        heds = open(path + 'values.csv', 'w+')
        heds.write('node, from, to, hed_cmplt, hed, pot, sum_hed-hed_cmplt, sum_hed-hed')
    h = '\n{:02d}, {:02d}, {:02d}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}'.format(
        n, f, t, v[0], v[1], v[2], v[3], v[4])
    heds.write(h)

def results(r):
    gain = '+{:.2f}%'.format(r['gain'])
    p = open(path + '_info.txt','w+')
    p.write('----- Parameters -----\n')
    p.write('Graph        -> %s\r\n' % path[12:path.index('-')])
    p.write('Player       -> %s\r\n' % str(r['player'])[10:-16])
    p.write('Alpha        -> %s\r\n' % str(r['alpha']))
    p.write('Gamma        -> %s\r\n' % str(r['gamma']))
    p.write('Max Clusters -> %s\r\n' % str(r['limit']))
    p.write('Errors limit -> %s\r\n' % str(r['errors']))
    p.write('-----  Results  -----\n')
    p.write('Finished at    -> %s\r\n' % str(r['now']))
    p.write('Converged in   -> %s\r\n' % str(r['time']))
    p.write('Init Potential -> %s\r\n' % str(r['pot']))
    p.write('Score          -> +%s\r\n' % str(r['score']))
    p.write('Gain in per100 -> %s\r\n' % gain)
    # How many moves?
    # Time interval between moves
    p.close()
    steps.close()
    tries.close()
    if heds != None: heds.close()

    g = open(path + 'graph.txt','w+')
    g.write('%s\r\n' % pformat(graph))
    g.close()

    i = 0
    for c in r['clusters']:
        title = 'cluster_' + str(i)
        f = open(path + title + '.txt','w+')
        f.write('%s\r\n' % pformat(c))
        f.close()
        i += 1
