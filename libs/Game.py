from .export     import *
from .convert    import csv_2_dict
from  datetime   import datetime
from .hedonics   import *
from .operations import move, set_graph

class Game:

    def __init__(self, p):
        self.graph     = p['graph']
        self.player    = p['player']
        self.alpha     = p['alpha']
        self.gamma     = p['gamma']
        self.limit     = p['limit']
        self.errors    = p['errors']
        self.path      = 0
        self.score     = 0
        self.clusters  = 0
        self.potential = 0
        self.config()

    def config(self):
        g      = self.graph
        now    = str(datetime.now())[2:-7]
        player = str(self.player)[10:-16]
        for _ in ': -': now = now.replace(_, '')
        self.path   = 'experiments/' + g + '-' + player + '-' + now  + '/'
        self.graph  = csv_2_dict('datasets/' + g + '.csv')
        self.errors = round(self.errors * len(self.graph))
        create_folder(self.path, self.graph)
        set_graph(self.graph)
        set_params(self.graph, self.alpha, self.gamma)
        # to-do: Init conditions
        self.clusters  = [self.graph]
        self.potential = global_potential(self.clusters)

    def start(self):
        c     = self.clusters
        begin = datetime.now()
        valid = invalid = 0
        running = True
        while running:
            if len(c) < self.limit: c.append({})
            p = self.player(c) if invalid < self.errors else None
            if self.valid(p):
                #diff = perspectives(c[p[1]], p[0], c[p[2]], self.clusters)
                diff = hedonic(p[0], c[p[2]]) - hedonic(p[0], c[p[1]])
                if diff > 0:
                    self.make_move(p[0], p[1], p[2], diff)
                    valid, invalid = self.update(valid, invalid)
                else: invalid += 1
            else:
                m = self.has_moves()
                if m:
                    self.make_move(m[0], m[1], m[2], m[3])
                    valid, invalid = self.update(valid, invalid)
                else:
                    register(valid, invalid)
                    running = False
            while {} in c: c.remove({})
        self.finish(datetime.now() - begin)

    def update(self, valid, invalid):
        if invalid > 0:
            register(valid, invalid)
            valid   = 1
            invalid = 0
        else:
            valid += 1
        return valid, invalid

    def valid(self, mv):
        c = self.clusters
        if type(mv) == list and len(mv) >= 3:
            if type(mv[0]) == type(mv[1]) == type(mv[2]) == int:
                if mv[1] < len(c) and mv[2] < len(c):
                    if mv[0] in c[mv[1]] and mv[0] not in c[mv[2]]:
                        return True
        return False

    def has_moves(self):
        c = self.clusters
        for here in c:
            for key in here:
                i_am = hedonic(key, here)
                for there in c:
                    diff = hedonic(key, there) - i_am
                    if diff > 0:
                        return [key, c.index(here), c.index(there), diff]
        return False

    def make_move(self, node, From, To, increased):
        self.clusters[From], self.clusters[To] = move(
            node, self.clusters[From], self.clusters[To])
        self.score += increased
        save_step([node, From, To, increased])
        #print('move:', node, From, To, increased)

    def finish(self, time):
        r = {
            'path'   : self.path,
            'now'    : str(datetime.now())[:-7],
            'pot'    : self.potential,
            'score'  : self.score,
            'gain'   : self.score * 100 / abs(self.potential),
            'time'   : time,
            'graph'  : self.graph,
            'player' : self.player,
            'alpha'  : self.alpha,
            'gamma'  : self.gamma,
            'limit'  : self.limit,
            'errors' : self.errors,
            'clusters' : self.clusters }
        results(r)
