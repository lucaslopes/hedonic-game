import random

def greedy(game):
    done = False
    while done is False:
        current_record  = 0
        profitable_node = None
        for node in list(game.nodes):
            p = game.profit(node)
            if p > current_record:
                current_record  = p
                profitable_node = node
        if profitable_node is not None:
            game.move(profitable_node)
        else:
            done = True

def stochastic(game):
    done = False
    while done is False:
        nodes = list(game.nodes)
        find = False
        while find is False:
            i = random.randrange(0, len(nodes))
            p = game.profit(nodes[i])
            if p > 0:
                find = True
                game.move(nodes[i])
            else:
                nodes[i], nodes[-1] = nodes[-1], nodes[i]
                nodes.pop()
                if len(nodes) == 0:
                    done = find = True

def sequential(game):
    moved = True
    while moved is True:
        moved = False
        for node in game.nodes:
            if game.profit(node) > 0:
                game.move(node)
                moved = True

def pique_pega(game):
    def next_it(list):
        for node in list:
            if game.profit(node) > 0:
                game.move(node)
                return node
        return False
    it = next_it(game.nodes)
    done = False
    while not done:
        if it is not False:
            it = next_it(game.nodes[it].friends)
        else:
            it = next_it(game.nodes)
            if it is False: done = True
