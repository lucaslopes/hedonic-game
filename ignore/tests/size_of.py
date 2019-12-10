import sys
import pickle

class Node:
    def __init__(self):
        self.friends = { 'a' : True }
        self.group = True
        self.with_me = 0

with open('my_node.pkl', 'wb') as output:
    node = Node()
    pickle.dump(node, output, pickle.HIGHEST_PROTOCOL)

del node

with open('my_node.pkl', 'rb') as input:
    node = pickle.load(input)
    print(node.friends)
    print(node.group)
    print(node.with_me)

################################################################################

nodes_dict = {}
for i in range(100):
    nodes_dict[f'{i}'] = { 'friends' : {'a':True}, 'group' : True, 'with_me' : 0 }
print(sys.getsizeof(nodes_dict))

nodes_obj = {}
for i in range(100):
    nodes_dict[f'{i}'] = Node()
print(sys.getsizeof(nodes_obj))
