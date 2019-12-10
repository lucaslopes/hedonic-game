class Node:

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def load(file):
        self = Node(file['name'], file['age'])
        return self

file = {}
file['name'] = 'lucas'
file['age'] = 24

meu_no = Node.load(file)
print(meu_no.name, meu_no.age)
