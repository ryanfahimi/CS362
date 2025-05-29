# We're implementing a graph as an adjacency list.
# The keys will be nodes.
# The values will be lists of edges.
# We'll use strings to store the values for src and dest.
# for example '1,1'


class Node:
    # val will be a string like '1,1'
    def __init__(self, val=""):
        self.value = val

    # all Nodes with the same value will have the same hash.
    def __hash__(self):
        return hash(self.value)

    def __repr__(self):
        return str(self.value)

    # all nodes with the same value are equal
    def __eq__(self, other):
        return self.value == other.value


## An edge has a src, a dest, and a value.
## For the mars_map, the value is always 1.
# if we had other maps, such as Romania, we would store this here.
class Edge:
    def __init__(self, src, dest, val=0):
        self.src = src
        self.dest = dest
        self.val = val

    def __repr__(self):
        return "(%s %s %d)" % (self.src, self.dest, self.val)


class Graph:
    def __init__(self, n_vertices=5):
        ## our adjacency list
        self.g = {}

    def __repr__(self):
        return "\n".join(
            [
                f"{node}: {' '.join(str(edge.dest) for edge in edges)}"
                for node, edges in self.g.items()
            ]
        )

    def add_node(self, index):
        self.g[index] = []

    def add_edge(self, e):
        self.g[e.src].append(e)

    def get_edge(self, src, dest):
        if src in self.g:
            edges = self.g[src]
            for e in edges:
                if e.dest == dest:
                    return e

    def get_edges(self, src):
        if src in self.g:
            return self.g[src]
