from collections import defaultdict, deque


class Graph:
    def __init__(self, directed):
        self.graph = defaultdict(list)
        self.directed = directed

    def addEdge(self, v, w):
        if self.directed:
            self.graph[v].append(w)
        else:
            if v not in self.graph:
                self.graph[v] = []
            if w not in self.graph:
                self.graph[w] = []
            self.graph[v].append(w)
            self.graph[w].append(v)

    def bfs(self, s, g=None):
        queue = deque([s])
        visited = set([s])

        while queue:
            node = queue.popleft()
            print(node, end=" ")
            if node == g:  # If a goal node is provided and found, stop the search
                break
            for neighbour in self.graph[node]:
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append(neighbour)


# dataGraph = {
#     "0": {"1", "4"},
#     "1": {"0", "2", "3", "4"},
#     "2": {"1", "3"},
#     "3": {"1", "2", "4"},
#     "4": {"0", "1", "3"},
# }

# graph = Graph()

# for node, neighbours in dataGraph.items():
#     for neighbour in neighbours:
#         graph.addEdge(node, neighbour)
# graph.bfs("0", "3")


dataGraph = {
    "A": {"B", "C", "D"},
    "B": {"E", "F"},
    "C": {"G"},
    "D": {},
    "E": {},
    "F": {},
    "G": {},
}

graph = Graph(True)
for node, neighbours in dataGraph.items():
    for neighbour in neighbours:
        graph.addEdge(node, neighbour)
graph.bfs("A", "G")
