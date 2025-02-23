from collections import defaultdict


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

    # Using stack for DFS

    def dfs(self, s, g=None):
        stack = []
        visited = set()

        stack.append(s)
        visited.add(s)

        while stack:
            node = stack.pop()
            print(node, end=" ")
            if node == g:  # If a goal node is provided and found, stop the search
                break
            for neighbour in self.graph[node]:
                if neighbour not in visited:
                    visited.add(neighbour)
                    stack.append(neighbour)

    # recursive function

    # def dfs(graph, start, visited=None):
    #     if visited is None:
    #         visited = set()
    #     visited.add(start)
    #     print(start, end=" ")

    #     for neighbor in graph[start]:
    #         if neighbor not in visited:
    #             dfs(graph, neighbor, visited)
    #     return visited


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

graph = Graph(False)
for node, neighbours in dataGraph.items():
    for neighbour in neighbours:
        graph.addEdge(node, neighbour)
graph.dfs("A", "G")
