# -*- coding: utf-8 -*-
"""
Author:Fateme Ordikhani 
chapter 2 :Graph Theory for Graph Neural Networks
book : Hands-On Graph Neural Networks Using Python
Practical techniques and architectures for building powerful graph and deep learning apps with PyTorch
Maxime Labonne

This script demonstrates the creation, visualization, and analysis of various types of graphs 
using the NetworkX library in Python.
"""

import networkx as nx
import matplotlib.pyplot as plt

# --- Creating and visualizing an undirected graph ---
# Create an undirected graph with edges connecting nodes
G = nx.Graph()
G.add_edges_from([
    ('A', 'B'),
    ('A', 'C'),
    ('B', 'D'),
    ('B', 'E'),
    ('C', 'F'),
    ('C', 'G')
])

# Draw the undirected graph
plt.figure(figsize=(8, 6))
nx.draw(G, with_labels=True, node_color='skyblue', node_size=700, edge_color='gray', font_size=12)
plt.title("Undirected Graph")
plt.show()

# --- Creating and visualizing a directed graph ---
# Create a directed graph with edges having a specific direction
DG = nx.DiGraph()
DG.add_edges_from([
    ('A', 'B'),
    ('A', 'C'),
    ('B', 'D'),
    ('B', 'E'),
    ('C', 'F'),
    ('C', 'G')
])

# Draw the directed graph
plt.figure(figsize=(8, 6))
nx.draw(DG, with_labels=True, node_color='lightgreen', node_size=700, edge_color='gray', font_size=12, arrowsize=20)
plt.title("Directed Graph")
plt.show()

# --- Creating and visualizing a weighted graph ---
# Create a graph with weights assigned to edges
WG = nx.Graph()
WG.add_edges_from([
    ('A', 'B', {"weight": 10}),
    ('A', 'C', {"weight": 20}),
    ('B', 'D', {"weight": 30}),
    ('B', 'E', {"weight": 40}),
    ('C', 'F', {"weight": 50}),
    ('C', 'G', {"weight": 60})
])

# Draw the weighted graph
pos = nx.spring_layout(WG)  # Determine node positions
plt.figure(figsize=(8, 6))
nx.draw(WG, pos, with_labels=True, node_color='lightcoral', node_size=700, edge_color='gray', font_size=12)
labels = nx.get_edge_attributes(WG, "weight")
nx.draw_networkx_edge_labels(WG, pos, edge_labels=labels)
plt.title("Weighted Graph")
plt.show()

# --- Checking connectivity of graphs ---
# Create two different graphs
G1 = nx.Graph()
G1.add_edges_from([(1, 2), (2, 3), (3, 1), (4, 5)])  # Disconnected graph

G2 = nx.Graph()
G2.add_edges_from([(1, 2), (2, 3), (3, 1), (1, 4)])  # Connected graph

# Check if graphs are connected
print(f"Is graph 1 connected? {nx.is_connected(G1)}")
print(f"Is graph 2 connected? {nx.is_connected(G2)}")

# --- Visualizing connected and disconnected graphs ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
nx.draw(G1, with_labels=True, ax=axes[0], node_color='gold', node_size=700)
axes[0].set_title(f"Graph 1 (Connected: {nx.is_connected(G1)})")

nx.draw(G2, with_labels=True, ax=axes[1], node_color='gold', node_size=700)
axes[1].set_title(f"Graph 2 (Connected: {nx.is_connected(G2)})")

plt.show()

# --- Analyzing node degrees ---
# Visualize the graph with node sizes proportional to their degrees
plt.figure(figsize=(8, 6))
degrees = dict(G.degree())
nx.draw(G, with_labels=True, node_color='lightblue', node_size=[v * 300 for v in degrees.values()], font_size=12)
plt.title("Graph with Node Degrees")
plt.show()

# Print degree information
print(f"deg(A) = {G.degree['A']}")  # Degree of node A in the undirected graph
print(f"deg^-(A) = {DG.in_degree['A']}")  # In-degree of node A in the directed graph
print(f"deg^+(A) = {DG.out_degree['A']}")  # Out-degree of node A in the directed graph

# --- Centrality measures ---
# Compute different centrality measures for the graph
degree_centrality = nx.degree_centrality(G)
closeness_centrality = nx.closeness_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)

# Visualize graph with node sizes based on degree centrality
plt.figure(figsize=(8, 6))
nx.draw(G, with_labels=True, node_color='lightblue', node_size=[v * 1000 for v in degree_centrality.values()], font_size=12)
plt.title("Graph (Node Size = Degree Centrality)")
plt.show()

# Print centrality measures
print(f"Degree centrality = {nx.degree_centrality(G)}")
print(f"Closeness centrality = {nx.closeness_centrality(G)}")
print(f"Betweenness centrality = {nx.betweenness_centrality(G)}")

# --- Graph adjacency matrix ---
# Visualize the adjacency matrix as an image
plt.figure(figsize=(8, 6))
nx.draw(G, with_labels=True, node_color='skyblue', node_size=700, font_size=12)
plt.title("Adjacency Matrix Graph")
plt.show()

# Print edge list representation
edge_list = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]
print("Edge list:", edge_list)

# --- BFS and DFS implementations ---
# BFS traversal function
def bfs(graph, node):
    visited, queue = [node], [node]
    while queue:
        node = queue.pop(0)
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.append(neighbor)
                queue.append(neighbor)
    return visited

# Run BFS and print result
result_bfs = bfs(G, 'A')
print("BFS Result:", result_bfs)  # Output: ['A', 'B', 'C', 'D', 'E', 'F', 'G']

# DFS traversal function
def dfs(visited, graph, node):
    if node not in visited:
        visited.append(node)
        for neighbor in graph[node]:
            dfs(visited, graph, neighbor)
    return visited

# Run DFS and print result
visited = []
result_dfs = dfs(visited, G, 'A')
print("DFS Result:", result_dfs)  # Output: ['A', 'B', 'D', 'E', 'C', 'F', 'G']

# --- Adjacency matrix ---
# Define a graph and compute its adjacency matrix
G = nx.Graph()
G.add_edges_from([
    ('A', 'B'),
    ('A', 'C'),
    ('B', 'D'),
    ('C', 'D'),
    ('D', 'E')
])

# Extract adjacency matrix
adj_matrix = nx.adjacency_matrix(G).todense()

# Display adjacency matrix
print("Adjacency Matrix:")
print(adj_matrix)

# Visualize graph
plt.figure(figsize=(8, 6))
nx.draw(G, with_labels=True, node_color='skyblue', node_size=700, font_size=12)
plt.title("Graph Visualization")
plt.show()

# Visualize adjacency matrix as an image
plt.figure(figsize=(6, 6))
plt.imshow(adj_matrix, cmap='Greys', interpolation='none')
plt.title("Adjacency Matrix (Visualized)")
plt.colorbar(label="Connection")
plt.xticks(range(len(G.nodes())), G.nodes(), rotation=90)
plt.yticks(range(len(G.nodes())), G.nodes())
plt.show()
