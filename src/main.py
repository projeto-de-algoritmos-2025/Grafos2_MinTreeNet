import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from collections import defaultdict
import heapq
import json


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True

class mensagemViaMst:
    def __init__(self):
        self.nodes = {}  
        self.edges = []  

    def add_node(self, node_id, name, position=None):
        """Adiciona um nó (cluster/servidor) ao grafo"""
        if position is None:
            # se for posicao aleatoria
            position = (np.random.uniform(0, 10), np.random.uniform(0, 10))
        self.nodes[node_id] = {'name': name, 'pos': position}

    def add_edge(self, node1, node2, weight=None):
        """Adiciona uma aresta com peso (custo de comunicação)"""
        if weight is None:
            
            pos1 = self.nodes[node1]['pos']
            pos2 = self.nodes[node2]['pos']
            weight = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

        self.edges.append((weight, node1, node2))

    def kruskal_mst(self):
        """Implementa o algoritmo de Kruskal para encontrar MST"""
        # ordena arestas por peso
        self.edges.sort()

        n = len(self.nodes)
        uf = UnionFind(max(self.nodes.keys()) + 1)
        mst_edges = []
        total_cost = 0

        for weight, u, v in self.edges:
            if uf.union(u, v):
                mst_edges.append((weight, u, v))
                total_cost += weight
                if len(mst_edges) == n - 1:
                    break

        return mst_edges, total_cost


    def visualize_mst(self):
        """Visualiza o grafo original e a MST"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        
        G_original = nx.Graph()
        for node_id, data in self.nodes.items():
            G_original.add_node(node_id, **data)

        for weight, u, v in self.edges:
            G_original.add_edge(u, v, weight=weight)

        pos = {node_id: data['pos'] for node_id, data in self.nodes.items()}
        labels = {node_id: data['name'] for node_id, data in self.nodes.items()}

        # grafo original
        nx.draw(G_original, pos, ax=ax1, with_labels=True, labels=labels,
                node_color='lightblue', node_size=1000, font_size=8)

        edge_labels = nx.get_edge_attributes(G_original, 'weight')
        edge_labels = {k: f'{v:.1f}' for k, v in edge_labels.items()}
        nx.draw_networkx_edge_labels(G_original, pos, edge_labels, ax=ax1, font_size=6)

        ax1.set_title('Grafo Original\n(Todas as conexões possíveis)')
        ax1.axis('equal')

        mst_edges, total_cost = self.kruskal_mst()
        

        G_mst = nx.Graph()
        for node_id, data in self.nodes.items():
            G_mst.add_node(node_id, **data)

        for weight, u, v in mst_edges:
            G_mst.add_edge(u, v, weight=weight)

        # desenho do MST
        nx.draw(G_mst, pos, ax=ax2, with_labels=True, labels=labels,
                node_color='lightgreen', node_size=1000, font_size=8,
                edge_color='red', width=2)

        mst_edge_labels = nx.get_edge_attributes(G_mst, 'weight')
        mst_edge_labels = {k: f'{v:.1f}' for k, v in mst_edge_labels.items()}
        nx.draw_networkx_edge_labels(G_mst, pos, mst_edge_labels, ax=ax2, font_size=6)

        ax2.set_title(f'Árvore Geradora Mínima (kruskal)\n'
                     f'Custo Total: {total_cost:.2f}')
        ax2.axis('equal')

        plt.tight_layout()
        plt.show()

        return mst_edges, total_cost

    def simulate_message_distribution(self, source_node, message="Olá clusters!"):
        """Simula a distribuição de mensagem pela MST"""
        mst_edges, _ = self.kruskal_mst()

        # Constrói árvore da MST
        tree = defaultdict(list)
        for _, u, v in mst_edges:
            tree[u].append(v)
            tree[v].append(u)

        # BFS para simular propagação da mensagem
        visited = set()
        queue = [(source_node, 0)]  # (nó, nível)
        distribution_order = []

        while queue:
            node, level = queue.pop(0)
            if node in visited:
                continue

            visited.add(node)
            distribution_order.append((node, self.nodes[node]['name'], level))

            for neighbor in tree[node]:
                if neighbor not in visited:
                    queue.append((neighbor, level + 1))

        return distribution_order



def carregar_clusters_json(caminho_arquivo):
    with open(caminho_arquivo, "r", encoding="utf-8") as f:
        clusters = json.load(f)
    return clusters

def exemplo_clusters():
    mst_system = mensagemViaMst()
    
    # Carrega clusters do JSON
    clusters = carregar_clusters_json("clusters.json")

    # Adiciona clusters
    for c in clusters:
        mst_system.add_node(c["id"], c["name"], tuple(c["pos"]))
    
    # Adiciona todas as conexões possíveis
    nodes = [c["id"] for c in clusters]
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            mst_system.add_edge(nodes[i], nodes[j])
    
    
    mst_edges, total_cost = mst_system.visualize_mst()
    
    mst_system.simulate_message_distribution(0, "Deploy da versão 2.1.0 iniciado!")
    return mst_system


# exemplo
if __name__ == "__main__":
    sistema = exemplo_clusters()
