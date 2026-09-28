# graph/ntu_rgb_d.py
"""
NTU RGB+D 25关节图结构
用于T²MR骨架编码器的图构建与NTU拓扑兼容。
"""

import numpy as np
from .tools import get_adjacency_matrix


class Graph:
    """NTU RGB+D 25-joint skeleton graph"""

    def __init__(self, labeling_mode='spatial'):
        self.num_node = 25
        self.self_link = [(i, i) for i in range(self.num_node)]
        self.inward = [
            (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
            (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
            (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
            (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25), (25, 12)
        ]
        self.outward = [(j, i) for (i, j) in self.inward]
        self.neighbor = self.inward + self.outward
        self.edge = self.self_link + self.neighbor
        self.center = 21  # spine center

        # 构建邻接矩阵
        self.A_binary = get_adjacency_matrix(self.edge, self.num_node)

        # 用于不同标记模式（spatial, temporal等）
        self.labeling_mode = labeling_mode
        if labeling_mode == 'spatial':
            self.A = self.get_spatial_graph()
        else:
            self.A = self.A_binary

    def get_spatial_graph(self):
        """获取空间图（包含不同距离的邻接关系）"""
        hop_dis = self.get_hop_distance(self.num_node, self.edge, max_hop=1)
        valid_hop = range(0, 2)  # 0-hop (self) and 1-hop (neighbor)

        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[hop_dis == hop] = 1

        return adjacency

    @staticmethod
    def get_hop_distance(num_node, edge, max_hop=1):
        """计算节点间的跳数距离"""
        A = np.zeros((num_node, num_node))
        for i, j in edge:
            A[j, i] = 1
            A[i, j] = 1

        # 使用BFS计算距离
        hop_dis = np.zeros((num_node, num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        for d in range(max_hop, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis


# 为了兼容旧代码，提供别名
class AdjMatrixGraph(Graph):
    """别名，指向Graph类"""
    pass


if __name__ == "__main__":
    # 测试
    graph = Graph()
    print(f"Number of nodes: {graph.num_node}")
    print(f"Adjacency matrix shape: {graph.A_binary.shape}")
    print(f"Number of edges: {len(graph.edge)}")
    print(f"Center node: {graph.center}")

    # 验证邻接矩阵对称性
    assert np.allclose(graph.A_binary, graph.A_binary.T), "Adjacency matrix should be symmetric"
    print("✓ Adjacency matrix is symmetric")

    # 验证自连接
    assert np.all(np.diag(graph.A_binary) == 1), "Diagonal should be all 1s (self-loops)"
    print("✓ Self-loops present")

    print("\n✓ NTU RGB+D graph structure verified!")