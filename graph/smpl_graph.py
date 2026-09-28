# graph/smpl_graph.py

import numpy as np
from .tools import get_adjacency_matrix

class SMPLGraph:
    def __init__(self):
        self.num_node = 22
        self_link = [(i, i) for i in range(self.num_node)]
        neighbor_link = [
            (0, 1), (0, 2), (0, 3),
            (1, 4), (2, 5), (3, 6),
            (4, 7), (5, 8), (6, 9),
            (7, 10), (8, 11),
            (9, 13), (9, 14),
            (12, 15),
            (13, 16), (14, 17),
            (16, 18), (17, 19),
            (18, 20), (19, 21)
        ]
        self.edge = self_link + neighbor_link
        self.center = 0
        self.A_binary = get_adjacency_matrix(self.edge, self.num_node)


class KITMLGraph:
    def __init__(self):
        self.num_node = 21
        self_link = [(i, i) for i in range(self.num_node)]
        neighbor_link = [
            (0, 1), (0, 11), (0, 16),
            (1, 2), (2, 3),
            (3, 4), (3, 5), (3, 8),
            (5, 6), (6, 7),
            (8, 9), (9, 10),
            (11, 12), (12, 13), (13, 14), (14, 15),
            (16, 17), (17, 18), (18, 19), (19, 20)
        ]
        self.edge = self_link + neighbor_link
        self.center = 0
        self.A_binary = get_adjacency_matrix(self.edge, self.num_node)

class Graph:
    """NTU RGB+D 25-joint skeleton graph"""

    def __init__(self, labeling_mode='spatial'):
        self.num_node = 25
        self.self_link = [(i, i) for i in range(self.num_node)]
        # self.inward = [
        #     (0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4),
        #     (6, 7), (7, 6), (8, 20), (9, 8), (10, 9), (11, 10),
        #     (12, 0), (13, 12), (14, 13), (15, 14), (16, 0), (17, 16),
        #     (18, 17), (19, 18), (21, 22), (20, 20), (22, 7), (23, 24), (24, 11)
        # ]
        inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                            (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                            (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                            (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
        self.inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
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