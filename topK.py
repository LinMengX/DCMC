import torch
def get_listMLE_topK(result_matrix, topK):
    '''
        使用 PyTorch 张量操作计算 Top-K 节点的索引。

        @param result_matrix: PyTorch 张量，形状为 |V| x |V|，包含节点之间的相似度
        @param topK: 需要提取的最相似的 Top-K 节点数量

        @return: top_indices: |V| x topK 张量，包含每个节点对应的 Top-K 最相似节点的索引
    '''
    # 初始化一个大小为 (|V| x topK) 的张量，用于存储每个节点的 Top-K 节点索引
    top_indices = torch.zeros((result_matrix.shape[0], topK), dtype=torch.int32)

    # 对每个节点计算其 Top-K 相似节点的索引
    for target_node in range(result_matrix.shape[0]):
        # 获取当前节点的相似度值，并按降序排序，提取前 topK 的节点索引
        target_node_res_sorted = torch.argsort(result_matrix[target_node, :], descending=True)[:topK]

        # 将排序后的 Top-K 节点索引存入 top_indices
        top_indices[target_node] = target_node_res_sorted

    print("Top {} similar nodes are fetched ... ".format(topK - 1))

    return top_indices