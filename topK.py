import torch
def get_listMLE_topK(result_matrix, topK):
    top_indices = torch.zeros((result_matrix.shape[0], topK), dtype=torch.int32)

    for target_node in range(result_matrix.shape[0]):
        
        target_node_res_sorted = torch.argsort(result_matrix[target_node, :], descending=True)[:topK]

        top_indices[target_node] = target_node_res_sorted

    print("Top {} similar nodes are fetched ... ".format(topK - 1))

    return top_indices