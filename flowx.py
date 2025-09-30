import copy
import os
from warnings import simplefilter
from dig.xgraph.method import FlowX
from dataset import get_dataset
from utils import check_dirs, get_logger
import torch
torch.cuda.empty_cache()
import hydra
from omegaconf import OmegaConf
import time
from gnnNets import get_gnnNets
from torch_geometric.utils import add_remaining_self_loops, k_hop_subgraph
IS_FRESH = False
import numpy as np
import torch
import copy
import time
import os
from torch_geometric.utils import k_hop_subgraph, add_remaining_self_loops
import torch_geometric.data

def pipeline(config, dataset, test_indices, model, device, logger):
    model.eval()
    model.to(device)
    test_indices = torch.tensor(test_indices)
    config.models.param.add_self_loop = False

    if torch.cuda.is_available():
        device = torch.device('cuda', index=config.device_id)
    else:
        device = torch.device('cpu')

    # --- 修改：将 GNNExplainer 实例化为 FlowX ---
    # FlowX 不需要 coff_size 和 coff_ent 参数，因此移除它们
    gnn_explainer = FlowX(model,
                          epochs=config.explainers.param[config.datasets.dataset_name].epochs,
                          lr=config.explainers.param[config.datasets.dataset_name].lr,
                          explain_graph=config.models.param.graph_classification)
    gnn_explainer.device = device

    if config.models.param.graph_classification:
        pass

    else:
        data = copy.deepcopy(dataset.data)
        data.to(device)

        with torch.no_grad():
            original_logits = model(data)
        original_probs = torch.softmax(original_logits, dim=-1)
        prediction = original_logits.argmax(-1)

        # --- 初始化用于累加 fidelity 的变量 ---
        total_fidelity_plus = 0.0
        total_fidelity_minus = 0.0
        total_time = 0.0
        all_explanation_edges = []
        all_explanation_nodes = []
        edge_index_with_loop = add_remaining_self_loops(data.edge_index)[0]
        for node_idx in test_indices:
            print(f"ground truth label: {data.y[node_idx].item()}")
            

            logger.info(f'test node: {node_idx.item()}')
            start_time = time.time()
            num_hops = len(config.models.param.gnn_latent_dim)
            subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
                node_idx.item(),
                num_hops,
                data.edge_index,
                relabel_nodes=True # 关键：对节点重编号，使子图独立
            )
            
            # 找到原始 node_idx 在子图中的新索引
            node_idx_local = mapping.item()
            
            # 创建一个只包含子图信息的新的 Data 对象

            sub_data = torch_geometric.data.Data(
                x=data.x[subset],
                edge_index=sub_edge_index
            ).to(device)

            sparsity = config.explainers.sparsity
            # 我们将 masks 命名为 edge_masks 以匹配后续代码
            walks, edge_masks, related_preds = \
                gnn_explainer(sub_data.x, sub_data.edge_index,
                              node_idx=node_idx_local, # 使用本地索引
                              sparsity=sparsity,
                              num_classes=dataset.num_classes)

            # --- 新增：手动从软掩码生成硬掩码 ---
            # FlowX 返回的是软掩码 (soft mask)，需要我们根据 sparsity 手动创建硬掩码
            hard_edge_masks = []
            for soft_mask in edge_masks:
                # 根据 sparsity 计算需要保留的边的数量
                num_edges = soft_mask.numel()
                # 假设 sparsity 是要保留的边的比例 (例如 0.7 表示保留最重要的 70% 的边)
                # GNNExplainer 通常用 size_coff 控制稀疏度，这里我们直接用 sparsity
                num_keep = int(num_edges * sparsity)
                
                # 获取分数最高的边的索引
                top_k_indices = torch.topk(soft_mask, k=num_keep).indices
                
                # 创建一个布尔类型的硬掩码
                hard_mask = torch.zeros_like(soft_mask, dtype=torch.bool)
                hard_mask[top_k_indices] = True
                hard_edge_masks.append(hard_mask)
            # --- 结束新增 ---

            original_pred_class = prediction[node_idx]
            original_prob_for_node = original_probs[node_idx, original_pred_class]

            # 1. 获取最重要的边的硬掩码 (hard mask)
            # hard_edge_masks 包含每个类别的解释，我们通常关注第一个
            hard_mask = hard_edge_masks[0]

            # 2. 计算解释子图中的边的数量
            num_explanation_edges = hard_mask.sum().item()
            
            sub_edge_index_with_loop, _ = add_remaining_self_loops(
                sub_data.edge_index, num_nodes=sub_data.num_nodes
            )
            # 3. 获取解释子图的边索引
            explanation_edge_index = sub_edge_index_with_loop[:, hard_mask] # 直接使用布尔掩码索引
            
            # 4. 计算解释子图中的节点的数量
            num_explanation_nodes = torch.unique(explanation_edge_index).numel()

            # 5. 存储结果
            all_explanation_edges.append(num_explanation_edges)
            all_explanation_nodes.append(num_explanation_nodes)
            
            # --- Calculate Fidelity+ ---
            # FlowX 的 related_preds 结构与 GNNExplainer 兼容
            masked_prob_for_node = related_preds[0]['masked']
            fidelity_plus = original_prob_for_node.item() - masked_prob_for_node

            # --- 手动计算 Fidelity- ---
            # hard_mask 是我们刚刚生成的，可以直接使用
            complement_edge_index = sub_edge_index_with_loop[:, ~hard_mask] # 使用布尔非操作
            with torch.no_grad():
                compl_logits = model(x=data.x, edge_index=complement_edge_index)
            compl_probs = torch.softmax(compl_logits, dim=-1)
            compl_prob_for_node = compl_probs[node_idx, original_pred_class]
            fidelity_minus = original_prob_for_node.item() - compl_prob_for_node.item()

            # --- Log metrics ---
            logger.info(f'fidelity+: {fidelity_plus:.4f}')
            logger.info(f'fidelity-: {fidelity_minus:.4f}')
            total_fidelity_plus += fidelity_plus
            total_fidelity_minus += fidelity_minus
            
            edge_masks_to_save = [edge_mask.to('cpu') for edge_mask in edge_masks] # <<< 修改：重命名变量以避免冲突
            explanation_saving_dir = os.path.join(config.explainers.explanation_result_dir,
                                                  config.datasets.dataset_name,
                                                  config.models.gnn_name)
            check_dirs(explanation_saving_dir)
            torch.save(edge_masks_to_save, os.path.join(explanation_saving_dir, # <<< 修改：保存重命名后的变量
                                                    f'{len(config.models.param.gnn_latent_dim)}_example_{node_idx.item()}.pt'))

            end_time = time.time()
            logger.info(f'Execution time for node_run: {end_time - start_time:.6f}')
            total_time += end_time - start_time
            
        num_test_nodes = len(test_indices)
        if num_test_nodes > 0:
            avg_fidelity_plus = total_fidelity_plus / num_test_nodes
            avg_fidelity_minus = total_fidelity_minus / num_test_nodes

            logger.info("=" * 30)
            logger.info(f"Final Average Fidelity+ over {num_test_nodes} nodes: {avg_fidelity_plus:.4f}")
            logger.info(f"Final Average Fidelity- over {num_test_nodes} nodes: {avg_fidelity_minus:.4f}")
            logger.info(f"total_time over {num_test_nodes} nodes: {total_time:.4f}")
            logger.info("=" * 30)
        else:
            logger.info("No test nodes were processed, cannot compute average fidelity.")
            
        avg_edges = np.mean(all_explanation_edges)
        avg_nodes = np.mean(all_explanation_nodes)
        print("\n--- 整体解释子图大小统计 ---")
        print(f"处理的节点总数: {len(test_indices)}")
        print(f"平均边的数量: {avg_edges:.2f}")
        print(f"平均节点数量: {avg_nodes:.2f}")
        print("---------------------------------")
        
@hydra.main(version_base=None, config_path='config', config_name='config')
def main(config):
    config.models.param = config.models.param[config.datasets.dataset_name]
    if config.datasets.dataset_name in ['CS', 'Physics', 'Facebook']:
        dataset, train_mask, valid_mask, test_mask = get_dataset(config.datasets.dataset_root,
                                                                 config.datasets.dataset_name)
    else:
        dataset = get_dataset(config.datasets.dataset_root, config.datasets.dataset_name)
    if dataset.data.x is not None:
        dataset.data.x = dataset.data.x.float()
    if config.datasets.dataset_name in ['products']:
        dataset.data.y = torch.argmax(dataset.data.y, dim=1)
    dataset.data.y = dataset.data.y.squeeze().long()

    log_file = (
        f"Gnnexplainer_{config.datasets.dataset_name}_{len(config.models.param.gnn_latent_dim)}.log"
    )
    logger = get_logger(config.log_path, log_file, config.console_log, config.log_level)
    logger.info(OmegaConf.to_yaml(config))
    if torch.cuda.is_available():
        device = torch.device('cuda', index=config.device_id)
    else:
        device = torch.device('cpu')
    logger.info(f'Using device: {device}')
    logger.info(f'Using data: {dataset.data}')

    config.models.gnn_savedir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    state_dict = torch.load(os.path.join(config.models.gnn_savedir,
                                         config.datasets.dataset_name,
                                         f'{config.models.gnn_name}_'
                                         f'{len(config.models.param.gnn_latent_dim)}l_best.pth'))['net']
    model = get_gnnNets(dataset.num_node_features, dataset.num_classes, config.models)
    print(model)
    model.load_state_dict(state_dict)
    if config.datasets.dataset_name in ['tree_grid', 'tree_cycle']:
        test_indices = torch.where(dataset[0].test_mask * dataset[0].y != 0)[0].tolist()
    # 新增一个 elif 分支来处理 CS, Physics, Facebook 这类数据集
    elif config.datasets.dataset_name in ['CS', 'Physics', 'Facebook']:
        # 直接使用从 get_dataset 返回的 test_mask 变量
        test_indices = torch.where(test_mask)[0].tolist()
    else:
        # 保持对其他数据集的原始处理方式
        test_indices = torch.where(dataset.data.test_mask)[0].tolist()
    num_samples = 100  # <-- 您可以在这里修改想采样的数量
    import random
    # 3. 根据总节点数和采样数决定最终的测试节点
    if len(test_indices) > num_samples:
        logger.info(f"从 {len(test_indices)} 个可用测试节点中随机采样 {num_samples} 个。")
        test_indices = random.sample(test_indices, num_samples)
    else:
        # 如果测试节点总数少于或等于采样数，就用全部节点
        logger.info(f"使用全部 {len(test_indices)} 个可用测试节点 (因为总数小于等于采样数 {num_samples})。")
        test_indices = test_indices
    logger.info(f'test nodes : {test_indices}')
    pipeline(config, dataset, test_indices, model, device, logger)

if __name__ == '__main__':
    import sys
    simplefilter(action="ignore", category=FutureWarning)
    sys.argv.append('explainers=flowx')
    sys.argv.append(f"datasets.dataset_root={os.path.join(os.path.dirname(__file__), 'datasets')}")
    sys.argv.append(f"models.gnn_saving_dir={os.path.join(os.path.dirname(__file__), 'checkpoints')}")
    main()
