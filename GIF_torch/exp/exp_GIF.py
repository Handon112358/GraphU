# 数组形状和元素内容说明：
# edge_index: [2, num_edges] - 整数，表示每条边的起始节点和终止节点
# unique_indices: [num_unique_edges] - 整数，表示满足 edge_index[0] < edge_index[1] 条件的边的索引
# unique_indices_not: [num_non_unique_edges] - 整数，表示满足 edge_index[0] > edge_index[1] 条件的边的索引
# unique_edge_index: [2, num_unique_edges] - 整数，表示唯一边的起始节点和终止节点
# delete_edge_indices: [num_unique_edges] - 布尔值，表示哪些唯一边需要删除
# remain_indices: [num_unique_edges] 或 [num_remaining_edges] - 布尔值或整数，表示哪些边未被删除
# remain_encode: [num_remaining_edges] - 整数，表示编码后的剩余边索引
# unique_encode_not: [num_non_unique_edges] - 整数，表示编码后的非唯一边索引
# sort_indices: [num_non_unique_edges] - 整数，表示排序后的非唯一边索引
# remain_indices_not: [num_remaining_non_unique_edges] - 整数，表示未被删除的非唯一边的索引
# remain_indices: [num_remaining_edges] - 整数，表示所有未被删除的边的索引（包括唯一边和非唯一边）
# delete_nodes: [num_delete_nodes] - 整数，表示需要删除的节点
# delete_edge_index: [num_delete_edges] - 整数，表示需要删除的边的索引
# self.train_indices: [num_train_nodes] - 整数，表示训练数据集中节点的索引
# self.test_indices: [num_test_nodes] - 整数，表示测试数据集中节点的索引
# self.data.x_unlearn: [num_nodes, num_features] - 浮点数，表示节点的特征数据
# self.data.edge_index_unlearn: [2, num_edges] - 整数，表示去学习后的边索引
# unique_nodes: [num_unique_nodes] - 整数，表示唯一节点的索引
# remove_indices: [num_remove_edges] - 整数，表示需要移除的边的索引
# remove_edges: [2, num_remove_edges] - 整数，表示需要移除的边的起始节点和终止节点
# influenced_nodes: [num_influenced_nodes] - 整数，表示受影响的节点
# neighbor_nodes: [num_neighbor_nodes] - 整数，表示邻居节点

from cgi import test  # 导入cgi模块中的test
import logging  # 导入日志模块
import time  # 导入时间模块
import os  # 导入操作系统模块

import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch的神经网络模块
import torch.nn.functional as F  # 导入PyTorch的功能模块
from torch.autograd import grad  # 从PyTorch的自动求导模块导入grad
import networkx as nx  # 导入NetworkX库
from sklearn.model_selection import train_test_split  # 从sklearn导入训练测试分割函数
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score  # 从sklearn导入评估指标

torch.cuda.empty_cache()  # 清空CUDA缓存
import torch_geometric.transforms as T  # 导入PyTorch Geometric的变换模块
from torch_geometric.datasets import Planetoid  # 导入PyTorch Geometric的数据集
from torch_geometric.data import NeighborSampler  # 导入PyTorch Geometric的邻居采样器
from torch_geometric.nn.conv.gcn_conv import gcn_norm  # 导入GCN的归一化函数
import numpy as np  # 导入NumPy库

from exp.exp import Exp  # 从exp模块导入Exp类
from lib_gnn_model.gat.gat_net_batch import GATNet  # 导入GAT网络
from lib_gnn_model.gin.gin_net_batch import GINNet  # 导入GIN网络
from lib_gnn_model.gcn.gcn_net_batch import GCNNet  # 导入GCN网络
from lib_gnn_model.graphsage.graphsage_net import SageNet  # 导入GraphSAGE网络
# from torch_geometric.nn.models import GraphSAGE  # 导入GraphSAGE模型（已注释）
from lib_gnn_model.sgc.sgc_net_batch import SGCNet  # 导入SGC网络
from lib_gnn_model.node_classifier import NodeClassifier  # 导入节点分类器
from lib_gnn_model.gnn_base import GNNBase  # 导入GNN基础类
from parameter_parser import parameter_parser  # 导入参数解析器
from lib_utils import utils  # 导入实用工具


class ExpGraphInfluenceFunction(Exp):  # 定义ExpGraphInfluenceFunction类，继承自Exp
    def __init__(self, args):  # 初始化
        super(ExpGraphInfluenceFunction, self).__init__(args)  # 调用父类的初始化

        self.logger = logging.getLogger('ExpGraphInfluenceFunction')  # 创建日志记录器
        self.deleted_nodes = np.array([])  # 初始化删除节点数组
        self.feature_nodes = np.array([])  # 初始化特征节点数组
        self.influence_nodes = np.array([])  # 初始化影响节点数组

        self.load_data()  # 加载数据
        self.num_feats = self.data.num_features  # 获取特征数量
        self.train_test_split()  # 进行训练测试分割
        self.unlearning_request()  # 处理去学习请求

        self.target_model_name = self.args['target_model']  # 获取目标模型名称

        # self.get_edge_indeces()  # 获取边索引（已注释）
        self.determine_target_model()  # 确定目标模型

        run_f1 = np.empty((0))  # 初始化F1分数数组
        run_f1_unlearning = np.empty((0))  # 初始化去学习F1分数数组
        unlearning_times = np.empty((0))  # 初始化去学习时间数组
        training_times = np.empty((0))  # 初始化训练时间数组
        run_accuracy = np.empty((0))  # 初始化准确率数组
        run_accuracy_unlearning = np.empty((0))  # 初始化去学习准确率数组
        run_recall = np.empty((0))  # 初始化召回率数组
        run_recall_unlearning = np.empty((0))  # 初始化去学习召回率数组
        run_precision = np.empty((0))  # 初始化精确率数组
        run_precision_unlearning = np.empty((0))  # 初始化去学习精确率数组

        for run in range(self.args['num_runs']):  # 遍历每次运行
            self.logger.info("Run %f" % run)  # 记录运行信息

            run_training_time, result_tuple = self._train_model(run)  # 训练模型并获取结果

            f1_score, accuracy, recall, precision = self.evaluate(run)  # 评估模型
            run_f1 = np.append(run_f1, f1_score)  # 记录F1分数
            run_accuracy = np.append(run_accuracy, accuracy)  # 记录准确率
            run_recall = np.append(run_recall, recall)  # 记录召回率
            run_precision = np.append(run_precision, precision)  # 记录精确率
            training_times = np.append(training_times, run_training_time)  # 记录训练时间

            # 使用GIF进行unlearning
            unlearning_time, f1_score_unlearning, accuracy_unlearning, recall_unlearning, precision_unlearning = self.gif_approxi(result_tuple)
            unlearning_times = np.append(unlearning_times, unlearning_time)  # 记录去学习时间
            run_f1_unlearning = np.append(run_f1_unlearning, f1_score_unlearning)  # 记录去学习F1分数
            run_accuracy_unlearning = np.append(run_accuracy_unlearning, accuracy_unlearning)  # 记录去学习准确率
            run_recall_unlearning = np.append(run_recall_unlearning, recall_unlearning)  # 记录去学习召回率
            run_precision_unlearning = np.append(run_precision_unlearning, precision_unlearning)  # 记录去学习精确率

        f1_score_avg = np.average(run_f1)  # 计算F1分数平均值
        f1_score_std = np.std(run_f1)  # 计算F1分数标准差
        accuracy_avg = np.average(run_accuracy)  # 计算准确率平均值
        accuracy_std = np.std(run_accuracy)  # 计算准确率标准差
        recall_avg = np.average(run_recall)  # 计算召回率平均值
        recall_std = np.std(run_recall)  # 计算召回率标准差
        precision_avg = np.average(run_precision)  # 计算精确率平均值
        precision_std = np.std(run_precision)  # 计算精确率标准差
        self.logger.info("f1_score: avg=%s, std=%s" % (f1_score_avg, f1_score_std))  # 记录F1分数信息
        self.logger.info("accuracy: avg=%s, std=%s" % (accuracy_avg, accuracy_std))  # 记录准确率信息
        self.logger.info("recall: avg=%s, std=%s" % (recall_avg, recall_std))  # 记录召回率信息
        self.logger.info("precision: avg=%s, std=%s" % (precision_avg, precision_std))  # 记录精确率信息
        self.logger.info("model training time: avg=%s seconds" % np.average(training_times))  # 记录模型训练时间信息

        f1_score_unlearning_avg = np.average(run_f1_unlearning)  # 计算去学习F1分数平均值
        f1_score_unlearning_std = np.std(run_f1_unlearning)  # 计算去学习F1分数标准差
        accuracy_unlearning_avg = np.average(run_accuracy_unlearning)  # 计算去学习准确率平均值
        accuracy_unlearning_std = np.std(run_accuracy_unlearning)  # 计算去学习准确率标准差
        recall_unlearning_avg = np.average(run_recall_unlearning)  # 计算去学习召回率平均值
        recall_unlearning_std = np.std(run_recall_unlearning)  # 计算去学习召回率标准差
        precision_unlearning_avg = np.average(run_precision_unlearning)  # 计算去学习精确率平均值
        precision_unlearning_std = np.std(run_precision_unlearning)  # 计算去学习精确率标准差
        unlearning_time_avg = np.average(unlearning_times)  # 计算去学习时间平均值
        self.logger.info("f1_score of GIF: avg=%s, std=%s" % (f1_score_unlearning_avg, f1_score_unlearning_std))  # 记录去学习F1分数信息
        self.logger.info("accuracy of GIF: avg=%s, std=%s" % (accuracy_unlearning_avg, accuracy_unlearning_std))  # 记录去学习准确率信息
        self.logger.info("recall of GIF: avg=%s, std=%s" % (recall_unlearning_avg, recall_unlearning_std))  # 记录去学习召回率信息
        self.logger.info("precision of GIF: avg=%s, std=%s" % (precision_unlearning_avg, precision_unlearning_std))  # 记录去学习精确率信息
        self.logger.info("GIF unlearning time: avg=%s seconds" % np.average(unlearning_time_avg))  # 记录去学习时间信息

    def load_data(self):  # 加载数据方法
        self.data = self.data_store.load_raw_data()  # 从数据存储中加载原始数据

    def train_test_split(self):  # 训练测试分割方法
        if self.args['is_split']:  # 如果需要分割
            self.logger.info('splitting train/test data')  # 记录分割信息
            # 使用数据集的默认分割
            if self.data.name in ['ogbn-arxiv', 'ogbn-products']:  # 如果数据集名称在指定列表中
                self.train_indices, self.test_indices = self.data.train_indices.numpy(), self.data.test_indices.numpy()  # 获取训练和测试索引
            else:
                self.train_indices, self.test_indices = train_test_split(np.arange((self.data.num_nodes)), test_size=self.args['test_ratio'], random_state=100)  # 使用sklearn进行分割
                
            self.data_store.save_train_test_split(self.train_indices, self.test_indices)  # 保存分割结果

            self.data.train_mask = torch.from_numpy(np.isin(np.arange(self.data.num_nodes), self.train_indices))  # 创建训练掩码
            self.data.test_mask = torch.from_numpy(np.isin(np.arange(self.data.num_nodes), self.test_indices))  # 创建测试掩码
        else:
            self.train_indices, self.test_indices = self.data_store.load_train_test_split()  # 加载训练和测试索引

            self.data.train_mask = torch.from_numpy(np.isin(np.arange(self.data.num_nodes), self.train_indices))  # 创建训练掩码
            self.data.test_mask = torch.from_numpy(np.isin(np.arange(self.data.num_nodes), self.test_indices))  # 创建测试掩码

    def unlearning_request(self):  
        self.logger.debug("Train data  #.Nodes: %f, #.Edges: %f" % (
            self.data.num_nodes, self.data.num_edges))  # 记录训练数据的节点和边数量

        self.data.x_unlearn = self.data.x.clone()  # 克隆特征数据
        self.data.edge_index_unlearn = self.data.edge_index.clone()  # 克隆边索引
        edge_index = self.data.edge_index.numpy()  # 获取边索引的numpy数组
        unique_indices = np.where(edge_index[0] < edge_index[1])[0]  # 获取唯一边索引

        if self.args["unlearn_task"] == 'node':  # 如果去学习任务是节点
            unique_nodes = np.random.choice(len(self.train_indices),
                                            int(len(self.train_indices) * self.args['unlearn_ratio']),
                                            replace=False)  # 随机选择节点
            self.data.edge_index_unlearn = self.update_edge_index_unlearn(unique_nodes)  # 更新边索引

        if self.args["unlearn_task"] == 'edge':  # 如果去学习任务是边
            remove_indices = np.random.choice(
                unique_indices,
                int(unique_indices.shape[0] * self.args['unlearn_ratio']),
                replace=False)  # 随机选择边索引
            remove_edges = edge_index[:, remove_indices]  # 获取要移除的边
            unique_nodes = np.unique(remove_edges)  # 获取唯一节点
        
            self.data.edge_index_unlearn = self.update_edge_index_unlearn(unique_nodes, remove_indices)  # 更新边索引

        if self.args["unlearn_task"] == 'feature':  # 如果去学习任务是特征
            unique_nodes = np.random.choice(len(self.train_indices),
                                            int(len(self.train_indices) * self.args['unlearn_ratio']),
                                            replace=False)  # 随机选择节点
            self.data.x_unlearn[unique_nodes] = 0.  # 将选中节点的特征置为0
        self.find_k_hops(unique_nodes)  # 查找k跳邻居

    def update_edge_index_unlearn(self, delete_nodes, delete_edge_index=None):  # 更新边索引方法
        edge_index = self.data.edge_index.numpy()  # 获取边索引的numpy数组

        unique_indices = np.where(edge_index[0] < edge_index[1])[0]  # 获取唯一边索引
        unique_indices_not = np.where(edge_index[0] > edge_index[1])[0]  # 获取非唯一边索引

        if self.args["unlearn_task"] == 'edge':  # 如果去学习任务是边
            remain_indices = np.setdiff1d(unique_indices, delete_edge_index)  # 获取剩余边索引
        else:
            unique_edge_index = edge_index[:, unique_indices]  # 获取唯一边索引
            delete_edge_indices = np.logical_or(np.isin(unique_edge_index[0], delete_nodes),
                                                np.isin(unique_edge_index[1], delete_nodes))  # 获取要删除的边索引
            remain_indices = np.logical_not(delete_edge_indices)  # 获取剩余边索引
            remain_indices = np.where(remain_indices == True)  # 获取剩余边索引的位置

        remain_encode = edge_index[0, remain_indices] * edge_index.shape[1] * 2 + edge_index[1, remain_indices]  # 编码剩余边索引
        unique_encode_not = edge_index[1, unique_indices_not] * edge_index.shape[1] * 2 + edge_index[0, unique_indices_not]  # 编码非唯一边索引
        sort_indices = np.argsort(unique_encode_not)  # 对非唯一边索引进行排序
        remain_indices_not = unique_indices_not[sort_indices[np.searchsorted(unique_encode_not, remain_encode, sorter=sort_indices)]]  # 获取剩余非唯一边索引
        remain_indices = np.union1d(remain_indices, remain_indices_not)  # 合并剩余边索引

        return torch.from_numpy(edge_index[:, remain_indices])  # 返回更新后的边索引

    def determine_target_model(self):  # 确定目标模型方法
        self.logger.info('target model: %s' % (self.args['target_model'],))  # 记录目标模型信息
        num_classes = len(self.data.y.unique())  # 获取类别数量

        self.target_model = NodeClassifier(self.num_feats, num_classes, self.args)  # 创建节点分类器

    def evaluate(self, run):  # 评估方法
        self.logger.info('model evaluation')  # 记录评估信息

        start_time = time.time()  # 记录开始时间
        posterior = self.target_model.posterior()  # 获取后验概率
        y_true = self.data.y[self.data['test_mask']].cpu().numpy()  # 获取真实标签
        y_pred = posterior.argmax(axis=1).cpu().numpy()  # 获取预测标签
        test_f1 = f1_score(y_true, y_pred, average="micro")  # 计算F1分数
        accuracy = accuracy_score(y_true, y_pred)  # 计算准确率
        recall = recall_score(y_true, y_pred, average="micro")  # 计算召回率
        precision = precision_score(y_true, y_pred, average="micro")  # 计算精确率

        evaluate_time = time.time() - start_time  # 计算评估时间
        self.logger.info("Evaluation cost %s seconds." % evaluate_time)  # 记录评估时间

        self.logger.info("Final Test F1: %s" % (test_f1,))  # 记录F1分数
        self.logger.info("Final Test Accuracy: %s" % (accuracy,))  # 记录准确率
        self.logger.info("Final Test Recall: %s" % (recall,))  # 记录召回率
        self.logger.info("Final Test Precision: %s" % (precision,))  # 记录精确率
        return test_f1, accuracy, recall, precision  # 返回评估结果

    def _train_model(self, run):  # 训练模型方法
        self.logger.info('training target models, run %s' % run)  # 记录训练信息

        start_time = time.time()  # 记录开始时间
        self.target_model.data = self.data  # 设置模型数据
        res = self.target_model.train_model(
            (self.deleted_nodes, self.feature_nodes, self.influence_nodes))  # 训练模型并获取结果
        train_time = time.time() - start_time  # 计算训练时间

        # self.data_store.save_target_model(run, self.target_model)  # 保存目标模型（已注释）
        self.logger.info("Model training time: %s" % (train_time))  # 记录训练时间

        return train_time, res  # 返回训练时间和结果
        
    def find_k_hops(self, unique_nodes):  # 查找k跳邻居方法
        edge_index = self.data.edge_index.numpy()  # 获取边索引的numpy数组
        
        ## 查找影响的邻居
        hops = 2  # 设置跳数
        if self.args["unlearn_task"] == 'node':  # 如果去学习任务是节点
            hops = 3  # 设置跳数为3
        influenced_nodes = unique_nodes  # 初始化影响节点
        for _ in range(hops):  # 遍历跳数
            target_nodes_location = np.isin(edge_index[0], influenced_nodes)  # 获取目标节点位置
            neighbor_nodes = edge_index[1, target_nodes_location]  # 获取邻居节点
            influenced_nodes = np.append(influenced_nodes, neighbor_nodes)  # 更新影响节点
            influenced_nodes = np.unique(influenced_nodes)  # 获取唯一影响节点（合并重复的index）
        neighbor_nodes = np.setdiff1d(influenced_nodes, unique_nodes)  # 获取邻居节点
        if self.args["unlearn_task"] == 'feature':  # 如果去学习任务是特征
            self.feature_nodes = unique_nodes  # 设置特征节点
            self.influence_nodes = neighbor_nodes  # 设置影响节点
        if self.args["unlearn_task"] == 'node':  # 如果去学习任务是节点
            self.deleted_nodes = unique_nodes  # 设置删除节点
            self.influence_nodes = neighbor_nodes  # 设置影响节点
        if self.args["unlearn_task"] == 'edge':  # 如果去学习任务是边
            self.influence_nodes = influenced_nodes  # 设置影响节点

    def gif_approxi(self, res_tuple):  # GIF近似方法
        '''
        res_tuple == (grad_all, grad1, grad2)  # 全部梯度，保留梯度，删除梯度
        '''
        # result_tuple = self.target_model.train_model((self.deleted_nodes, self.feature_nodes, self.influence_nodes))
        start_time = time.time()  # 记录开始时间
        iteration, damp, scale = self.args['iteration'], self.args['damp'], self.args['scale']  # 获取参数

        if self.args["method"] == "GIF":  # 如果方法是GIF
            v = tuple(grad1 - grad2 for grad1, grad2 in zip(res_tuple[1], res_tuple[2]))  # 计算v
        if self.args["method"] == "IF":  # 如果方法是IF
            v = res_tuple[1]  # 设置v
        h_estimate = tuple(grad1 - grad2 for grad1, grad2 in zip(res_tuple[1], res_tuple[2]))  # 计算h估计
        for _ in range(iteration):  # 遍历迭代次数
            model_params = [p for p in self.target_model.model.parameters() if p.requires_grad]  # 获取模型参数
            hv = self.hvps(res_tuple[0], model_params, h_estimate)  # 计算HVPs
            with torch.no_grad():  # 不计算梯度
                h_estimate = [v1 + (1 - damp) * h_estimate1 - hv1 / scale
                              for v1, h_estimate1, hv1 in zip(v, h_estimate, hv)]  # 更新h估计

        params_change = [h_est / scale for h_est in h_estimate]  # 计算参数变化
        params_esti = [p1 + p2 for p1, p2 in zip(params_change, model_params)]  # 估计参数

        test_F1 = self.target_model.evaluate_unlearn_F1(params_esti)  # 评估去学习F1分数
        y_true = self.data.y[self.data['test_mask']].cpu().numpy()  # 获取真实标签
        y_pred = self.target_model.posterior().argmax(axis=1).cpu().numpy()  # 获取预测标签

        accuracy = accuracy_score(y_true, y_pred)  # 计算准确率
        recall = recall_score(y_true, y_pred, average="micro")  # 计算召回率
        precision = precision_score(y_true, y_pred, average="micro")  # 计算精确率

        return time.time() - start_time, test_F1, accuracy, recall, precision  # 返回结果

    def hvps(self, grad_all, model_params, h_estimate):  # 计算HVPs方法
        element_product = 0  # 初始化元素乘积
        for grad_elem, v_elem in zip(grad_all, h_estimate):  # 遍历梯度和h估计
            element_product += torch.sum(grad_elem * v_elem)  # 计算元素乘积
        
        return_grads = grad(element_product, model_params, create_graph=True)  # 计算返回梯度
        return return_grads  # 返回梯度