# -*- coding: utf-8 -*-
import numpy as np
import jieba
import copy
from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

import matplotlib.pyplot as plt

def fusionVisual(data, predicted_labels):
    # 可视化聚类结果
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    # 使用PCA降维到2维
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(data)

    # 绘制散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=predicted_labels, cmap='viridis')
    plt.colorbar()
    plt.title('KNN Text Clustering')
    plt.savefig('result_fig/pca_visual.png')

class Engine():
    def __init__(self, config):
        self.config = config
    
    def sentence_vector(self,sentence,):
        words = list(jieba.cut(sentence))
        # 初始化句子向量为全零向量
        sentence_vec = np.zeros(self.model.vector_size)
        word_count = 0
        for word in words:
            if word in self.model:
                sentence_vec += self.model[word] ##
                word_count += 1
        if word_count > 0:
            sentence_vec /= word_count
        return sentence_vec

    def data_fusion(self, title_list, embedding_list, fusion_type='K-means', draw=False):
        embedding_array = np.array(embedding_list)
        if fusion_type == 'K-means':
            # Kmeans聚类
            kmeans = KMeans(n_clusters=self.config['num_clusters'], init='k-means++', max_iter=100, n_init=1, verbose=0) # 设定聚类数量
            kmeans.fit(embedding_array)
            # 获取每个标题所属的簇
            cluster_labels = kmeans.labels_
            # 创建一个字典来存储每个簇中的标题
            clustered_titles = {i: [] for i in range(self.config['num_clusters'])}

            # 根据聚类标签分配标题到相应的簇
            for i, label in enumerate(cluster_labels):
                clustered_titles[label].append(title_list[i])
            return clustered_titles
            '''
            df_titles = pd.DataFrame.from_dict(clustered_titles, orient='index').transpose()
            df_titles.to_csv('dataFusion_result.csv', header=False, index=False)
            print('数据融合结果已保存到文件：dataFusion_result.csv')
            
            if draw:
                fusionVisual(embedding_array, cluster_labels)
            '''    
        
        elif fusion_type == 'DBSCAN':
            # 数据通常需要标准化
            X = StandardScaler().fit_transform(embedding_array)
            
            # 选择合适的eps（邻域半径）和min_samples（邻域内的最小样本数）参数
            db = DBSCAN(eps=0.3, min_samples=10).fit(X)
            
            # labels数组将包含每个点的聚类标签, 噪声点的标签为-1
            labels = db.labels_
            
            # 检查聚类数量以及是否有噪声点
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise_ = list(labels).count(-1)
            print(f'Estimated number of clusters: {n_clusters_}')
            print(f'Estimated number of noise points: {n_noise_}')

            # 使用matplotlib来可视化数据点和它们所属的聚类
            
            unique_labels = set(labels)
            colors = [plt.cm.Spectral(each)
                    for each in np.linspace(0, 1, len(unique_labels))]
            for k, col in zip(unique_labels, colors):
                if k == -1:
                    # 黑色用来标记噪声
                    col = [0, 0, 0, 1]

                class_member_mask = (labels == k)

                xy = X[class_member_mask]
                plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                        markeredgecolor='k', markersize=6)

            plt.title('Number of clusters: %d' % n_clusters_)
            plt.savefig('result_fig/data_fusion.png')
        
        else:
            print('未定义数据融合模型！')
            return None

    def cal_interest(self, input_title_list):
        count = 0
        for title_text in input_title_list:
            if title_text:  # 检查标题是否为空
                vec1 = self.sentence_vector(title_text)
                if count == 0:
                    interest_embd = copy.deepcopy(vec1)
                else:
                    interest_embd += vec1
                count += 1
        interest_embd /= count
        target_dict = {'history_titles':input_title_list,
                       'interest_embd':interest_embd}
        return target_dict

    def recService(self, target_dict, title_list, embedding_list:list):
        # 处理用户浏览数据标题及其embedding，得到兴趣向量
        input_title = target_dict['history_titles']
        interest_embd =  target_dict['interest_embd']

        # 由兴趣向量与候选数据标题embedding计算相似度
        similarities = []
        for embd in embedding_list:
            similarity = np.dot(interest_embd, embd) / (np.linalg.norm(interest_embd) * np.linalg.norm(embd))
            similarities.append(similarity)
        
        # 获取top-k相关标题的索引
        top_k_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:self.config['topK']]
        related_title_list = [title_list[i] for i in top_k_indices]
        similarity_score_list = [similarities[i] for i in top_k_indices]
        # 以标题-相似度的csv文件格式保存数据结果
        result_dict = {}
        result_dict['recommend_title'] = related_title_list
        result_dict['recommend_similarity'] = similarity_score_list
        # result_df.to_csv('relatedIssures.csv', header=False, index=False)

        '''
        # 输出结果到文件
        output_file_path = "similar_titles.txt"
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            output_file.write("开始输出...\n")
            for idx in top_k_indices:
                related_title = title_list[idx]
                similarity_score = similarities[idx]
                output_line = f"标题: {related_title}，相似度分数: {similarity_score:.4f}\n"
                output_file.write(output_line)
        print('推荐结果已保存到文件: {}'.format(output_file_path))
        '''
        return result_dict