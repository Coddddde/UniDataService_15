from gensim.models import KeyedVectors
import os

from datarecEngine import Engine

# sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5.bz2是小型预训练未微调模型
# tencent-ailab-embedding-zh-d100-v0.2.0-s/tencent-ailab-embedding-zh-d100-v0.2.0-s.txt 是100维预训练小词向量模型
# tencent-ailab-embedding-zh-d100-v0.2.0/tencent-ailab-embedding-zh-d100-v0.2.0.txt 是100维预训练大词向量模型
# 词向量模型下载可以参考腾讯AI实验室：https://ai.tencent.com/ailab/nlp/en/embedding.html, 下载的模型需要解压至同级文件夹下，保证model_path读取正确


class RecModel(Engine):
    def __init__(self, config) -> None:
        super(RecModel,self).__init__(config)
        model_path = "tencent-ailab-embedding-zh-d100-v0.2.0/tencent-ailab-embedding-zh-d100-v0.2.0.txt"
        # 检查模型文件是否存在
        if os.path.exists(model_path):
            # 如果模型文件存在，加载模型
            print("开始加载数据推荐模型文件...")
            self.model = KeyedVectors.load_word2vec_format(model_path, binary=False)
            print("加载完成")
        else:
            print(f"模型文件 {model_path} 不存在，请检查文件路径是否正确。")

class MixModel(Engine):
    def __init__(self, config) -> None:
        super(MixModel,self).__init__(config)
        model_path = "sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5.bz2"
        # 检查模型文件是否存在
        if os.path.exists(model_path):
            # 如果模型文件存在，加载模型
            print("开始加载数据融合模型文件...")
            self.model = KeyedVectors.load_word2vec_format(model_path, binary=False)
            print("加载完成")
        else:
            print(f"模型文件 {model_path} 不存在，请检查文件路径是否正确。")