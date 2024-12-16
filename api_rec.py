from flask import Flask, request, jsonify
from model import RecModel, MixModel

app = Flask(__name__)

CONFIG = {'topK':10, # 最终推荐结果展示topK条数据
          'num_clusters':100,} # 融合呈现的数据簇个数
RECMODEL = RecModel(config=CONFIG)
# MIXMODEL = MixModel(config=CONFIG)

@app.route('/alive')
def alive():
    return jsonify({'Message': 'Success'})

# 数据推荐
@app.route('/rec', methods = ['GET','POST'])
def data_recommend():
    data = request.get_json()
    
    # 获取用户浏览序列
    user_browsing_titles = data.get('user_browsing_titles',[])
    all_data_titles = data.get('all_data_titles',[])

    # 获取非重复待推荐标题
    candidate_titles = list(set(all_data_titles)-set(user_browsing_titles))
    
    title_list = [] # 存放去空值的候选数据标题列表
    data_embedding_list = [] # 存放处理后的候选数据标题嵌入向量
    for title_text in candidate_titles:
        if title_text:  # 检查标题是否为空
            vec = RECMODEL.sentence_vector(title_text)
            data_embedding_list.append(vec)
            title_list.append(title_text)

    # 计算用户兴趣向量
    user_dict = RECMODEL.cal_interest(user_browsing_titles)
    rec_result = RECMODEL.recService(target_dict=user_dict, title_list=title_list, embedding_list=data_embedding_list) # 保存结果
    return jsonify(rec_result)

# 数据封装
@app.route('/fuse', methods = ['GET','POST'])
def data_fuse():
    data = request.get_json()
    all_data_titles = data.get('all_data_titles',[])

    data_embedding_list = [] # 存放处理后的候选数据标题嵌入向量
    for title_text in all_data_titles:
        vec = RECMODEL.sentence_vector(title_text)
        data_embedding_list.append(vec)

    fuse_result = RECMODEL.data_fusion(all_data_titles, data_embedding_list)
    return jsonify(fuse_result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)