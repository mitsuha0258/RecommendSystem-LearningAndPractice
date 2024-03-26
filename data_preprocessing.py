import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, LabelEncoder

"""
1. 读入数据集， 并进行缺失值的填充， 这里为了简单一些， 直接类别特征填充“-1”， 数值特征填充0
2. 类别特征的编码， 用的LabelEncoder编码， 数值特征的归一化处理
3. 划分开训练集和验证集保存到dataset/cretio/preprocessed/文件夹下
"""


# feature formation
def sparseFeature(feat_name, feat_num, embed_dim=4):
    """
        create dictionary for sparse feature
        :param feat: feature_name
        :param feat_num: the total number of sparse features that do not repeat
        :param embed_dim: embedding dimension
        :return
    """
    return {'feat_name': feat_name, 'feat_num': feat_num, 'embed_dim': embed_dim}


def denseFeature(feat):
    """
        create dictionary for dense feature
        :param feat: dense feature name
        : return
    """
    return {'feat_name': feat}


# load and preprocess the data of cretio
def create_cretio_data(embed_dim=8):
    data_df = pd.read_csv('dataset/cretio/cretio.csv')

    # process the feature separately
    sparse_feats = [col for col in data_df.columns if col[0] == 'C']
    dense_feats = [col for col in data_df.columns if col[0] == 'I']
    data_df[sparse_feats] = data_df[sparse_feats].fillna('-1')
    data_df[dense_feats] = data_df[dense_feats].fillna(0)

    feature_columns = [[denseFeature(feat) for feat in dense_feats]] + \
                      [[sparseFeature(feat, len(data_df[feat].unique()), embed_dim) for feat in sparse_feats]]
    np.save('dataset/cretio/preprocessed/feat_col.npy', feature_columns)

    # # 类别特征 标签映射
    # for feat in sparse_feats:
    #     data_df[feat] = LabelEncoder().fit_transform(data_df[feat])
    # 连续特征归一化
    data_df[dense_feats] = MinMaxScaler().fit_transform(data_df[dense_feats])
    data_df = pd.get_dummies(data_df)

    data_df.to_csv('dataset/cretio/preprocessed/cretio_data.csv', index=0)


if __name__ == '__main__':
    create_cretio_data()
