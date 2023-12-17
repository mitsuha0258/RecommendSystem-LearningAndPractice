import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

"""
1. 读入数据集， 并进行缺失值的填充， 这里为了简单一些， 直接类别特征填充“-1”， 数值特征填充0
2. 类别特征的编码， 用的LabelEncoder编码， 数值特征的归一化处理
3. 划分开训练集和验证集保存到prepeocessed/文件夹下
"""


# feature formation
def sparseFeature(feat, feat_num, embed_dim=4):
    """
        create dictionary for sparse feature
        :param feat: feature_name
        :param feat_num: the total number of sparse features that do not repeat
        :param embed_dim: embedding dimension
        :return
    """
    return {'feat': feat, 'feat_num': feat_num, 'embed_dim': embed_dim}


def denseFeature(feat):
    """
        create dictionary for dense feature
        :param feat: dense feature name
        : return
    """
    return {'feat': feat}


# load and preprocess the data of cretio
def create_cretio_data(embed_dim=8, test_size=0.2):
    train_df = pd.read_csv('./data/train.csv')
    test_df = pd.read_csv('./data/test.csv')

    labels = train_df['Label']
    del train_df['Label']

    data_df = pd.concat((train_df, test_df))
    del data_df['Id']

    print(data_df.columns)

    # process the feature separately
    sparse_feas = [col for col in data_df.columns if col[0] == 'C']
    dense_feas = [col for col in data_df.columns if col[0] == 'I']

    data_df[sparse_feas] = data_df[sparse_feas].fillna(-1)
    data_df[dense_feas] = data_df[dense_feas].fillna(0)

    feature_columns = [[denseFeature(feat) for feat in dense_feas]] + \
                      [[sparseFeature(feat, len(data_df[feat].unique()), embed_dim) for feat in sparse_feas]]
    np.save('preprocessed_data/fea_col.npy', feature_columns)

    # feature encode
    for feat in sparse_feas:
        data_df[feat] = LabelEncoder().fit_transform(data_df[feat])

    data_df[dense_feas] = MinMaxScaler().fit_transform(data_df[dense_feas])

    train, test = data_df[:train_df.shape[0]], data_df[train_df.shape[0]:]
    train['Label'] = labels

    train_set, val_set = train_test_split(train, test_size=test_size, random_state=2023)

    train_set.to_csv('preprocessed_data/train_set.csv', index=0)
    val_set.to_csv('preprocessed_data/val_set.csv', index=0)
    test.to_csv('preprocessed_data/test_set.csv', index=0)


if __name__ == '__main__':
    create_cretio_data()