import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import sys
sys.path.append('.')
from feature_extraction import feature_extraction
import re
import numpy as np
from tensorflow.keras import regularizers



def convert_to_tf_tensor(pytorch_tensor):
    np_tensor = pytorch_tensor.numpy()
    tf_tensor = tf.convert_to_tensor(np_tensor)
    return tf_tensor

def convert_txt_tensor_to_list(text_tensor):
    string_list = str(text_tensor)
    string_list = string_list.replace('tensor', '')
    string_list = re.sub(r'[\]\[\(\)\'"]', '', string_list)
    string_list = re.split(',\s+', string_list)
    float_list = [float(i) for i in string_list]
    return float_list


csv_path = r"C:\Users\me250041\OneDrive - Teradata\Desktop\master\scrapping\data-factory\new_scrapping_all_out.txt"
model_path = r'C:\Users\me250041\OneDrive - Teradata\Desktop\master\arabert-master\examples\aubmindlab\bert-base-arabert'

feature_extraction_object = feature_extraction(csv_path, model_path,True)
# df = feature_extraction_object.df
print("feature_extraction_object")
feature_extraction_object.df_def_tokenization('def', 'def_tokenized')
print('df_def_tokenization')
feature_extraction_object.df_choose_example('quran_ex', 'example', 'choosed_example')
print('df_choose_example')
feature_extraction_object.df_segmenting_examples('choosed_example', 'example_tokenized')
print('df_segmenting_examples')
feature_extraction_object.df_segmenting_word('word', 'word_stemed')
print('df_segmenting_word')
feature_extraction_object.df_get_word_index('word', 'word_stemed', 'example_tokenized', 'choosed_example', 'word_index')
feature_extraction_object.df = feature_extraction_object.df.dropna()
print('df_get_word_index')
feature_extraction_object.df_get_contextualized_word_embedding_by_average_embedding\
        ('example_tokenized','word_index',4,'context_word_embeddings')
print('df_get_contextualized_word_embedding_by_average_embedding')
feature_extraction_object.df_get_contextualized_word_embedding_layers_concat \
        ('example_tokenized', 'word_index', 4, 'context_word_embeddings_concat')
print('df_get_contextualized_word_embedding_layers_concat')
feature_extraction_object.df_generate_def_embeddings('def_tokenized', 'def_embeddings')
print('get_def_tokens_tensor')

# df = feature_extraction_object.df
feature_extraction_object.df = feature_extraction_object.df.dropna()
feature_extraction_object.df['context_word_embeddings'] = feature_extraction_object.df.apply(lambda row: list(row['context_word_embeddings'][0]), axis=1)
feature_extraction_object.df['def_embeddings'] = feature_extraction_object.df.apply(lambda row: list(row['def_embeddings']),axis=1)
feature_extraction_object.df['context_word_embeddings_concat'] = feature_extraction_object.df.apply(lambda row: list(row['context_word_embeddings_concat'][0]), axis=1)

# df = pd.DataFrame(df)

# d = pd.DataFrame()
# d.to_parquet()
feature_extraction_object.df.to_parquet('sample_pickled_df_30_only_def')
sys.exit(0)

df = pd.read_csv(r"C:\Users\me250041\OneDrive - Teradata\Desktop\master\arabert-master\examples\newDF.csv",
                 encoding='utf-8')


df['def_embeddings'] = df.apply(lambda row: convert_txt_tensor_to_list(row['def_embeddings']), axis=1)
df['context_word_embeddings'] = df.apply(lambda row: convert_txt_tensor_to_list(row['context_word_embeddings']), axis=1)
df['context_word_embeddings'] = df.apply(lambda row: convert_txt_tensor_to_list(row['context_word_embeddings']), axis=1)

df = df.dropna()
# msk = np.random.rand(len(df)) < 0.8
# train = df[msk]
# test = df[~msk]
#
# df_train_lable = train['context_word_embeddings']
#
# df_test_lable = test['context_word_embeddings']
#
# df_train_feat = train['def_embeddings']
#
# df_test_feat = test['def_embeddings']
#
# df_train_feat = pd.DataFrame(list(df_train_feat))
#
# df_test_feat = pd.DataFrame(list(df_test_feat))
#
# df_train_lable = pd.DataFrame(list(df_train_lable))
#
# df_test_lable = pd.DataFrame(list(df_test_lable))
#
# train_dataset = tf.data.Dataset.from_tensor_slices((df_train_feat.values, df_train_lable.values))
#
# test_dataset = tf.data.Dataset.from_tensor_slices((df_test_feat.values, df_test_lable.values))
#
# train_dataset = train_dataset.shuffle(len(train_dataset)).batch(1)
#
# for feat, targ in train_dataset.take(1):
#     print('Features: {}, Target: {}'.format(feat.shape, targ.shape))
#
#
# opt = keras.optimizers.Adam(learning_rate=0.01)
#
#
# def get_compiled_model():
#     model = tf.keras.Sequential([
#         tf.keras.layers.Dense(350, activation='relu', kernel_regularizer=regularizers.l2(l2=1e-4)),
#         tf.keras.layers.Dense(200, activation='relu', kernel_regularizer=regularizers.l2(l2=1e-4)),
#         tf.keras.layers.Dense(150, activation='relu', kernel_regularizer=regularizers.l2(l2=1e-4)),
#         tf.keras.layers.Dense(768)
#     ])
#
#     model.compile(optimizer=opt,
#                   loss=tf.keras.losses.mse,
#                   metrics='mse'
#                   )
#     return model
#
#
# model = get_compiled_model()
# model.fit(train_dataset, epochs=10)
#
#
#
# def predict(py_list):
#     return model.predict(tf.convert_to_tensor(py_list))
#
# model.predict(tf.reshape(tf.convert_to_tensor(df['context_word_embeddings'][0]), (1, -1)))
#
# df['predictions'] = df.apply(
#     lambda row: model.predict(tf.reshape(tf.convert_to_tensor(row['context_word_embeddings']), (1, -1))), axis=1)
#
# evaluation = model.evaluate(df_test_feat.values, df_test_lable.values)
# print("eval = ", str(evaluation))
# df.to_csv('./newpredicitons_takinglast4layers_mse', encoding='utf-8', index=False)
df["ID"] = df.index

df2 = df.copy()
df3 = df.copy()
df['label']=1
df2['label']=0
df3['label'] = 0

def generate_random(word,id,data):
    data_new =df.loc[df['ID']!=id]
    new_row = data_new.sample(1)
    return new_row['def_embeddings'].iloc[0]
def generate_hard(word,id,data):
    df_new = df.loc[(df['word']==word) & df['ID']!=id]
    if len(df_new)>0:
        new_row = df_new.sample(1)
        return new_row['def_embeddings']
    else:
        return None


df2['def_embeddings'] = df2.apply(lambda row : generate_random(row['word'],row['ID'],df),axis=1 )
df3['def_embeddings'] = df3.apply(lambda row : generate_hard(row['word'],row['ID'],df),axis=1 )
# df3.dropna()
concated_df = pd.concat(  [df,df2,df3])


df2['def_embeddings'] = df2.apply(lambda row : generate_random(row['word'],row['ID'],df),axis=1 )
df3['def_embeddings'] = df3.apply(lambda row : generate_hard(row['word'],row['ID'],df),axis=1 )
# df3.dropna()
concated_df = pd.concat(  [df,df2,df3])


from sklearn.metrics import mean_squared_error
df_temp = df.copy()
def mse(w_row,df_temp):
    df_temp['mse']=df_temp.apply(lambda row: mean_squared_error(w_row['predictions'][0],row['def_embeddings']),axis=1)
    df_temp = df_temp.sort_values(by=['mse'])
    top10 = list(df_temp['ID'][-10:])
    top3 = list(df_temp['ID'][-3:])
    top1 = list(df_temp['ID'][-1:])
    w_row['top10'] = top10
    w_row['top3'] = top3
    w_row['top1'] = top1
    if w_row['ID'] in top10:
        w_row['in_top10'] = 1
    else:
        w_row['in_top10'] = 0
    if w_row['ID'] in top3:
        w_row['in_top3'] = 1
    else:
        w_row['in_top3'] = 0
    if w_row['ID'] in top1:
        w_row['in_top1'] = 1
    else:
        w_row['in_top1'] = 0
    return w_row

df_new = df.apply(lambda row:mse(row,df_temp), axis=1)

print('top1 ', str(df_new['in_top1'].sum()/df_new['in_top1'].count()))
print('top 3 ', str(df_new['in_top3'].sum()/df_new['in_top1'].count()))
print('top10 ', str(df_new['in_top10'].sum()/df_new['in_top1'].count()))