import  sys
sys.path.append('.')
import re
from utilities_class import utilities_class
import pandas as pd
import torch

class feature_extraction:
    def __init__(self,df_path,model_path,sample=False,tokenized=True, def_cutted=False):
        self.util_object = utilities_class(model_path)
        header_list = ["word", "body", "def", 'quran_ex', 'example']
        self.df = pd.read_csv(df_path,delimiter='$', encoding='utf-8', names=header_list)
        if sample:
            self.df = self.df.sample(10)
        self.df.drop(['body'], axis='columns', inplace=True)
        self.df.drop_duplicates()
        self.def_cutted = def_cutted
        if def_cutted:
            self.df["def_cutted"] = self.df.apply(lambda row: self.cut_def(row['def']), axis=1)
        self.tokenized = tokenized

    def cut_def(self, definition):
        lst = re.split('\s+', definition)
        if len(lst) > 31:
            lst = lst[:30]
            return ' '.join(lst)
        else:
            return definition

    def df_def_tokenization(self, input_col,out_col):
        self.df[out_col] = self.df.apply(lambda row: self.util_object.farasa_toknizing(row[input_col]), axis=1)
    # you should ignore the fisrt and the last embeddings

    def df_segmenting_examples(self,example,out_col):
        self.df[out_col] = self.df.apply(lambda row: self.util_object.toknize_examples(row[example]), axis=1)

    def df_choose_example(self,quran_ex,example,out_col):
        self.df[out_col] = self.df.apply(lambda row: self.util_object.choose_example(row[quran_ex], row[example]), axis=1)

    def df_segmenting_word(self,input,out_col):
        self.df[out_col] = self.df.apply(lambda row: self.util_object.farasa_stemming(row[input]), axis=1)

    def df_get_word_index(self, word, word_segmented, example_tokenized, choosed_Example, out_col):
        self.df[out_col] = self.df.apply(
            lambda row: self.util_object.get_word_index(
                row[example_tokenized],
                row[choosed_Example],
                row[word],
                row[word_segmented])
            , axis=1)

    def df_get_contextualized_word_embedding_by_average_embedding(self,example_col,word_index,n,out_col):

        self.df[out_col] = self.df.apply(
            lambda row: self.util_object.get_contextualized_word_embedding_layers_sum(row[example_col],
                 row[word_index], n), axis=1)

    def df_get_contextualized_word_embedding_layers_concat(self,example_col,word_index,n,out_col):
        self.df[out_col] = self.df.apply(
        lambda row: self.util_object.get_contextualized_word_embedding_layers_concat(row[example_col],
                                                                                  row[word_index], n), axis = 1)
    def df_generate_def_embeddings(self,def_col,out_col):

        self.df[out_col] = self.df.apply(lambda  row:
             self.util_object.get_sentence_embedding_by_average_embedding(row[def_col]),axis=1)



if __name__ == '__main__':
    csv_path = r"C:\Users\me250041\OneDrive - Teradata\Desktop\master\scrapping\data-factory\new_scrapping_all_out.txt"
    model_path = r'C:\Users\me250041\OneDrive - Teradata\Desktop\master\arabert-master\examples\aubmindlab\bert-base-arabert'
    feature_extraction_object = feature_extraction(csv_path, model_path)
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
    print('df_get_sentence_embedding_by_average_embedding')
    feature_extraction_object.df_get_contextualized_word_embedding_layers_concat \
        ('example_tokenized', 'word_index', 4, 'context_word_embeddings_concat')
    print('df_get_contextualized_word_embedding_layers_concat')

    feature_extraction_object.df_generate_def_embeddings('def_tokenized','def_embeddings')
    print('get_def_tokens_tensor')
