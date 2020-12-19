#!/usr/bin/env python
# coding: utf-8

# In[ ]:




# In[1]:


from transformers import AutoTokenizer, AutoModel
from arabert.preprocess_arabert import never_split_tokens, preprocess
from farasa.segmenter import FarasaSegmenter
import torch

arabert_tokenizer = AutoTokenizer.from_pretrained(
    "aubmindlab/bert-base-arabert",
    do_lower_case=False,
    do_basic_tokenize=True,
    never_split=never_split_tokens)
arabert_model = AutoModel.from_pretrained("aubmindlab/bert-base-arabert") #you can replace the path here with the folder containing the the pytorch model

farasa_segmenter = FarasaSegmenter(interactive=True)


# In[2]:


text= "الجو جميل اليوم"
text_preprocessed = preprocess( text,
                                do_farasa_tokenization = True,
                                farasa = farasa_segmenter,
                                use_farasapy = True) # if you want to use AraBERT v0.1 do_farasa_tokenization = False
print(text)
print("---------------------")
print(text_preprocessed)


# In[24]:


arabert_input = arabert_tokenizer.encode(text_preprocessed,add_special_tokens=True)
print(arabert_input)
print(arabert_tokenizer.convert_ids_to_tokens(arabert_input))
# you should ignore the fisrt and the last embeddings


# In[23]:


tensor_input_ids = torch.tensor(arabert_input).unsqueeze(0)
print(tensor_input_ids)


# In[5]:


output = arabert_model(tensor_input_ids)


# In[6]:


output[0].shape # batch_size x seq_len x emb_dim


# In[27]:


embeddings = output[0][0][1:-1]


# In[25]:


print(embeddings.shape)
print(embeddings)


# In[9]:


import pandas as pd

header_list = ["word", "body", "def",'quran_ex','example']


df = pd.read_csv(r"C:\Users\me250041\OneDrive - Teradata\Desktop\master\scrapping\data-factory\new_scrapping_all_out.txt",delimiter='$',encoding='utf-8',names=header_list)


df[['word','def']]


# In[10]:


def farasa_segmenting(text):
    text_preprocessed = preprocess( text,
                                do_farasa_tokenization = True,
                                farasa = farasa_segmenter,
                                use_farasapy = True)
    return text_preprocessed


# In[11]:


df['def_tokenized']=df.apply(lambda row: farasa_segmenting(row['def']),axis=1)




def segmenting_examples(quran_ex,ex):
    if quran_ex.strip() != "":
        return farasa_segmenting(quran_ex)
    elif ex.strip() != '':
        return farasa_segmenting(ex)
    else:
        return None



df['example_tokenized']=df.apply(lambda row: segmenting_examples(row['quran_ex'],row['example']),axis=1)



import re
def example_encoding(example_s,word):
    word = word.strip()
    example_s = str(example_s)
    word = str(word)
    example_s = re.sub('\+',' ',example_s)
    example_s = re.sub('\s+',' ',example_s)
    example_s_list = re.split('\s+',example_s)
    if not word  in example_s_list:
        return None
    
    x = example_s_list.index(word)
    arabert_input = arabert_tokenizer.encode(example_s,add_special_tokens=True)
#     print(arabert_input)
#     print(arabert_tokenizer.convert_ids_to_tokens(arabert_input))
    tensor_input_ids = torch.tensor(arabert_input).unsqueeze(0)
    output = arabert_model(tensor_input_ids)
    
    embeddings = output[0][0][1:-1]
    
    return embeddings[x].tolist()

# embd = example_encoding('و+ عرض +نا جهنم يومئذ ل+ ال+ كافر +ين عرض +ا	','عرض')

# df['context_word_embeddings']=df.apply(lambda row: example_encoding(row['example_tokenized'],row['word']),axis=1)

examples = df['example_tokenized'].tolist()

words = df['word'].tolist()

result = []
for exam,word in zip(examples,words):
    result.append(example_encoding(exam,word))



# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('distilbert-base-nli-mean-tokens')
#
#
# # In[ ]:
#
#
# sentences = ['This framework generates embeddings for each input sentence',
#     'Sentences are passed as a list of string.',
#     'The quick brown fox jumps over the lazy dog.']
# sentence_embeddings = model.encode(sentences)
# And that's it already. We now have a list of numpy arrays with the embeddings.
#
# for sentence, embedding in zip(sentences, sentence_embeddings):
#     print("Sentence:", sentence)
#     print("Embedding:", embedding)
#     print("")
#
