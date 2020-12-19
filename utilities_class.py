import torch
from pytorch_transformers import BertTokenizer
from pytorch_transformers import BertModel
from arabert.preprocess_arabert import never_split_tokens, preprocess
from farasa.segmenter import FarasaSegmenter
from farasa.stemmer import FarasaStemmer
import torch
import re
import numpy as np

class utilities_class:
    def __init__(self, model_path):
        self.farasa_segmenter = FarasaSegmenter(interactive=True)
        self.farasaStemmer = FarasaStemmer(interactive=True)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertModel.from_pretrained(model_path,output_hidden_states=True)
        self.model.eval()
        self.counter = 0


    def farasa_toknizing(self, text):
        text_preprocessed = preprocess(text,
                                       do_farasa_tokenization=True,
                                       farasa=self.farasa_segmenter,
                                       use_farasapy=True)
        return text_preprocessed

    def farasa_stemming(self, text):
        return self.farasaStemmer.stem(text)


    def toknize_examples(self, ex):
        if ex is None:
            return None
        if ex.strip() != '':
            return self.farasa_toknizing(ex)
        else:
            return None

    def choose_example(self,quran_ex, ex):
        if quran_ex.strip() != "":
            return quran_ex
        elif ex.strip() != '':
            return ex
        else:
            return None

    def segmenting_word(self,word):

        seg = self.farasa_segmenting(word)
        seg_list = re.split('\s+', seg)
        for s in seg_list:
            if '+' not in s:
                return s

        max_i = -1
        max_l = -1
        i = 0
        for s in seg_list:
            if len(s) > max_l:
                max_l = len(s)
                max_i = i
            i += 1
        return seg_list[max_i]

    def marke_text(self,text):
        marked_text = "[CLS] " + text + " [SEP]"
        return marked_text

    def bert_tokenizer(self,marked_text):
        tokenized_text = self.tokenizer.tokenize(marked_text)
        # Map the token strings to their vocabulary indeces.
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        return indexed_tokens

    def convert_input_to_tourch_tensor(self,indexed_tokens):
        tokens_tensor = torch.tensor([indexed_tokens])
        return tokens_tensor

    def get_def_tokens_tensor(self,text):

        marked_text = "[CLS] " + text + " [SEP]"
        tokenized_text = self.tokenizer.tokenize(marked_text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        return tokens_tensor

    def get_contextualized_word_embedding_layers_sum(self,example,ID,n):

        if ID is None:
            return None

        ID = int(ID)

        tokens_tensor = self.get_def_tokens_tensor(example)
        with torch.no_grad():
            outputs = self.model(tokens_tensor)
            
            if n < 1:
                # can use last hidden state as word embeddings
                last_hidden_state = outputs[0]
                word_embed_1 = last_hidden_state
        # Evaluating the model will return a different number of objects               based on how it's  configured in the `from_pretrained` call earlier. In this case, becase we set `output_hidden_states = True`, the third item will be the hidden states from all layers. See the documentation for more details:https://huggingface.co/transformers/model_doc/bert.html#bertmodel
            hidden_states = outputs[2]
            word_embed = torch.stack(hidden_states[-1*n:]).sum(0)
            return np.asarray(word_embed[:, ID+1, :])

    def get_contextualized_word_embedding_layers_concat(self, example, ID, n):
        if ID is None:
            return None
        ID = int(ID)
        tokens_tensor = self.get_def_tokens_tensor(example)
        with torch.no_grad():
            outputs = self.model(tokens_tensor)
            # Evaluating the model will return a different number of objects               based on how it's  configured in the `from_pretrained` call earlier. In this case, becase we set `output_hidden_states = True`, the third item will be the hidden states from all layers. See the documentation for more details:https://huggingface.co/transformers/model_doc/bert.html#bertmodel
            hidden_states = outputs[2]
          
            word_embeds = torch.cat([hidden_states[i] for i in range(len(hidden_states)-n,len(hidden_states))], dim=-1)
            return np.asarray(word_embeds[:, ID+1, :])

    def get_sentence_embedding_by_average_embedding (self, def_text):
        # `hidden_states` has shape [13 x 1 x 22 x 768]
        # `token_vecs` is a tensor with shape [22 x 768]
        def_text = str(def_text)
        def_text = re.sub('\s+', ' ', def_text)
        definition_splited = re.split('\s+', def_text)
        # print(len(definition_splited))
        if len(definition_splited) > 450:
            def_text = ' '.join(definition_splited[:200])
        with torch.no_grad():
            try:
                tokens_tensor = self.get_def_tokens_tensor(def_text)

                outputs = self.model(tokens_tensor)

            except:
                self.counter += 1
                print(self.counter)
                return None
            hidden_states = outputs[2]
            token_vecs = hidden_states[-2][0]
            # Calculate the average of all 22 token vectors.
            sentence_embedding = torch.mean(token_vecs, dim=0)
            # print("Our final sentence embedding vector of shape:", sentence_embedding.size())
            # Our final sentence embedding vector of shape: torch.Size([768])
            return np.asarray(sentence_embedding)

    def get_sentence_embedding_default (self,def_text ):
        # `hidden_states` has shape [13 x 1 x 22 x 768]
        # `token_vecs` is a tensor with shape [22 x 768]
        def_text = str(def_text)
        def_text = re.sub('\s+', ' ', def_text)
        definition_splited = re.split('\s+', def_text)
        print(len(definition_splited))
        if len(definition_splited) > 450:
            def_text = ' '.join(definition_splited[:449])

        tokens_tensor = self.get_def_tokens_tensor(def_text)

        with torch.no_grad():
            outputs = self.model(tokens_tensor)
            sentence_embed = outputs[1]
            return sentence_embed

    def get_word_index(self,example_tokenized, example, word, segmented_word,tokized_model=True):
        word = word.strip()
        if tokized_model:
            example_tokenized = str(example_tokenized)
            word = str(word)
            example_tokenized = re.sub('\+', ' ', example_tokenized)
            example_tokenized = re.sub('\s+', ' ', example_tokenized)
            example_tokenized_list = re.split('\s+', example_tokenized)
            x = -1
            if word not in example_tokenized_list:

                if segmented_word in example_tokenized_list:
                    x = example_tokenized_list.index(segmented_word)

                elif word[-1] in ['ه', 'ة']:
                    sub_word = word[:-1]
                    if sub_word in example_tokenized_list:
                        x = example_tokenized_list.index(sub_word)
                elif segmented_word[-1] in ['ه', 'ة']:
                    sub_word = segmented_word[:-1]
                    if sub_word in example_tokenized_list:
                        x = example_tokenized_list.index(sub_word)
                else:
                    for i in range(len(example_tokenized_list)):
                        if word in example_tokenized_list[i]:
                            x = i
                            break

                if x == -1:
                    return None

            else:
                x = example_tokenized_list.index(word)
            return int(x)
        else:
            example_splited = re.split('\s+',example)
            if word in example_splited:
                x= example_splited.index(word)
            else:
                for i in range(len(example_splited)):
                    if word in example_splited[i]:
                        x = i
                        break
            return x