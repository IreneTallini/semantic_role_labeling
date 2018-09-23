from collections import OrderedDict
import numpy as np

# Input: 
#1) file: embeddings file
#2) words: relevant words to embed
# Output: 
#1) matrix: embedding matrix where only rows corresponding to relevant words are selected
#2) vocabulary: dict mapping a word w into the index of the matrix row corresponding to the embedding of w
def fixed_embeddings(file, words):
    vocabulary = {}
    matrix = []
    with open(file, encoding='utf-8') as f:
        raw_file = f.readlines()
    i = 0
    for line in raw_file:
        partition = line.split(' ', maxsplit=1)
        word = partition[0]
        emb = [float(component) for component in partition[1].split(' ')]
        if word in words:
            vocabulary[word] = i
            matrix.append(emb)
            i += 1
    #print('embedding read in:', clock() - tic)
    return vocabulary, matrix
	
def to_one_hot(index, vocabulary_length):
    return [0] * index + [1] + [0] * (vocabulary_length - index - 1)
	
# Input: 
#1) word: OrderedDict 
#2) word_index: position of the word in the sentence
#3) predicate_index: position of the predicate p in the sentence
#4) predicate_num: number of preceding predicates in the sentence
# Output: 
#1) word1: same as word but:
    #a) if word is a predicate different from p 'FILLPRED' and 'PRED' are set to '_'
    #b) if word is not an argument of p 'APREDs' are set to '_'
    #c) if word is an argument of p but his role is not in set_roles 'APREDs' are set to '_'
def bind_to_predicate(word, word_index, predicate_index, predicate_num, set_roles, has_labels=False):
    word1 = OrderedDict(word)
    # a)
    if word_index != predicate_index:
        word1['FILLPRED'] = '_'
        word1['PRED'] = '_'
        # b) or c)
    if has_labels and not word['APREDs'][predicate_num] in set_roles:
        for i in range(len(word['APREDs'])):
            word1['APREDs'][i] = '_'
    return word1
	
# Input: 
#1) words: iterable of strings
#2) vocabulary: can be None (default) or a dictionary mapping each word to a different int
# Output: 
#1) d: a dictionary mapping each word to a different int (the same as vocabulary if its field is not None)
#2) d_r: reverse dictionary of d
def create_dictionary(words, vocabulary=None):
    #print('set_words', wordz)
    if vocabulary == None:
        d = {word : i for i, word in enumerate(words)}
        d_r = {item[1] : item[0] for item in d.items()}
        return d, d_r
    else:
        d = {word : vocabulary[word] for word in words if word in vocabulary}
        d_r = {item[1] : item[0] for item in d.items()}
        return d, d_r 
    
# Input: 
#1) dataset file: path of the dataset file
#2) set_roles: roles to use
#3) embeddings_file: path of the fixed words embeddings file
#4) has_labels: boolean which idicates if the dataset file is labeled
#5) train_dataset: when != None, indicates the dataset used for training
# Output: 
#1) dataset: a dictionary which is a container for all the tensors and dictionaries
# to be used in the model
def generate_training_set(dataset_file, set_roles, embeddings_file, d_LCL, has_labels=False, train_dataset=None):
    import csv
    fn = ['ID', 'FORM', 'LEMMA', 'PLEMMA', 'POS', 'PPOS', 'FEAT', 
      'PFEAT', 'HEAD', 'PHEAD', 'DEPREL', 'PDEPREL', 'FILLPRED', 'PRED']
    set_words = set(['unk'])
    set_lemmas = set(['unk'])
    set_pos = set(['unk'])
    set_pred = set(['unk'])
    set_roles = set_roles | set(['unk', '_'])
    dataset = {}    
    
    #a) Identifies words, lemmas, pos and predicates in the dataset and b) parses it in sentences
    with open(dataset_file, newline='') as csvfile:
        CONLL = csv.DictReader(csvfile, delimiter='\t', fieldnames=fn, restkey='APREDs')
        sent_CONLL1 = []
        CONLL1 = []
        #a)
        for row in CONLL:
            set_words.add(row['FORM'])
            set_lemmas.add(row['LEMMA'])
            set_pos.add(row['POS'])
            set_pred.add(row['PRED'])
            #b)
            if row['ID'] == '1':
                CONLL1.append(sent_CONLL1)
                sent_CONLL1 = []
            sent_CONLL1.append(row)
        CONLL1 = CONLL1[1:]
    
    # Creates dictionaries
    if train_dataset == None:
        words_vocabulary, words_matrix = fixed_embeddings(embeddings_file, set_words)
        d_words, d_words_r = create_dictionary(set_words, vocabulary=words_vocabulary)
        d_lemmas, d_lemmas_r = create_dictionary(set_lemmas)
        d_pos, d_pos_r = create_dictionary(set_pos)
        d_pred, d_pred_r = create_dictionary(set_pred)
        d_roles, d_roles_r = create_dictionary(set_roles)
    else:
        words_matrix = train_dataset['words_matrix']
        d_words = train_dataset['dictionaries']['words']['wti']
        d_words_r = train_dataset['dictionaries']['words']['itw']
        d_lemmas = train_dataset['dictionaries']['lemmas']['wti']
        d_lemmas_r = train_dataset['dictionaries']['lemmas']['itw']
        d_pos = train_dataset['dictionaries']['pos']['wti']
        d_pos_r = train_dataset['dictionaries']['pos']['itw']
        d_pred = train_dataset['dictionaries']['pred']['wti']
        d_pred_r = train_dataset['dictionaries']['pred']['itw']
        d_roles = train_dataset['dictionaries']['roles']['wti']
        d_roles_r = train_dataset['dictionaries']['roles']['itw']

    # When multiple (n) predicates are present in a sentence splits it in n sentences each referring to a single predicate
    CONLL2 = []
    labels = []
    predicate_ids = []
    predicate_indices = []
    count = 0
    for sentence in CONLL1:
        predicate_num = 0
        for word_index, word in enumerate(sentence):
            sent_CONLL2 = []
            sent_labels = []
            # When it finds a verbal predicate it binds all the sentence to it (see bind_to_predicate function)
            if word['FILLPRED'] == 'Y':
                if 'VB' in word['POS'] and word['PRED'] in d_LCL:
                    sent_CONLL2 = [bind_to_predicate(word, w_index, word_index, predicate_num, set_roles, has_labels) 
                                   for w_index, word in enumerate(sentence)]
                    
                    if has_labels:
                        sent_labels = [d_roles[word['APREDs'][predicate_num]] for word in sent_CONLL2]
                        #Deletes sentences with no arguments
                        if not all([label == d_roles['_'] for label in sent_labels]):  
                            CONLL2.append(sent_CONLL2)
                            predicate_ids.append(d_pred[word['PRED']] if word['PRED'] in d_pred else d_pred['unk'])
                            predicate_indices.append(word_index)
                            labels.append(sent_labels)
                        else:
                            #print(sent_labels)
                            count += 1
                    else:
                        CONLL2.append(sent_CONLL2)
                        predicate_ids.append(d_pred[word['PRED']] if word['PRED'] in d_pred else d_pred['unk'])
                        predicate_indices.append(word_index)
                predicate_num += 1
    #print('sent no arg:', count)
                
                
    words = [[d_words[word['FORM']] if word['FORM'] in d_words else d_words['unk'] for word in sentence] 
             for sentence in CONLL2]
    lemmas = [[d_lemmas[word['LEMMA']] if word['LEMMA'] in d_lemmas else d_lemmas['unk'] for word in sentence] 
             for sentence in CONLL2]
    pos = [[d_pos[word['POS']] if word['POS'] in d_pos else d_pos['unk'] for word in sentence] 
             for sentence in CONLL2]
    seq_len = [len(sentence) for sentence in CONLL2]
        
    dataset['words'] = words
    dataset['lemmas'] = lemmas
    dataset['pos'] = pos
    dataset['seq_len'] = seq_len
    dataset['predicate_ids'] = predicate_ids
    dataset['predicate_indices'] = predicate_indices
    dataset['labels'] = labels if has_labels else []
  
    dataset['dictionaries'] = {'words' : {'wti' : d_words, 'itw' : d_words_r}, 'lemmas' : {'wti' : d_lemmas, 'itw' : d_lemmas_r}, 
                               'pos' : {'wti' : d_pos, 'itw' : d_pos_r}, 'roles' : {'wti' : d_roles, 'itw' : d_roles_r},
                               'pred' : {'wti' : d_pred, 'itw' : d_pred_r}}
    dataset['words_matrix'] = words_matrix
    
    #print('dataset created in:', clock() - tic)
    
    return dataset

# Input: 
#1) data: output of the function generate_batch
#2) batch_size: sel explanatory
#3) start: where to start splitting the batch 
# Output: 
#1) data_new: a dictionary in the form of data, but with size batch_size
def generate_batch(data, batch_size=None, start=0):
    
    #Chose batch_size sentences at random
    num_sent = len(data['words'])
    if batch_size != None:
        id_sent = np.random.randint(0, num_sent, size=batch_size)
    else:
        id_sent = range(start, start + 30)
        batch_size = 30
    data_new = {field : [] for field in data.keys()}
    for field in set(data.keys()) - set(['dictionaries', 'words_matrix']):
        data_new[field] = [list(data[field][i]) if not 'int' in str(type(data[field][i])) else data[field][i] for i in id_sent]
    data_new['dictionaries'] = {item[0] : item[1] for item in data['dictionaries'].items()}
    data_new['words_matrix'] = [sent for sent in data['words_matrix']]
    
    # Pad with unk
    max_length = max(data_new['seq_len'])
    to_pad_fields = ['words', 'lemmas', 'pos']
    for i in range(batch_size):
        for field in to_pad_fields:
            data_new[field][i] += [data_new['dictionaries'][field]['wti']['unk']]*(max_length - data_new['seq_len'][i])
        data_new['labels'][i] += [data_new['dictionaries']['roles']['wti']['unk']]*(max_length - data_new['seq_len'][i])
    data_new['labels'] = [[to_one_hot(label, 4) for label in sent] 
                                   for sent in data_new['labels']]
    return data_new
 