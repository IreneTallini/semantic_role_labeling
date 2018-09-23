#IMPORT
from os import pardir, makedirs
from os.path import join, exists
from Dataset import NewVerbInventory
from tqdm import tqdm
import pickle
import Preprocessing
import numpy as np

#FOLDERS
Data_folder_path = join(pardir, 'Data')
bin_folder_path = join(pardir, 'bin')
ND_folder_path = join(Data_folder_path, 'LCL')
CONLL_folder_path = join(Data_folder_path, join('SRLData', 'EN'))
NVI_folder_path = join(Data_folder_path, 'LCL')
Test_folder_path = join(Data_folder_path, 'TestData')
eval_folder_path = join(pardir, 'eval')
models_folder_path = join(pardir, 'models')
clusters_file = join(NVI_folder_path, "training_cluster_strarg.csv")
verbs_file = join(NVI_folder_path, "ilaria.txt")

#CREATE NEW VERB INVENTORY
d_NVI = {}
NVI = NewVerbInventory(clusters_file, verbs_file)
for cluster in NVI.getClusters():
    for verb in cluster.getVerbs():
        d_NVI[verb.getPBFrameset()] = {'cluster_arg1' : cluster.getType1Arg(), 'cluster_arg2' : cluster.getType2Arg()}
        
# CREATE TRAIN DATASET
SET_ROLES = set(['A0', 'A1'])
dataset_file = join(CONLL_folder_path, 'CoNLL2009-ST-English-train.txt')
embeddings_file = join(Data_folder_path, 'glove.42B.300d.txt')
#train = Preprocessing.generate_training_set(dataset_file, SET_ROLES, embeddings_file, d_NVI, has_labels=True)
with open(join(bin_folder_path, 'train_dataset'), 'rb') as f:
    train = pickle.load(f)

# NEURAL MODEL
import tensorflow as tf
tf.reset_default_graph()
num_pos = len(train['dictionaries']['pos']['wti'])
num_words = len(train['dictionaries']['words']['wti'])
num_lemmas = len(train['dictionaries']['lemmas']['wti'])
num_roles = len(SET_ROLES) + 2

WORD_EMB_SIZE = 100
POS_EMB_SIZE = 16
LEMMA_EMB_SIZE = 100
PRED_EMB_SIZE = 128
ROLE_EMB_SIZE = 128
SEM_INFO_WORD_SIZE = 8
SEM_INFO_PRED_SIZE = 8
NUM_LSTM_LAYERS = 2
HIDDEN_SIZE_LSTM = 512
BATCH_SIZE = 30

fixed_word_embeddings_size = 300
emb_size = fixed_word_embeddings_size + WORD_EMB_SIZE + POS_EMB_SIZE + LEMMA_EMB_SIZE 

with tf.name_scope('inputs'):
    words = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None])
    lemmas = tf.placeholder(tf.int32, shape = [BATCH_SIZE, None])
    pos = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None])
    seq_len = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
    predicate_ids = tf.placeholder(tf.int32, shape = [BATCH_SIZE])
    predicate_indices = tf.placeholder(tf.int32, shape = [BATCH_SIZE])
    loss_mask = tf.placeholder(tf.int32, shape = [BATCH_SIZE, None])
    eval_mask = tf.placeholder(tf.bool, shape = [BATCH_SIZE, None])
    roles_mask = tf.placeholder(tf.bool, shape = [BATCH_SIZE, num_roles]) 
    one_hot_labels = tf.placeholder(tf.int32, shape = [BATCH_SIZE, None, num_roles])
    
with tf.name_scope('embedding_layer'):
    fixed_word_embeddings_matrix = tf.Variable((train['words_matrix']), trainable=False)
    word_embeddings_matrix = tf.Variable(tf.random_normal([num_words, WORD_EMB_SIZE])) 
    pos_embeddings_matrix = tf.Variable(tf.random_normal([num_pos, POS_EMB_SIZE])) 
    lemma_embeddings_matrix = tf.Variable(tf.random_normal([num_lemmas, LEMMA_EMB_SIZE])) 
    
    fixed_word_embeddings = tf.nn.embedding_lookup(fixed_word_embeddings_matrix, words)
    word_embeddings = tf.nn.embedding_lookup(word_embeddings_matrix, words)
    pos_embeddings = tf.nn.embedding_lookup(pos_embeddings_matrix, pos)
    lemma_embeddings = tf.nn.embedding_lookup(lemma_embeddings_matrix, lemmas)
    
    embeddings = tf.concat([fixed_word_embeddings, word_embeddings, pos_embeddings, 
                            lemma_embeddings], axis=-1)
    print('embeddings:', embeddings.shape)
    
with tf.name_scope('lstm_layers'):
    lstm_output, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn( 
        cells_fw=[tf.contrib.rnn.LSTMCell(HIDDEN_SIZE_LSTM) for _ in range(NUM_LSTM_LAYERS)], 
        cells_bw=[tf.contrib.rnn.LSTMCell(HIDDEN_SIZE_LSTM) for _ in range(NUM_LSTM_LAYERS)],
        inputs=embeddings, sequence_length=seq_len, dtype=tf.float32)
    print('lstm_output:', lstm_output.shape)
    pred_word_concat = tf.stack(
        [tf.map_fn(
            lambda word: tf.concat([word, lstm_output[i][predicate_indices[i]]], axis=0), lstm_output[i])
        for i in range(BATCH_SIZE)])
    print('pred_word_concat:', pred_word_concat.shape)

with tf.name_scope('classifier'):
    predicate_embeddings_matrix = tf.Variable(tf.random_normal([num_lemmas, PRED_EMB_SIZE])) 
    role_embeddings_matrix = tf.Variable(tf.random_normal([num_roles, ROLE_EMB_SIZE]))
    
    predicate_embeddings = tf.nn.embedding_lookup(predicate_embeddings_matrix, predicate_ids)
    predicate_embeddings = tf.expand_dims(predicate_embeddings, 1)
    predicate_embeddings = tf.tile(predicate_embeddings, [1, num_roles, 1])
    print('predicate_embeddings:', predicate_embeddings.shape)
    
    role_embeddings = tf.stack(
        [tf.where(roles_mask[i], role_embeddings_matrix, tf.zeros_like(role_embeddings_matrix)) for i in range(BATCH_SIZE)])
    print('role_embeddings:', role_embeddings.shape)    
    
    U = tf.Variable(tf.random_normal([PRED_EMB_SIZE + ROLE_EMB_SIZE, HIDDEN_SIZE_LSTM*4]))
    print('U:', U.shape)
    W = tf.nn.relu(tf.tensordot(tf.concat([predicate_embeddings, role_embeddings], axis=-1), U, axes=[[2], [0]]))
    
    logits = tf.stack([tf.tensordot(pred_word_concat[i], W[i], axes=[[1], [1]]) for i in range(BATCH_SIZE)])
    print('logits:', logits.shape)

with tf.name_scope('loss'):
    loss = tf.losses.softmax_cross_entropy(one_hot_labels, logits)
        
with tf.name_scope('optimizer'):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

with tf.name_scope('evaluation'):
    pred = tf.argmax(logits, axis=-1)
    pred = tf.reshape(pred, [-1])
    lab = tf.argmax(one_hot_labels, axis=-1)
    lab = tf.reshape(lab, [-1])
    eval_mas = tf.reshape(eval_mask, [-1])
    confusion_matrix = tf.confusion_matrix(lab, pred, weights=eval_mas)

saver = tf.train.Saver()

init = tf.global_variables_initializer()

# MODEL TRAINING
num_steps = 5
new_dataset = False
session_eval_folder_path = join(eval_folder_path, 
        "ND"+str(new_dataset)+"BS"+str(BATCH_SIZE)+"NS"+str(num_steps)+"LSTM"+str(NUM_LSTM_LAYERS))
session_models_folder_path = join(models_folder_path, 
        "ND"+str(new_dataset)+"BS"+str(BATCH_SIZE)+"NS"+str(num_steps)+"EC"+"LSTM"+str(NUM_LSTM_LAYERS))

loss_list = []
conf_mat_list = []
steps = []
with tf.Session() as session:
    init.run()
    for step in tqdm(range(num_steps), total=num_steps, unit="batch"):
        batch = Preprocessing.generate_batch(train, batch_size=BATCH_SIZE)
        l = len(batch['words'][0])
        em = [[True if i < sl else False for i in range(l)] for sl in batch['seq_len']]
        sl, cm, lo, _ = session.run(
            [seq_len, confusion_matrix, loss, optimizer],
            feed_dict={words:batch['words'], lemmas:batch['lemmas'], pos:batch['pos'], seq_len:batch['seq_len'],
                      predicate_ids:batch['predicate_ids'], predicate_indices:batch['predicate_indices'],
                       eval_mask:em, roles_mask:[[True]*4]*BATCH_SIZE, one_hot_labels:batch['labels']})
        steps.append(step)
        loss_list.append(lo)
        conf_mat_list.append(cm)
        if step%1 == 0:
            dr = batch['dictionaries']['roles']['wti']
            print('roles dictionary', batch['dictionaries']['roles']['wti'])
            #print('seq len', sl)
            print('loss', lo)
            print('confusion matrix: a[i][j] -> label = i, pred = j')
            print(cm)
            TP = cm[dr['A0']][dr['A0']] + cm[dr['A1']][dr['A1']]
            FP = sum([cm[i][dr['A0']] for i in range(4)]) + sum([cm[i][dr['A1']] for i in range(4)]) - TP
            FN = cm[dr['A0']][dr['_']] + cm[dr['A1']][dr['_']]
            prec = TP/(TP+FP)
            rec = TP/(TP+FN)
            if prec + rec != 0:
                f1 = 2*(prec*rec)/(prec+rec)
            else:
                f1 = 0
            print('prec', prec)
            print('rec', rec)
            print('f1', f1)
    
    if not exists(session_eval_folder_path):
        makedirs(session_eval_folder_path)
    if not exists(session_models_folder_path):
        makedirs(session_models_folder_path)
    with open(join(session_eval_folder_path, "steps"), 'wb') as f:
        pickle.dump(steps, f)
    with open(join(session_eval_folder_path, "loss"), 'wb') as f:
        pickle.dump(loss_list, f)
    with open(join(session_eval_folder_path, "conf_mat"), 'wb') as f:
        pickle.dump(conf_mat_list, f)
    with open(join(session_eval_folder_path, "train"), 'wb') as f:
        pickle.dump(train, f)
    save_path = saver.save(session, join(session_models_folder_path, "model.ckpt"))
    print("Model saved in path: %s" % save_path)

# MODEL EVALUATION
with open(join(bin_folder_path, 'dev_dataset'), 'rb') as f:
    dev = pickle.load(f)
with tf.Session() as session: 
    saver.restore(session, join(session_models_folder_path, "model.ckpt"))
    cm_list = []
    for i in range(0, len(dev['words']) - BATCH_SIZE - 1, BATCH_SIZE):
        test = Preprocessing.generate_batch(dev, batch_size=None, start=i)
        l = len(test['words'][0])
        em = [[True if i < sl else False for i in range(l)] for sl in test['seq_len']]
        size = len(test['words'])
        cm = session.run(
            [confusion_matrix],
            feed_dict={words:test['words'], lemmas:test['lemmas'], pos:test['pos'], seq_len:test['seq_len'],
                      predicate_ids:test['predicate_ids'], predicate_indices:test['predicate_indices'],
                       eval_mask:em, roles_mask:[[True]*4]*size, one_hot_labels:test['labels']})
        cm_list.append(cm)
        
cm_total = []
for i in range(4):
    cm_total_col = []
    for j in range(4):
        cm_total_col.append(sum([m[0][i][j] for m in cm_list]))
    cm_total.append(cm_total_col)
print('confusion matrix:')
print(np.array(cm_total))
print('roles:', test['dictionaries']['roles']['itw'])
cm = cm_total
dr = test['dictionaries']['roles']['wti']
TP = cm[dr['A0']][dr['A0']] + cm[dr['A1']][dr['A1']]
FP = sum([cm[i][dr['A0']] for i in range(4)]) + sum([cm[i][dr['A1']] for i in range(4)]) - TP
FN = cm[dr['A0']][dr['_']] + cm[dr['A1']][dr['_']]
prec = TP/(TP+FP)
rec = TP/(TP+FN)
if prec + rec != 0:
    f1 = 2*(prec*rec)/(prec+rec)
else:
    f1 = 0
print('prec', prec)
print('rec', rec)
print('f1', f1)
print('acc', )
