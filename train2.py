import csv
import dssm as dssm
import words_to_matrix as wtm
from random import randint

import tensorflow as tf

dssm_model = dssm.Dssm()

# applying optimizer and loss function
dssm_model.compile(optimizer='adam', loss='cosine_similarity', metrics='cosine_similarity')

input_data = []
target_data = []
neg_docs = []
pos_docs = []
main_query = ''

with open('trainnew.csv') as csvfile:
    try:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            query = [row[3]]
            title = [row[2]]
            if not main_query:
                main_query = query
            if main_query != query:  # train model when got all titles for query
                if len(neg_docs) >= 4 and len(pos_docs) >= 1:
                    input_data.append(pos_docs[0])
                    neg_indexes = []
                    while len(neg_indexes) != 4:
                        rand_index = randint(0, len(neg_docs) - 1)
                        if not neg_indexes.__contains__(rand_index):
                            neg_indexes.append(rand_index)
                    for index in neg_indexes:
                        input_data.append(neg_docs[index])
                    dssm_model.fit(input_data, target_data)
                    dssm_model.save_weights('./new_checkpoints/my_new_checkpoint')  # save weights of trained model
                input_data.clear()
                target_data.clear()
                pos_docs.clear()
                neg_docs.clear()
                main_query = query
            query_trigram = wtm.sentences_to_bag_of_trigrams(query)  # the entry of query of trigrams into dictionary of all trigrams
            title_trigram = wtm.sentences_to_bag_of_trigrams(title)  # the entry of title of trigrams into dictionary of all trigrams
            if round(float(row[4])) < 3:
                neg_docs.append((query_trigram, title_trigram))
            else:
                pos_docs.append((query_trigram, title_trigram))
            tnsr = tf.convert_to_tensor([float(row[4]) / 1.51], dtype='float32')  # relevance (should be [0; 2))
            target_data.append(tnsr)
    except Exception as e:
        print(e)
csvfile.close()
