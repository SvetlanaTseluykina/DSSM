import csv
import dssm as dssm
import words_to_matrix as wtm

import tensorflow as tf

dssm_model = dssm.Dssm()

# applying optimizer and loss function
dssm_model.compile(optimizer='adam',
                   loss='binary_crossentropy')

input_data = []
target_data = []
main_title = ''
with open('train.csv') as csvfile:
    try:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            query = [row[3]]
            title = [row[2]]
            if not main_title:
                main_title = title
            if main_title != title:  # train model when got all queries for title
                dssm_model.fit(input_data, target_data)
                input_data.clear()
                target_data.clear()
                dssm_model.save_weights('./checkpoints/my_checkpoint')  # save weights of trained model
                main_title = title
            query_trigram = wtm.sentences_to_bag_of_trigrams(query)  # the entry of query of trigrams into dictionary of all trigrams
            title_trigram = wtm.sentences_to_bag_of_trigrams(title)  # the entry of title of trigrams into dictionary of all trigrams
            input_data.append((query_trigram, title_trigram))  # append pair (query, title) to input data
            tnsr = tf.convert_to_tensor([float(row[4]) / 1.51], dtype='float32')  # relevance (should be [0; 2))
            target_data.append(tnsr)
    except:
        print(row)
csvfile.close()




