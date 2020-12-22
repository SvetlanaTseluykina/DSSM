import csv
import dssm as dssm
import words_to_matrix as wtm

dssm_model = dssm.Dssm()

dssm_model.compile(optimizer='adam',
                   loss='binary_crossentropy')



dssm_model.load_weights('./new_checkpoints/my_new_checkpoint')  # load weights from trained model

main_query = ''
data = []
with open('testnew.csv') as csvfile:
    csvreader = csv.reader(csvfile)
    cnt = 0
    for row in csvreader:
        query = [row[3]]
        title = [row[2]]
        if not main_query:
            main_query = query
        if main_query != query:
            print(dssm_model.predict(data))  # predict relevance (range [0;1]) from test data
            data.clear()
            main_query = query
        query_trigram = wtm.sentences_to_bag_of_trigrams(query)
        title_trigram = wtm.sentences_to_bag_of_trigrams(title)
        data.append((query_trigram, title_trigram))
