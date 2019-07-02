import csv

import pandas as pd

description_file = pd.read_csv('description_sentiment.csv')
message_file = pd.read_csv('message_sentiment.csv')
comments_file = pd.read_csv('comments_sentiment_raw.csv')
cnn_file = pd.read_csv('image_cnn.csv')
caption_file = pd.read_csv('caption_sentiment.csv')
reaction_file = pd.read_csv('reactions.csv')

# post_ids = []
# for row in range(description_file.shape[0]):
#     post_id_des = description_file.at[row, 'post_id']
#     post_ids.append(post_id_des)
#
# # for row in range(message_file.shape[0]):
# #     post_id_mes = message_file.at[row, 'post_id']
# #     if post_id_mes not in post_ids:
# #         print('weird')
#
# # comments_file = comments_file.set_index('post_id')
# unwanted_rows = []
# for row in range(comments_file.shape[0]):
#     post_id_com = comments_file.at[row, 'post_id']
#     if post_id_com not in post_ids:
#         print(post_id_com)
#         unwanted_rows.append(row)
#
# comments_file.drop(comments_file.index[unwanted_rows], inplace=True)
#
# print(description_file.shape[0], comments_file.shape[0])
# comments_file.to_csv('comments_sentiment.csv', encoding='utf8')
comments_file = pd.read_csv('comments_sentiment.csv')

datacolumns = [['post_id', 'description_subjectivity', 'description_polarity',
                            'message_subjectivity', 'message_polarity',
                            'comments_subjectivity', 'comments_positivity', 'comments_negativity',
                            'cnn_score',
                            'caption_subjectivity', 'caption_polarity',
                            'reaction_like', 'reaction_love', 'reaction_haha', 'reaction_wow', 'reaction_sad', 'reaction_angry',
                            'CLASSNAME']]

for row in range(description_file.shape[0]):
    post_id = description_file.at[row, 'post_id']

    description_subjectivity = description_file.at[row, 'description_subjectivity']
    description_polarity = description_file.at[row, 'description_polarity']

    message_subjectivity = message_file.at[row, 'message_subjectivity']
    message_polarity = message_file.at[row, 'message_polarity']

    comments_subjectivity = comments_file.at[row, 'comments_subjectivity']
    comments_positivity = comments_file.at[row, 'comments_positivity']
    comments_negativity = comments_file.at[row, 'comments_negativity']

    cnn_score = cnn_file.at[row, 'cnn_score']

    caption_subjectivity = caption_file.at[row, 'image_caption_subjectivity']
    caption_polarity = caption_file.at[row, 'image_caption_polarity']

    reaction_like = reaction_file.at[row, 'like']
    reaction_love = reaction_file.at[row, 'love']
    reaction_haha = reaction_file.at[row, 'haha']
    reaction_wow = reaction_file.at[row, 'wow']
    reaction_sad = reaction_file.at[row, 'sad']
    reaction_angry = reaction_file.at[row, 'angry']

    class_label = description_file.at[row, 'CLASSNAME']

    calculated_row = [post_id, description_subjectivity, description_polarity,
                      message_subjectivity, message_polarity,
                      comments_subjectivity, comments_positivity, comments_negativity,
                      cnn_score,
                      caption_subjectivity, caption_polarity,
                      reaction_like, reaction_love, reaction_haha, reaction_wow, reaction_sad, reaction_angry,
                      class_label]
    datacolumns.append(calculated_row)

data = pd.DataFrame(datacolumns[1:], columns=datacolumns[0])
data.to_csv('data.csv', encoding='utf8')
