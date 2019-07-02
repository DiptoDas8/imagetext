from textblob import TextBlob

import pandas as pd
import csv

csv_file = '../data/comments.csv'
data_file = pd.read_csv(csv_file)
sentiment_file = [['post_id', 'comments_subjectivity', 'comments_positiity', 'comments_negativity', 'CLASSNAME']]

for row in range(data_file.shape[0]):
    im_id = data_file.at[row, 'post_id']
    class_label = data_file.at[row, 'CLASSNAME']
    comments = data_file.at[row, 'comments']
    if comments != '***':
        positivity = 0
        negativity = 0
        subjectivity = 0
        try:
            comments = comments.split('!@#$%^&*()_+')
            positivity = 0
            negativity = 0
            subjectivity = 0
            for comment in comments:
                analyzer = TextBlob(comment)
                # print(analyzer.sentiment)
                subjectivity += analyzer.sentiment.subjectivity
                if analyzer.sentiment.polarity > 0:
                    positivity += analyzer.sentiment.polarity
                else:
                    negativity += analyzer.sentiment.polarity
            print(im_id, subjectivity, positivity, negativity)
        except Exception as ex:
            print(ex)
        # calculated_row = [im_id, subjectivity, polarity, class_label]
    else:
        print('no comments found')
        subjectivity = 0
        positivity = 0
        negativity = 0
    calculated_row = [im_id, subjectivity, positivity, negativity, class_label]
    sentiment_file.append(calculated_row)

sentiment_df = pd.DataFrame(sentiment_file[1:], columns=sentiment_file[0])
sentiment_df.to_csv('comments_sentiment_raw.csv', encoding='utf-8')
