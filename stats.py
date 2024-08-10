from urlextract import URLExtract
import pandas as pd
from collections import Counter
from wordcloud import WordCloud
import numpy as np
import emoji

# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-classification", model="avichr/heBERT_sentiment_analysis")


extract = URLExtract()
# pip install urlextract


def fetchstats(selected_user, df):

    # if the selected user is a specific user,then make changes in the dataframe,else do not make any changes

    if selected_user != 'Overall':
        df = df[df['User'] == selected_user]

    num_messages = df.shape[0]
    words = []
    for message in df['Message']:
        words.extend(message.split())

    # counting the number of media files shared

    mediaommitted = df[df['Message'] == '<Media omitted>']

    # number of links shared

    links = []
    for message in df['Message']:
        links.extend(extract.find_urls(message))

    return num_messages, len(words), mediaommitted.shape[0], len(links)


# most busy users {group level}

def fetchbusyuser(df):

    df = df[df['User'] != 'Group Notification']
    count = df['User'].value_counts().head()

    newdf = pd.DataFrame((df['User'].value_counts()/df.shape[0])*100)
    return count, newdf


def createwordcloud(selected_user, df):

    if selected_user != 'Overall':
        df = df[df['User'] == selected_user]

    wc = WordCloud(width=500, height=500,
                   min_font_size=10, background_color='Black')

    df_wc = wc.generate(df['Message'].str.cat(sep=" "))

    return df_wc


# get most common words,this will return a dataframe of
# most common words

def getcommonwords(selecteduser, df):

    # getting the stopwords

    file = open('stop_words.txt', 'r')
    stopwords = file.read()
    stopwords = stopwords.split('\n')

    if selecteduser != 'Overall':
        df = df[df['User'] == selecteduser]

    temp = df[(df['User'] != 'Group Notification') |
              (df['User'] != '<Media omitted>')]

    words = []
    #innu class ille
    for message in temp['Message']:
        for word in message.lower().split():
            if word not in stopwords:
                words.append(word)

    mostcommon = pd.DataFrame(Counter(words).most_common(20))
    return mostcommon


def getemojistats(selecteduser, df):

    if selecteduser != 'Overall':
        df = df[df['User'] == selecteduser]

    emojis = []
    for message in df['Message']:
        emojis.extend([c for c in message if c in emoji.EMOJI_DATA])

    emojidf = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))

    return emojidf


def monthtimeline(selecteduser, df):

    if selecteduser != 'Overall':
        df = df[df['User'] == selecteduser]

    temp = df.groupby(['Year', 'Month_num', 'Month']).count()['Message'].reset_index()

    time = []
    for i in range(temp.shape[0]):
        time.append(temp['Month'][i]+"-"+str(temp['Year'][i]))

    temp['Time'] = time

    return temp


def monthactivitymap(selecteduser, df):

    if selecteduser != 'Overall':
        df = df[df['User'] == selecteduser]

    return df['Month'].value_counts()


def weekactivitymap(selecteduser, df):

    if selecteduser != 'Overall':
        df = df[df['User'] == selecteduser]

    return df['Day_name'].value_counts()


def Sentiment_analysis(selecteduser, df):
    
    if selecteduser != 'Overall':
        df = df[df['User'] == selecteduser]

    # Analyze sentiments
    results = pipe(df['Message'].tolist())
    
    
    
    # Overall by Majority

    # Count the occurrences of each sentiment
    sentiment_counts = {'POSITIVE': 0, 'NEGATIVE': 0, 'NEUTRAL': 0}

    for result in results:
        label = result['label'].upper()
        if label in sentiment_counts:
            sentiment_counts[label] += 1
        else:
            sentiment_counts['NEUTRAL'] += 1  # Handle unexpected labels as neutral

    # Determine the overall sentiment based on the majority
    if sentiment_counts['POSITIVE'] > sentiment_counts['NEGATIVE'] and sentiment_counts['POSITIVE'] > sentiment_counts['NEUTRAL']:
        overall_sentiment = 'Positive'
    elif sentiment_counts['NEGATIVE'] > sentiment_counts['POSITIVE'] and sentiment_counts['NEGATIVE'] > sentiment_counts['NEUTRAL']:
        overall_sentiment = 'Negative'
    elif sentiment_counts['NEUTRAL'] > sentiment_counts['POSITIVE'] and sentiment_counts['NEUTRAL'] > sentiment_counts['NEGATIVE']:
        overall_sentiment = 'Neutral'
    else:
        # If there's a tie, calculate the weighted score
        weights = {'POSITIVE': 1, 'NEGATIVE': -1, 'NEUTRAL': 0}
        weighted_score = (sentiment_counts['POSITIVE'] * weights['POSITIVE'] +
                          sentiment_counts['NEGATIVE'] * weights['NEGATIVE'] +
                          sentiment_counts['NEUTRAL'] * weights['NEUTRAL'])

        if weighted_score > 0:
            overall_sentiment = 'Positive'
        elif weighted_score < 0:
            overall_sentiment = 'Negative'
        else:
            overall_sentiment = 'Neutral'

    return overall_sentiment, sentiment_counts