import pandas as pd
import numpy as np
from PIL import Image
import random
from nltk.util import ngrams
from wordcloud import WordCloud
from collections import Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

def word_count(tweets, num_gram):
    '''
    Get word count dataframe from tweet data

    :param tweets: dataframe of cleaned tweets
    :param num_gram: number of words to pair together in word count (eg. 2 -> "Face mask")
    :returns: word count dataframe
    '''
    n_grams = list(ngrams(tweets, num_gram))
    common_words = Counter(n_grams).most_common()
    word_count = pd.DataFrame(data=common_words,
                              columns=['word', 'frequency'])
    # Convert list to string
    word_count['word'] = word_count['word'].apply(' '.join)
    # Plot word count graph
    word_count.head(20).sort_values('frequency').plot.barh(
        x='word', y='frequency', title='Word Frequency', figsize=(19, 10))
    word_count = word_count[word_count.index != 0]
    return word_count


def blue_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    # Color function for wordcloud styling
    return "hsl(217, 49%%, %d%%)" % random.randint(15, 35)


def wordcloud(word_count_df):
    '''
    Create a wordcloud from word count dataframe

    :param word_count_df: dataframe of word counts in tweet data returned by word_count()
    :returns: WordCloud object
    '''

    # Convert DataFrame to Map
    word_count_dict = {}
    for w, f in word_count_df.values:
        word_count_dict[w] = f
    # Generate word cloud
    mask_image = np.array(Image.open('./static/img/mask.png'))
    wordcloud = WordCloud(max_words=50, width=2400, height=1500, mask=mask_image,
                          random_state=12, background_color='white')
    wordcloud.generate_from_frequencies(word_count_dict)
    wordcloud.recolor(color_func=blue_color_func)
    return wordcloud, word_count_dict

def tweet_analysis(cleaned_tweets_df, query):
    num_grams = 1
    # Get only text data from tweet dataframe (exclude date, username, etc.)
    tweets_text = [word for one_tweet in cleaned_tweets_df['token'] for word in one_tweet]
    # Obtain word count and create, save wordcloud
    word_count_df = word_count(tweets_text, num_gram=num_grams)
    tweets_wordcloud, word_count_dict = wordcloud(word_count_df)
    tweets_wordcloud.to_file('./figures/{}.png'.format(query))
    # Create separate arrays of words and their respective frequencies for bar chart
    words = []
    freqs = []
    iter = 0
    for key, value in word_count_dict.items():
        if iter < 5:
            words.append(key)
            freqs.append(value)
    # Computer overall sentiment of tweets
    sentiment_text = ' '.join(tweets_text)
    sentiment = sid.polarity_scores(sentiment_text)
    return words[:5], freqs[:5], sentiment