import datetime, re
import pandas as pd
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import nltk

from twitterscraper import query_tweets

# Stop words: additional words to be filtered
additional_stops=['next','see','hcm','booth','tech','la','vega','last',
                   'look','technology','work', 'announce','product','new',
                   'team','use','happen','time','take','make','everyone',
                   'anyone','week','day','year','let','go','come','word',
                   'employee','get','people','today','session','need',
                   'meet','help','talk','join','start','awesome','great',
                   'achieve','job','tonight','everyday','room','ready',
                   'one','company','say','well','data','share','love',
                   'want','like','good','business','sure','miss','demo',
                   'live','min','play','always','would','way','almost',
                   'thank','still','many','much','info','wow','play','full',
                   'org','create','leave','back','front','first','may',
                   'tomorrow','yesterday','find','stay','add','conference',
                   'top','stop','expo','hall','detail','row','award','hey',
                   'continue','put','part','whole','some','any','everywhere',
                   'convention','center','forget','congratulation','every',
                   'agenda','gift','card','available','behind','meeting',
                   'best','happen','unlockpotentialpic','half','none',
                   'human', 'resources','truly','win','possible','thanks',
                   'know','check','visit','fun','give','think','forward',
                   'twitter','com','pic','rt','via','really','very','like']


def wordnet_pos(word):
    """
    POS tag to first char lemmatize() accepts
    """
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def clean(text):
    '''
    Clean and tokenize text

    :param text: raw text string
    :returns: cleaned text
    '''
    # Convert to lowercase
    clean_text = text.lower()
    # Remove symbols and non-letters
    clean_text = re.sub(r'[^a-zA-Z]|(\w+:\/\/\S+)', ' ', clean_text).split()
    # Remove short words
    clean_text = [w for w in clean_text if len(w) > 2]
    # Lemmatize text
    lemmatizer = WordNetLemmatizer()
    clean_text = [lemmatizer.lemmatize(w, wordnet_pos(w)) for w in clean_text]
    # Filter out stop words
    stops = set(stopwords.words('english')).union(additional_stops)
    clean_text = [w for w in clean_text if w not in stops]

    return clean_text


def fetch_data(tweet_query):
    # Set start date to 5 days ago, end date to now
    start_date = datetime.datetime.now() - datetime.timedelta(days=5)
    start_date = start_date.date()
    end_date = datetime.datetime.now().date()
    # Number of tweets scraped (limit) can be modified to change performance
    list_of_tweets = query_tweets(tweet_query,
                                  begindate=start_date,
                                  enddate=end_date,
                                  limit=500,
                                  lang='en')
    tweets_df = pd.DataFrame([vars(x) for x in list_of_tweets])

    # Clean text
    cleaned_tweets_df = tweets_df.copy(deep=True)
    cleaned_tweets_df['token'] = [clean(x) for x in tweets_df['text']]
    return cleaned_tweets_df
