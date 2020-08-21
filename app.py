from flask import Flask, request,  render_template, send_from_directory
from fetch_data import *
from analysis import *

app = Flask(__name__)

@app.route('/')
def home():
    '''
    Homepage
    '''
    return render_template('index.html')

@app.route('/analytics', methods=['POST'])
def analytics():
    '''
    Analytics page
    '''
    # Get inputed search query
    query = request.form['query']
    # Scrape, clean, and return relevant tweet data in dataframe format
    cleaned_tweets_df = fetch_data(query)
    # Obtain arrays of most used words/frequencies and overall sentiment
    words, freqs, sentiment = tweet_analysis(cleaned_tweets_df, query)
    pos = round(sentiment.get('pos') * 100, 1)
    neg = round(sentiment.get('neg') * 100, 1)

    return render_template('analytics.html', query=query, words=words, freqs=freqs, pos=pos, neg=neg)

@app.route('/figures/<filename>')
def send_output_file(filename):
    # Properly route requests to wordcloud figures
    return send_from_directory('./figures', filename)

if __name__ == "__main__":
    app.run(debug=True)
