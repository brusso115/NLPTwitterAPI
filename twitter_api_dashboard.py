# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import dash_bootstrap_components as dbc
import tensorflow as tf
import emoji
from transformers import AutoTokenizer
from Tweet import Tweet
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from plotly.subplots import make_subplots
import tweepy
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import hdbscan
import umap.umap_ as umap
import logging
import string
import dash_table
from Request import Request
from dash.dependencies import Input, Output, State, MATCH
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
pio.templates.default = "plotly_dark"
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
loaded_model = tf.keras.models.load_model('twitter-sentiment-model-1')

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options

app.layout = html.Div(children=[
    html.H1(children='Twitter Search Term Analysis',style={'text-align': 'center', 'backgroundColor':'rgb(17,17,17)',
                                                           'color': 'white', 'font': 'Helvetica'}),
    html.Div([
        dcc.Input(
            id='search-term',
            placeholder='Enter search term...',
            type='text',
            value='',
            style={'height': '40px', 'backgroundColor': 'rgb(51, 52, 50)', 'border': 'rgb(17,17,17)',
                   'border-radius': '5px', 'width': '99%', 'color': 'white', 'padding-left': '10px'},
        ),
        html.Button('Search', id='submit', n_clicks=0),

    ], style={'display':'flex'},),

    dcc.Loading(
        id='loading-3',
        type='default',
        children=[dcc.Graph(id="sentiment-chart")]
    ),

    dcc.Loading(
        id="loading-1",
        type="default",
        children=[dcc.Graph(id="scatter-plot")]
    ),

    dcc.Loading(
        id="loading-2",
        type="default",
        children=[html.Div(id='topics-list', children='')]
    )

    #dcc.Graph(id='top-words', style={'height':'1000px'})
])

@app.callback(
    [Output(component_id='scatter-plot', component_property='figure'),
     Output(component_id='topics-list', component_property='children'),
     Output(component_id='sentiment-chart', component_property='figure')],
    [Input(component_id='submit', component_property='n_clicks')],
    [State(component_id='search-term', component_property='value')],

)
def update_scatter(submit, search_term):
    print(submit)
    print(search_term)

    if submit >= 1:
        query = f'{search_term} -filter:retweets -filter:replies'
        max_tweets = 10

        api = Request().requestApiAccess()

        try:
            results = []
            for status in tweepy.Cursor(api.search, q=query, lang="en", result_type="mixed").items(max_tweets):
                results.append(status)
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error('Failed to upload to ftp: ' + str(e))

        tweetsNoStop = getTweetsNoStop(results)
        tweetsOrig = getOrigTweets(results)
        vectorizer = TfidfVectorizer()
        embedding_tfidf = vectorizer.fit_transform(tweetsNoStop)
        model_bert = SentenceTransformer('paraphrase-mpnet-base-v2')
        embedding_bert = np.array(model_bert.encode(tweetsNoStop, show_progress_bar=True))

        bert_tfidf_embedding = pd.concat([pd.DataFrame(embedding_bert), pd.DataFrame(embedding_tfidf.toarray())], axis=1)
        embedding_umap = umap.UMAP(n_neighbors=15, n_components=5, metric='cosine', min_dist=0.0).fit_transform(
            bert_tfidf_embedding)
        cluster = hdbscan.HDBSCAN(min_cluster_size=2, metric='euclidean', cluster_selection_method='eom').fit(
            embedding_umap)

        umap_data = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine').fit_transform(
            bert_tfidf_embedding)
        result = pd.DataFrame(umap_data, columns=['x', 'y'])
        result['labels'] = cluster.labels_

        outliers = result.loc[result.labels == -1, :]
        clustered = result.loc[result.labels != -1, :]
        clustered = pd.concat([pd.Series(tweetsOrig), clustered], axis=1)
        clustered = clustered.rename(columns={0: 'Tweet'})
        clustered = clustered.sort_values(by='labels')
        clustered['labels'] = clustered['labels'].astype(str)

        fig1 = px.scatter(
            clustered, x="x", y="y",
            color="labels",
            hover_data=['Tweet'],
            color_discrete_sequence = px.colors.qualitative.Bold)

        docs_df = pd.DataFrame(tweetsOrig, columns=["Doc"])
        docs_df['Topic'] = cluster.labels_
        docs_df['Doc_ID'] = range(len(docs_df))
        docs_per_topic = docs_df.groupby(['Topic'], as_index=False).agg({'Doc': ' '.join})

        tf_idf, count = c_tf_idf(docs_per_topic.Doc.values, m=len(tweetsOrig))
        top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20)
        topic_sizes = extract_topic_sizes(docs_df)

        topics = []
        for i in range(len(top_n_words)):
            topics.append(
                html.Div(children=[
                    dbc.CardHeader(
                        dbc.Button(
                            f'Topic {i}',
                            id={
                                'type': 'topic-button',
                                'index': i
                            }
                        ),
                    ),

                    dbc.Collapse(
                        html.Div(children=[dbc.CardBody(tweet) for tweet in docs_df[docs_df['Topic'] == i]['Doc']]),
                        id={
                            'type': 'topic-card',
                            'index': i
                        },
                        is_open=False
                    )
                ])

            )

        single_test_ids = np.zeros((len(results), 65))
        single_test_mask = np.zeros((len(results), 65))

        for i, sentence in enumerate(tweetsOrig):
            single_test_ids[i, :], single_test_mask[i, :] = tokenize(sentence)

        predictions = loaded_model.predict([single_test_ids, single_test_mask])
        pred_label = [np.argmax(pred) for pred in predictions]

        fig3 = px.bar(x=["Negative", "Positive"], y=[pred_label.count(0), pred_label.count(1)])

        '''
        fig2 = make_subplots(rows=round(len(top_n_words)/2) + 1, cols=2,
                             shared_xaxes=False,
                             vertical_spacing=0.03,
                             specs=[[{"type": "table"} for i in range(2)] for j in range(round(len(top_n_words)/2) + 1)]
                             )

        j = 1
        k = 1
        for i in range(0, len(top_n_words) - 1):

            if j >= 3:
                j=1

            print(k, j)
            fig2.add_trace(
                go.Table(
                    header=dict(
                        values=["Word", "Score"],
                        font=dict(size=10),
                        align="left"
                    ),
                    cells=dict(
                        values=[pd.DataFrame(top_n_words[i])[k][:5] for k in pd.DataFrame(top_n_words[i]).columns[:]],
                        align="left",
                    ),


                ),
                row=k, col=j
            ),
            j = j + 1
            if i % 2 != 0:
                k = k + 1
                
            '''

    else:
        fig1 = px.scatter(data_frame=[])
        topics = []
        fig3 = px.bar(data_frame=[])

    return fig1, topics, fig3

@app.callback(
    Output({'type': 'topic-card', 'index': MATCH}, 'is_open'),
    Input({'type': 'topic-button', 'index': MATCH}, 'n_clicks'),
    State({'type': 'topic-card', 'index': MATCH}, 'is_open'),
)
def display_tweets(n, is_open):
    if n:
        return not is_open
    return is_open

def getOrigTweets(results):
    tweetsOrig = []
    for tweet in results:
        tweet = Tweet(tweet)
        tweetsOrig.append(tweet.tweet_text)

    return tweetsOrig

def getTweetsNoStop(results):
    tweetsNoStop = []
    for tweet in results:
        tweet = Tweet(tweet)
        tweet.tweet_text = cleanTweet(tweet.tweet_text)
        tweet.tweet_text = removeStopwords(tweet.tweet_text)
        tweetsNoStop.append(tweet.tweet_text)
    return tweetsNoStop

def getTweetsWithStop(results):
    i = 0
    tweetsWithStop = []
    for tweet in results:
        tweet = Tweet(tweet)
        tweet.tweet_text = cleanTweet(tweet.tweet_text)
        tweetsWithStop.append(tweet.tweet_text)
    return tweetsWithStop


def cleanTweet(tweet):
    tweet = tweet.lower()
    tweet = tweet.replace("'", "").strip()
    tweet = tweet.replace('"', '').strip()
    tweet = tweet.replace('rt', '')
    tweet = " ".join(filter(lambda x: x[0] != '@', tweet.split()))
    tweet = " ".join([item for item in tweet.split() if not item.startswith('https')])

    tweet = tweet.replace('[^\w\s]', '')
    tweet = tweet.translate({ord(c): None for c in '!@#$\n]['})
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))

    return tweet

def removeStopwords(tweet):
    stop = set(stopwords.words('english'))
    tweet = [item for item in tweet.split() if item not in stop]
    return ' '.join(tweet)


def c_tf_idf(documents, m, ngram_range=(1, 1)):
    count = CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(documents)
    t = count.transform(documents).toarray()
    w = t.sum(axis=1)
    tf = np.divide(t.T, w)
    sum_t = t.sum(axis=0)
    idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
    tf_idf = np.multiply(tf, idf)

    return tf_idf, count

def extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20):
    words = count.get_feature_names()
    labels = list(docs_per_topic.Topic)
    tf_idf_transposed = tf_idf.T
    indices = tf_idf_transposed.argsort()[:, -n:]
    top_n_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in enumerate(labels)}
    return top_n_words

def extract_topic_sizes(df):
    topic_sizes = (df.groupby(['Topic'])
                     .Doc
                     .count()
                     .reset_index()
                     .rename({"Topic": "Topic", "Doc": "Size"}, axis='columns')
                     .sort_values("Size", ascending=False))
    return topic_sizes


def tokenize(sentence):
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
    tokens = tokenizer.encode_plus(sentence, max_length=65,
                                   truncation=True, padding='max_length',
                                   add_special_tokens=True, return_attention_mask=True,
                                   return_token_type_ids=False, return_tensors='tf')
    return tokens['input_ids'], tokens['attention_mask']


if __name__ == '__main__':
    app.run_server(debug=True)