import dash
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc

import spacy
nlp = spacy.load('en_core_web_md')

import tensorflow_hub as hub
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np

#url = 'https://tfhub.dev/google/elmo/3'
#embed = hub.Module(url)

#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
#app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1('The ELMo based information retrieval system'),
    html.Hr(),
    html.Strong('Instructions: Paste the text you would like to search in the "Text Input" box. Next, write the question you would like to ask in the "Question Input" box. Finally, click "Search" to search for the top matching sentences.'),
    html.Hr(),

    html.Div([
        html.Div([
            html.H3('Text Input:'),
            dcc.Textarea(
                id='text-input',
                placeholder='Input your text here',
                value='''David Howell Petraeus AO, MSC (/pɪˈtreɪ.əs/; born November 7, 1952) is a retired United States Army general and public official. He served as Director of the Central Intelligence Agency from September 6, 2011, until his resignation on November 9, 2012. Prior to his assuming the directorship of the CIA, Petraeus served 37 years in the United States Army. His last assignments in the Army were as commander of the International Security Assistance Force (ISAF) and commander, U.S. Forces – Afghanistan (USFOR-A) from July 4, 2010, to July 18, 2011. His other four-star assignments include serving as the 10th commander, U.S. Central Command (USCENTCOM) from October 13, 2008, to June 30, 2010, and as commanding general, Multi-National Force – Iraq (MNF-I) from February 10, 2007, to September 16, 2008. As commander of MNF-I, Petraeus oversaw all coalition forces in Iraq.

Petraeus has a B.S. degree from the United States Military Academy, from which he graduated in 1974 as a distinguished cadet (top 5% of his class). In his class were three other future four-star generals, Martin Dempsey, Walter L. Sharp and Keith B. Alexander. He was the General George C. Marshall Award winner as the top graduate of the U.S. Army Command and General Staff College class of 1983. He subsequently earned an M.P.A. in 1985 and a Ph.D. degree in international relations in 1987 from the Woodrow Wilson School of Public and International Affairs at Princeton University. He later served as assistant professor of international relations at the United States Military Academy and also completed a fellowship at Georgetown University.

Petraeus has repeatedly stated that he has no plans to run for elected political office. On June 23, 2010, President Barack Obama nominated Petraeus to succeed General Stanley McChrystal as commanding general of the International Security Assistance Force in Afghanistan, technically a step down from his position as Commander of United States Central Command, which oversees the military efforts in Afghanistan, Pakistan, Central Asia, the Arabian Peninsula, and Egypt.

On June 30, 2011, Petraeus was unanimously confirmed as the Director of the CIA by the U.S. Senate 94–0. Petraeus relinquished command of U.S. and NATO forces in Afghanistan on July 18, 2011, and retired from the U.S. Army on August 31, 2011. On November 9, 2012, he resigned from his position as director of the CIA, citing his extramarital affair with his biographer Paula Broadwell, which was reportedly discovered in the course of an FBI investigation. In January 2015, officials reported the FBI and Justice Department prosecutors had recommended bringing felony charges against Petraeus for allegedly providing classified information to Broadwell while serving as director of the CIA. Eventually, Petraeus pleaded guilty to one misdemeanor charge of mishandling classified information. ''',
                rows=10,
                style={'width': '50%'}
                ),
            html.Br()
        ], className="six columns"),

    html.Hr(),   

        html.Div([
            html.H3('Question Input:'),
            dcc.Textarea(
                id='question-input',
                placeholder='Input your question here',
                value='Was Petraeus confirmed by the Senate?',
                rows=2,
                style={'width': '50%'}
                ),  
        
            html.Br(),
        
            html.Button('Search', id='button'),
            html.H3(id='button-1'),
            html.Br(), 

        ], className="six columns"),

    html.Hr(),

        html.Div([
            html.H2('Top Matching Sentences:'),
            html.Div(id='output')
        ], className="six columns")


    ], className="row")
])



@app.callback(
        Output('output', 'children'),
        [Input('button', 'n_clicks')],
        [State('question-input', 'value'),
            State('text-input', 'value')])
def compute(n_clicks, question, text):
    #text = text.lower().replace('\n', ' ').replace('\t', ' ').replace('\xa0',' ') #get rid of problem chars
    text = text.replace('\n', ' ').replace('\t', ' ').replace('\xa0',' ') #get rid of problem chars
    text = ' '.join(text.split()) #a quick way of removing excess whitespace

    doc = nlp(text)

    sentences = []
    for i in doc.sents:
        if len(i)>1:
            sentences.append(i.string.strip()) #tokenize into sentencesprint(len(sentences))
 
    print(sentences) 

    url = 'https://tfhub.dev/google/elmo/3'
    embed = hub.Module(url)

    embeddings = embed([question], signature='default',
            as_dict=True)['default']

    #Start a session and run ELMo to return the embeddings in variable x

    with tf.Session() as sess:
       sess.run(tf.global_variables_initializer())
       sess.run(tf.tables_initializer())
       ques = sess.run(embeddings)
    
    embeddings = embed(sentences, signature='default',
            as_dict=True)['default']
    
    #Start a session and run ELMo to return the embeddings in variable x
    
    with tf.Session() as sess:
       sess.run(tf.global_variables_initializer())
       sess.run(tf.tables_initializer())
       ans = sess.run(embeddings)
    
    sim_matrix = cosine_similarity(ques,ans)
    
    print(sim_matrix)

    sentences = list(np.array(sentences)[np.argsort(sim_matrix[0])[::-1][:5]])
    print(sentences)
    return html.Table([
        html.Tbody([
            html.Tr([
                html.Td(i) 
                ]) for i in sentences
            ])
        ])
    #return sentences
    #return 'Question is {}. Answer is {}. Clicks are'.format(input1, input2, n_clicks)

if __name__ == '__main__':
    #app.run_server(debug=True)
    app.run_server()
