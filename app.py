import pickle
import json
import flask
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# filename
filename = 'trained_model.sav'
# load the model from disk
cosine_sim = pickle.load(open(filename, 'rb')) 
# load titles
all_Titles = []
with open('titles.txt') as json_file:
    all_Titles += json.load(json_file)["titles"]
 
# initialize flask application
app = flask.Flask(__name__, template_folder='templates')


#recommnedation part
def recommendation(title, cosine_sim=cosine_sim):
    id_i = all_Titles.index(title)
    sim_scores = list(enumerate(cosine_sim[id_i]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    game_i = [i[0] for i in sim_scores]
    return (all_Titles[index] for index in game_i)


# Set up the main route
@app.route('/', methods=['GET', 'POST'])

def index():
    if flask.request.method == 'GET':
        return(flask.render_template('index.html',game_list=all_Titles))
            
    if flask.request.method == 'POST':
        return flask.render_template('index.html', 
        game_list=all_Titles, 
        recommend_list=recommendation(flask.request.form['get_game_name']),
         m_name=flask.request.form['get_game_name'])

if __name__ == '__main__':
    app.run(debug=True)