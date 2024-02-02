import uuid
import base64
from io import BytesIO
import gc
from flask import Flask
from flask import render_template, request, session, Response
import redis
from flask_kvsession import KVSessionExtension
from simplekv.memory.redisstore import RedisStore
from flask_sqlalchemy import SQLAlchemy

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
plt.style.use('ggplot')
import matplotlib
matplotlib.use('agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import numpy as np

from envPacMan import env
from agent import agent
import sqlite3
import os
from pathlib import Path
from PIL import Image, ImageFont, ImageDraw, ImageColor
# Load the font from a local file
font_path = Path('static/styles/fonts/emulogic-font/Emulogic-zrEw.ttf')  # Specify the path to your font file


def text_to_image(
text: str,
font_filepath: str,
font_size: int,
color: (int, int, int), #color is in RGB
font_align="center"):

   font = ImageFont.truetype(font_filepath, size=font_size)
   box = font.getsize_multiline(text)
   img = Image.new("RGBA", (box[0], box[1]))
   draw = ImageDraw.Draw(img)
   draw_point = (0, 0)
   draw.multiline_text(draw_point, text, font=font, fill=color, align=font_align)
   return img

store = RedisStore(redis.StrictRedis())
app = Flask(__name__)
KVSessionExtension(store, app)
app.secret_key = "test2"

if not os.path.exists('result_database.sqlite'):
        print("Creating database.")
    # If the file doesn't exist, initialize the database
        conn = sqlite3.connect('result_database.sqlite')
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE annotations (
                        sessionID TEXT,
                        env INTEGER,
                        feedback INTEGER,
                        action TEXT,
                        ce REAL
                     )''')
        conn.commit()
        conn.close()

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:////Users/jonathanerskine/University of Bristol/Human_Machine_Communication/expert_humans/RL-multiple-human-feedback/USER_INTERFACE_ALEX/RL-multiple-human-feedback/result_database.sqlite"
db = SQLAlchemy(app)



# colors
c_bg1 = '#FFFF00'
c_bg2 = '#000000'
c_fig = '#2F2E2E'
c_axislabels = '#F3F3F3'


class annotations(db.Model):
    __tablename__ = 'annotations'
    sessionID = db.Column(db.String, primary_key=True)
    env = db.Column(db.Integer)
    feedback = db.Column(db.String, unique=False)
    action = db.Column(db.Integer, unique=False)
    ce = db.Column(db.Float,unique=False)
    def __repr__(self):
        return '<sessionID %r>' % self.sessionID


def plot_confidence(Ce_list):
    fig, ax = plt.subplots(figsize = (6,3))
    ax.plot(Ce_list, color=c_bg2)
    ax.set_facecolor(c_bg2)
    ax.set_xlabel('Number of feedbacks given')
    ax.set_ylabel('Estimated Confidence')
    ax.xaxis.label.set_color(c_axislabels)
    ax.yaxis.label.set_color(c_axislabels)
    ax.tick_params(axis='x', colors=c_axislabels)
    ax.tick_params(axis='y', colors=c_axislabels)
    ax.set_ylim(0.0, 1.0)
    return fig, ax


def base64EncodeFigure(fig):
    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor=c_bg2)
    data = base64.b64encode(buf.getbuffer()).decode('ascii')
    buf.flush()
    buf.seek(0)
    plt.close()
    # gc.collect()
    return data


def initialise_environment():
    environment = env()
    environment.reset(random=True)
    disp = environment.display()

    algID   = 'tabQL_ps_Cest'
    agent_h  = agent(algID, environment.nStates(), len(environment.action_list()))
    action_list = environment.action_list()
    action = 0
    ob = environment.st2ob()      # observation
    rw = 0                        # reward
    totRW = 0                     # total reward in this episode
    done = False                  # episode completion flag
    C  = np.array([0.2])
    fb = np.ones(len(C)) * np.NaN # Human feedback
    current_environment = environment.display()

    fig, _ = environment.plot()
    data = base64EncodeFigure(fig)
    # Save to session
    session['environment'] = environment
    session['current_environment'] = data
    session['current_environment_idx'] = 0
    session['environment_display_list'] = [data]
    session['environment_integer_list'] = [environment.st2ob()]
    session['agent'] = agent_h
    session['action_list'] = action_list
    session['action_taken_list'] = [-1]
    session['current_action'] = -1
    session['obs'] = [ob]
    session['rws'] = [rw]
    session['totRW'] = totRW
    session['C'] = C
    session['fb'] = fb
    session['status'] = [done]
    disp = ''.join([x+'\n' for x in current_environment])
    session['disp'] = disp
    session['Ce_list'] = [0.5]



def new_environment_plots():
    fig, _ = session['environment'].plot()
    # fig2 = plt.Figure(figsize=(5, 5),edgecolor = c_bg2)


    # Create a plot without axes
    font_path = Path("static/styles/fonts/emulogic-font/Emulogic-zrEw.ttf")

    # Convert the Path object to a string
    font_path_str = str(font_path)

    # Create a FontProperties object with the font file path
    custom_font = FontProperties(fname=font_path_str)
    fig2, ax = plt.subplots(figsize=(5, 5),edgecolor = c_bg2)
    ax.axis('off')

    # Add text at the center
    ax.text(0.5, 0.5, "The first frame is omitted. \n\nYou can just pick any \n\ndirection and hit 'Submit'",
             ha='center', va='center',c = '#FFFF00', fontsize=10, color='red',fontproperties=custom_font)


    data = base64EncodeFigure(fig)
    data2 = base64EncodeFigure(fig2)

    fig3, _ = plot_confidence(session['Ce_list'])
    data3 = base64EncodeFigure(fig3)
    return data, data2, data3


@app.route('/')
def index():
    # initialse user
    session['uid'] = uuid.uuid4()
    initialise_environment()
    data, data2, data3 = new_environment_plots()
    session['graph_data'] = data3
    return render_template("index.html",
                           img1=data2,
                           img2=data,
                           graph=data3,
                           session_status=session['status'][-1])


@app.route('/newepisode', methods=['POST'])
def new_episode():
    session['environment'].reset(random=True)
    session['agent'].prev_obs = []

    # Reset environment
    fig, _ = session['environment'].plot()
    data = base64EncodeFigure(fig)
    session['current_environment'] = data
    session['current_environment_idx'] = 0
    session['environment_display_list'] = [data]
    session['environment_integer_list'] = [session['environment'].st2ob()]
    session['action_taken_list'] = [-1]
    session['current_action'] = -1
    session['obs'] = [session['environment'].st2ob()]
    session['rws'] = [0]
    session['totRW'] = 0
    session['status'] = [False]
    disp = ''.join([x+'\n' for x in session['environment'].display()])
    session['disp'] = disp

    data, data2, data3 = new_environment_plots()
    return render_template("index.html",
                           img1=data2,
                           img2=data,
                           graph=data3,
                           session_status=session['status'][-1])


@app.route('/next', methods=['POST'])
def nextFrame():
    idx = session['current_environment_idx']
    if not session['status'][idx]:
        if idx + 1 == \
            len(session['environment_display_list']):
            action = session['agent'].act(
                session['current_action'],
                session['obs'][idx],
                session['rws'][idx],
                session['status'][idx],
                np.ones(len(session['C'])) * np.NaN,
                0.5,
                skip_confidence=True)
            session['current_action'] = action
            session['action_taken_list'].append(action)
            ob, rw, status = \
                session['environment'].step(
                    session['action_list'][action])
            session['obs'].append(ob)
            session['rws'].append(rw)
            session['status'].append(status)
            fig, _ = session['environment'].plot()
            data = base64EncodeFigure(fig)
            session['current_environment'] = data
            session['environment_display_list'].append(data)
            session['current_environment_idx'] += 1
            session['environment_integer_list'].append(session['environment'].st2ob())
        else:
            session['current_environment_idx'] += 1
            session['current_environment'] = \
                session['environment_display_list'][session['current_environment_idx']]
            session['current_action'] = session['action_taken_list'][session['current_environment_idx']]
    data2 = session['environment_display_list'][session['current_environment_idx']-1]
    disp = ''.join([x+'\n' for x in session['environment'].display()])
    return render_template("index.html",
                           img1=data2,
                           img2=session['current_environment'],
                           graph=session['graph_data'],
                           session_status=session['status'][-1])


@app.route('/previous', methods=['POST'])
def previousFrame():
    if session['current_environment_idx'] != 0:
        session['current_environment_idx'] -= 1
        session['current_environment'] = \
            session['environment_display_list'][session['current_environment_idx']]
        session['current_action'] = session['action_taken_list'][session['current_environment_idx']]
    if session['current_environment_idx'] == 0:
        fig2 = plt.Figure(figsize=(5, 5),edgecolor = c_bg2)
        data2 = base64EncodeFigure(fig2)
    else:
        data2 = session['environment_display_list'][session['current_environment_idx']-1]
    return render_template("index.html",
                           img1=data2,
                           img2=session['current_environment'],
                           graph=session['graph_data'],
                           session_status=session['status'][-1])


@app.route('/submit', methods=['POST'])
def submit():
    graph_data = session['graph_data']
    if session['current_environment_idx'] == 0:
        fig = plt.Figure(figsize=(5, 5))
        data2 = base64EncodeFigure(fig)
        
    else:
        # for plotting
        data = request.get_json()
        arrow_dict = {'arrowup':'n',
                      'arrowdown':'s',
                      'arrowleft':'w',
                      'arrowright':'e',}
        
        feedback_string = arrow_dict[data['arrow']]
        
        # feedback_string = request.form['feedback_button']
        action_list = session['action_list']
        idx = session['current_environment_idx']
        # Get integer corresponding to the action
        action = action_list.index(feedback_string)
        taken_action = session['action_taken_list'][session['current_environment_idx']]
        fb = [1.0] if action == taken_action else [0.0]
        
        # Check what action the agent would take - check if it's the same as human feedback - 
        # 0 or 1 for feedback to the agent.
        _ = session['agent'].act(
                action,
                session['obs'][idx],
                session['rws'][idx],
                session['status'][idx],
                fb,
                0.5,
                skip_confidence=False)
        session['Ce_list'].append(session['agent'].Ce[0])
        fig, ax = plot_confidence(session['Ce_list'])
        graph_data = base64EncodeFigure(fig)
        session['graph_data'] = graph_data

        sessionID = session['uid']
        idx = session['current_environment_idx']
        env = session['environment_integer_list'][idx-1]
        action = session['action_list'][session['current_action']]
        
        entry = annotations(
            sessionID=str(sessionID), env=int(env), action=action,
            feedback=feedback_string, ce = session['agent'].Ce[0])
        db.session.add(entry)
        db.session.commit()
        data2 = session['environment_display_list'][session['current_environment_idx']-1]
    return render_template("index.html",
                           img1=data2,
                           img2=session['current_environment'],
                           graph=graph_data,
                           session_status=session['status'][-1])


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)