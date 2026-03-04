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
from feedback import Feedback
import sqlite3
import os
from pathlib import Path

# Load the font from a local file
font_path = Path('static/styles/fonts/emulogic-font/Emulogic-zrEw.ttf')  # Specify the path to your font file


store = RedisStore(redis.StrictRedis())
app = Flask(__name__)
KVSessionExtension(store, app)
app.secret_key = "test3"

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

# Get the path to the current directory
current_directory = Path(__file__).resolve().parent

# Construct the file path relative to the current directory
file_path = current_directory / "result_database.sqlite"
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{file_path}"
db = SQLAlchemy(app)



# ── Feedback type ─────────────────────────────────────────────────────────────
# Set this to control which interface is shown to participants.
# Options: 'binary-feedback', 'ranked-feedback', 'ordinal-feedback'
FEEDBACK_TYPE = 'ordinal-feedback'

_FEEDBACK_TEMPLATES = {
    'binary-feedback':  'index.html',
    'ranked-feedback':  'index_ranked.html',
    'ordinal-feedback': 'index_ordinal.html',
}

def get_template():
    return _FEEDBACK_TEMPLATES.get(FEEDBACK_TYPE, 'index.html')

# ── Active learning ────────────────────────────────────────────────────────────
# When True, nextFrame() auto-advances through states that have already
# received enough informative feedback, only pausing at states where the
# agent's own count-based utility (matching feedback.py 'count' mode) is high.
ACTIVE_LEARNING = True
# Threshold on the combined score Nsa[s,a] + |hp[:,s,a] - hm[:,s,a]|.
# Feedback is requested while this score stays below the threshold —
# i.e. few visits AND uncertain feedback both contribute to requesting more.
ACTIVE_LEARNING_THRESHOLD = 3

def _wants_feedback():
    """Return True if feedback is informative at the current state-action pair.

    Uses the same count-based utility as feedback.py's 'count' active mode:
        utility = 1 / max(Nsa[s,a] + sum|hp[:,s,a] - hm[:,s,a]|, 0.1)
    which is equivalent to requesting feedback while
        Nsa[s,a] + feedback_certainty < ACTIVE_LEARNING_THRESHOLD.
    """
    if not ACTIVE_LEARNING:
        return True
    idx = session.get('current_environment_idx', 0)
    if idx == 0:
        return True
    ag  = session['agent']
    obs = session['obs'][idx]
    act = session['action_taken_list'][idx]
    if act < 0 or not hasattr(ag, 'Nsa') or ag.Nsa is None:
        return True
    nsa = float(ag.Nsa[obs, act])
    fb_certainty = float(np.abs(ag.hp[:, obs, act] - ag.hm[:, obs, act]).sum()) \
        if hasattr(ag, 'hp') and ag.hp is not None else 0.0
    return (nsa + fb_certainty) < ACTIVE_LEARNING_THRESHOLD

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

    algID   = 'tabQL_Cest_vi_t2'
    agent_h  = agent(algID, environment.nStates(), len(environment.action_list()))
    action_list = environment.action_list()
    action = 0
    ob = environment.st2ob()      # observation
    rw = 0                        # reward
    totRW = 0                     # total reward in this episode
    done = False                  # episode completion flag
    C  = np.array([0.2])
    fb = [[]] # Human feedback
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
    session['frame_positions'] = [[list(environment.pacman.pos), list(environment.ghost.pos)]]



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
    return render_template(get_template(),
                           img1=data2,
                           img2=data,
                           graph=data3,
                           session_status=session['status'][-1])


@app.route('/newepisode', methods=['POST'])
def new_episode():
    session['environment'].reset(random=True)
    session['agent'].prev_obs = None

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
    session['frame_positions'] = [[list(session['environment'].pacman.pos), list(session['environment'].ghost.pos)]]

    data, data2, data3 = new_environment_plots()
    return render_template(get_template(),
                           img1=data2,
                           img2=data,
                           graph=data3,
                           session_status=session['status'][-1])


@app.route('/next', methods=['POST'])
def nextFrame():
    # Advance up to 50 frames; stop as soon as we reach a state that needs
    # feedback (or the episode ends).  When ACTIVE_LEARNING is False every
    # frame needs feedback, so the loop always stops after a single step.
    for _ in range(50):
        idx = session['current_environment_idx']
        if session['status'][idx]:
            break  # episode done – don't advance further
        if idx + 1 == len(session['environment_display_list']):
            # Generate a new frame
            action = session['agent'].act(
                session['current_action'],
                session['obs'][idx],
                session['rws'][idx],
                session['status'][idx],
                [[]],
                0.5,
                update_Cest=False)
            session['current_action'] = action
            session['action_taken_list'].append(action)
            ob, rw, status = session['environment'].step(
                session['action_list'][action])
            session['obs'].append(ob)
            session['rws'].append(rw)
            session['status'].append(status)
            trail = session['frame_positions'][-2:]
            fig, _ = session['environment'].plot(trail=trail)
            data = base64EncodeFigure(fig)
            session['frame_positions'].append([
                list(session['environment'].pacman.pos),
                list(session['environment'].ghost.pos)
            ])
            session['current_environment'] = data
            session['environment_display_list'].append(data)
            session['current_environment_idx'] += 1
            session['environment_integer_list'].append(session['environment'].st2ob())
        else:
            # Browse to an already-generated frame
            session['current_environment_idx'] += 1
            session['current_environment'] = \
                session['environment_display_list'][session['current_environment_idx']]
            session['current_action'] = \
                session['action_taken_list'][session['current_environment_idx']]
        if _wants_feedback():
            break  # this state needs human feedback – show it
    idx = session['current_environment_idx']
    data2 = session['environment_display_list'][idx - 1] if idx > 0 \
        else session['environment_display_list'][0]
    return render_template(get_template(),
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
    return render_template(get_template(),
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
        data = request.get_json()
        print(data)
        arrow_dict = {'arrowup': 'n',
                      'arrowdown': 's',
                      'arrowleft': 'w',
                      'arrowright': 'e'}
        action_list = session['action_list']
        idx = session['current_environment_idx']
        taken_action = session['action_taken_list'][idx]
        n_actions = len(action_list)

        if FEEDBACK_TYPE == 'binary-feedback':
            feedback_string = arrow_dict[data['arrow']]
            action = action_list.index(feedback_string)
            # Label the human's chosen action (for Ce estimation)
            if action == taken_action:
                fb_obj = Feedback(state=session['obs'][idx], good_actions=[action], conf_good_actions=1.0)
            else:
                fb_obj = Feedback(state=session['obs'][idx], bad_actions=[action], conf_bad_actions=1.0)
            fb_label = feedback_string  # for DB

        elif FEEDBACK_TYPE == 'ranked-feedback':
            # data['ranking'] is a list of arrow strings best→worst
            ranking = data['ranking']
            Q_synthetic = np.zeros(n_actions)
            for rank_0, arrow_str in enumerate(ranking):
                act_idx = action_list.index(arrow_dict[arrow_str])
                Q_synthetic[act_idx] = n_actions - 1 - rank_0
            sorted_idx = np.argsort(-Q_synthetic).tolist()
            fb_obj = Feedback(
                state=session['obs'][idx],
                good_actions=sorted_idx[:-2],
                bad_actions=sorted_idx[-2:],
                conf_good_actions=1.0,
                conf_bad_actions=1.0)
            feedback_string = arrow_dict[ranking[0]] if ranking else 'n'
            action = action_list.index(feedback_string)
            fb_label = feedback_string

        elif FEEDBACK_TYPE == 'ordinal-feedback':
            # data['values'] is a dict {'arrowup': 0.8, 'arrowleft': 0.3, ...}
            values = data['values']
            Q_synthetic = np.zeros(n_actions)
            for arrow_str, val in values.items():
                act_idx = action_list.index(arrow_dict[arrow_str])
                Q_synthetic[act_idx] = float(val)
            fb_obj = Feedback(
                state=session['obs'][idx],
                good_actions=Q_synthetic.tolist(),
                conf_good_actions=1.0)
            best_idx = int(np.argmax(Q_synthetic))
            action = best_idx
            fb_label = action_list[best_idx]

        fb = [[fb_obj]]

        _ = session['agent'].act(
                action,
                session['obs'][idx],
                session['rws'][idx],
                session['status'][idx],
                fb,
                0.5,
                update_Cest=True)
        session['Ce_list'].append(session['agent'].Ce[0])
        fig, ax = plot_confidence(session['Ce_list'])
        graph_data = base64EncodeFigure(fig)
        session['graph_data'] = graph_data

        sessionID = session['uid']
        idx = session['current_environment_idx']
        env = session['environment_integer_list'][idx-1]
        agent_action = session['action_list'][session['current_action']]

        entry = annotations(
            sessionID=str(sessionID), env=int(env), action=agent_action,
            feedback=fb_label, ce=session['agent'].Ce[0])
        db.session.add(entry)
        db.session.commit()
        data2 = session['environment_display_list'][session['current_environment_idx']-1]
    return render_template(get_template(),
                           img1=data2,
                           img2=session['current_environment'],
                           graph=graph_data,
                           session_status=session['status'][-1])


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)