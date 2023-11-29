import uuid
import logging
import base64
from io import BytesIO

from flask import Flask
from flask import render_template, request, session, Response
import redis
from flask_kvsession import KVSessionExtension
from simplekv.memory.redisstore import RedisStore
from flask_sqlalchemy import SQLAlchemy

import matplotlib.pyplot as plt
plt.style.use('ggplot')
import matplotlib
matplotlib.use('agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import pandas as pd
import numpy as np

from envPacMan import env
from agent import agent

store = RedisStore(redis.StrictRedis())
app = Flask(__name__)
KVSessionExtension(store, app)
app.secret_key = "test"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:////home/ah13558/Documents/taku_pacman/RL-Alex/result_database.sqlite"
db = SQLAlchemy(app)


class annotations(db.Model):
    __tablename__ = 'annotations'
    sessionID = db.Column(db.String, primary_key=True)
    env = db.Column(db.Integer)
    feedback = db.Column(db.String, unique=False)
    action = db.Column(db.Integer, unique=False)
    def __repr__(self):
        return '<sessionID %r>' % self.sessionID


def plot_confidence(Ce_list):
    fig, ax = plt.subplots(figsize = (6,3))
    ax.plot(Ce_list, color='#FFFF00')
    ax.set_facecolor('#2F2E2E')
    ax.set_xlabel('Number of feedbacks given')
    ax.set_ylabel('Estimated Confidence')
    ax.xaxis.label.set_color('#F3F3F3')
    ax.yaxis.label.set_color('#F3F3F3')
    ax.tick_params(axis='x', colors='#F3F3F3')
    ax.tick_params(axis='y', colors='#F3F3F3')
    ax.set_ylim(0.0, 1.0)
    return fig, ax


def base64EncodeFigure(fig):
    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor='#2F2E2E')
    del fig
    data = base64.b64encode(buf.getbuffer()).decode('ascii')
    return data


@app.route('/')
def index():
    # initialse user
    session['uid'] = uuid.uuid4()
    # Initialise Environment
    environment = env()
    environment.reset(random=False)
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

    fig, ax = environment.plot()
    fig2 = plt.Figure(figsize=(5, 5))
    data = base64EncodeFigure(fig)
    data2 = base64EncodeFigure(fig2)
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

    fig3, ax = plot_confidence(session['Ce_list'])
    data3 = base64EncodeFigure(fig3)
    session['graph_data'] = data3
    return render_template("index.html", img1=data2, img2=data, graph=data3)


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
            fig, ax = session['environment'].plot()
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
                           graph=session['graph_data'])


@app.route('/previous', methods=['POST'])
def previousFrame():
    if session['current_environment_idx'] != 0:
        session['current_environment_idx'] -= 1
        session['current_environment'] = \
            session['environment_display_list'][session['current_environment_idx']]
        session['current_action'] = session['action_taken_list'][session['current_environment_idx']]
    if session['current_environment_idx'] == 0:
        fig2 = plt.Figure(figsize=(5, 5))
        data2 = base64EncodeFigure(fig2)
    else:
        data2 = session['environment_display_list'][session['current_environment_idx']-1]
    return render_template("index.html",
                           img1=data2,
                           img2=session['current_environment'],
                           graph=session['graph_data'])


@app.route('/submit', methods=['POST'])
def submit():
    graph_data = session['graph_data']
    if session['current_environment_idx'] == 0:
        fig = plt.Figure(figsize=(5, 5))
        data2 = base64EncodeFigure(fig)
    else:
        # for plotting
        feedback = [request.form['Feedback%d'%i] if request.form.get('Checkbox%d'%i) else
                    'None' for i in range(1,5)]
        action_list = session['action_list']
        feedback = dict(map(lambda i,j : (i, j), action_list, feedback))
        idx = session['current_environment_idx']
        if any(key!='None' for key in feedback.values()):
            # If feedback is provided, then we train the agent for one
            # iteration.
            a = max(feedback, key=lambda k: float(feedback[k] if feedback[k]!='None' else 0.0))
            # Get integer corresponding to the action
            action = action_list.index(a)
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
            print(session['Ce_list'])
            print(session['agent'].Ce)
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
                feedback=str(feedback))
            db.session.add(entry)
            db.session.commit()
            data2 = session['environment_display_list'][session['current_environment_idx']-1]

    return render_template("index.html",
                           img1=data2,
                           img2=session['current_environment'],
                           graph=graph_data)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)