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
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import pandas as pd
import numpy as np

from envPacMan import env
from agent import agent

store = RedisStore(redis.StrictRedis())
app = Flask(__name__)
KVSessionExtension(store, app)
app.secret_key = "test"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:////home/ah13558/Documents/taku_pacman/RL-multiple-human-feedback/result_database.sqlite"
db = SQLAlchemy(app)


class annotations(db.Model):
    __tablename__ = 'annotations'
    sessionID = db.Column(db.String, primary_key=True)
    env = db.Column(db.Integer)
    label = db.Column(db.Integer, unique=False)
    action = db.Column(db.Integer, unique=False)
    def __repr__(self):
        return '<sessionID %r>' % self.sessionID

def base64EncodeFigure(fig):
    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png', bbox_inches='tight')
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
    session['ob'] = ob
    session['rw'] = rw
    session['totRW'] = totRW
    session['done'] = done
    session['C'] = C
    session['fb'] = fb
    disp = ''.join([x+'\n' for x in current_environment])
    session['disp'] = disp
    print(disp)

    return render_template("index.html", img1=data2, img2=data)


@app.route('/dontagree', methods=['POST'])
def dontagree():
    if session['current_environment_idx'] != 0:
        sessionID = session['uid']
        idx = session['current_environment_idx']
        env = session['environment_integer_list'][idx-1]
        action = session['action_list'][session['current_action']]
        entry = annotations(sessionID=str(sessionID), env=int(env), action=action, label=0)
        db.session.add(entry)
        db.session.commit()
        data2 = session['environment_display_list'][session['current_environment_idx']-1]
    else:
        fig = plt.Figure(figsize=(5, 5))
        data2 = base64EncodeFigure(fig)
    return render_template("index.html",
                           img1=data2,
                           img2=session['current_environment'])


@app.route('/agree', methods=['POST'])
def agree():
    if session['current_environment_idx'] != 0:
        sessionID = session['uid']
        idx = session['current_environment_idx']
        env = session['environment_integer_list'][idx-1]
        action = session['action_list'][session['current_action']]
        entry = annotations(sessionID=str(sessionID), env=int(env), action=action, label=1)
        db.session.add(entry)
        db.session.commit()
        data2 = session['environment_display_list'][session['current_environment_idx']-1]
    else:
        fig = plt.Figure(figsize=(5, 5))
        data2 = base64EncodeFigure(fig)
    return render_template("index.html",
                           img1=data2,
                           img2=session['current_environment'])


@app.route('/next', methods=['POST'])
def nextFrame():
    if session['current_environment_idx'] + 1 == \
        len(session['environment_display_list']):
        action = session['agent'].act(
            session['current_action'],
            session['ob'],
            session['rw'],
            session['done'],
            session['fb'],
            0.5)
        session['current_action'] = action
        session['action_taken_list'].append(action)
        session['ob'], session['rw'], session['done'] = \
            session['environment'].step(
                session['action_list'][action])
        # session['current_environment'] = session['environment'].display()
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
    print(disp)
    return render_template("index.html",
                           img1=data2,
                           img2=session['current_environment'])


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
                           img2=session['current_environment'])


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)