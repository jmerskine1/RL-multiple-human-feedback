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
import json

from envPacMan import env
from agent import agent
from feedback import Feedback, get_active_utility
from trainer import PacmanTrainer
import sqlite3
import os
from pathlib import Path

# Load the font from a local file
font_path = Path('static/styles/fonts/emulogic-font/Emulogic-zrEw.ttf')  # Specify the path to your font file


store = RedisStore(redis.StrictRedis())
app = Flask(__name__)
KVSessionExtension(store, app)
app.secret_key = "test3"

# Ensure feedback directory exists at startup
BASE_DIR = Path(__file__).resolve().parent
FEEDBACK_DIR = BASE_DIR / 'feedback'
os.makedirs(FEEDBACK_DIR, exist_ok=True)

@app.before_request
def ensure_session_initialized():
    # Only run for actual routes, not static files
    if request.endpoint in ['static', 'favicon']:
        return
        
    if 'uid' not in session or 'session_hash' not in session:
        session['uid'] = str(uuid.uuid4())
        session['session_hash'] = uuid.uuid4().hex[:8]
        session.modified = True

    # Always ensure paths are derived from the hash
    db_filename = f"session_{session['session_hash']}.sqlite"
    brain_filename = f"session_{session['session_hash']}_brain.pkl"
    session['db_path'] = str(FEEDBACK_DIR / db_filename)
    session['brain_path'] = str(FEEDBACK_DIR / brain_filename)
    
    if not os.path.exists(session['db_path']):
        # Initialize session-specific DB
        print(f"Initializing session DB at: {session['db_path']}")
        conn = sqlite3.connect(session['db_path'])
        conn.execute('''CREATE TABLE IF NOT EXISTS annotations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        sessionID TEXT,
                        env INTEGER,
                        feedback TEXT,
                        action TEXT,
                        ce REAL
                     )''')
        conn.commit()
        conn.close()
        session.modified = True

def get_db_connection():
    if 'db_path' not in session:
        return None
    conn = sqlite3.connect(session['db_path'])
    conn.row_factory = sqlite3.Row
    return conn

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

# colors
c_bg1 = '#FFFF00'
c_bg2 = '#000000'
c_fig = '#2F2E2E'
c_axislabels = '#F3F3F3'


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


# ── Active learning ────────────────────────────────────────────────────────────
# When True, nextFrame() auto-advances through states that have already
# received enough informative feedback, only pausing at states where the
# agent's own count-based utility (matching feedback.py 'count' mode) is high.
ACTIVE_LEARNING = True
# If True, runs an episode and selects the best states for feedback.
BATCH_MODE = True
# Number of feedbacks to request per episode in Batch Mode
FEEDBACKS_PER_EPISODE = 10
# Mode can be 'count' or 'entropy'
ACTIVE_LEARNING_MODE = 'count'


def initialise_environment():
    # 1. Setup Trainer
    trainer = PacmanTrainer(
        algID='tabQL_Cest_vi_t2',
        env_size='small',
        active_feedback_type=ACTIVE_LEARNING_MODE,
    )
    
    # Reload brain if it exists
    if os.path.exists(session['brain_path']):
        try:
            with open(session['brain_path'], 'rb') as f:
                brain_data = pickle.load(f)
                trainer.load_brain(brain_data)
        except Exception as e:
            print(f"Error loading brain: {e}")

    # 2. Get all previously labelled states from DB
    labelled_memory = set()
    conn = get_db_connection()
    if conn:
        try:
            rows = conn.execute('SELECT env FROM annotations').fetchall()
            for row in rows: labelled_memory.add(int(row['env']))
        except: pass
        finally: conn.close()

    # 3. Generate trajectory and filter
    for attempt in range(5):
        trainer.reset_episode(random=True)
        all_plots, all_obs, all_actions, all_status, all_valid_moves, frame_positions = [], [], [], [], [], []
        
        for _ in range(500):
            obs = trainer.ob
            pos = [list(trainer.env.pacman.pos), list(trainer.env.ghost.pos)]
            frame_positions.append(pos)
            
            vm = [trainer.env.pacman.newPos(trainer.env.pacman.pos, d) != trainer.env.pacman.pos for d in ['n', 's', 'e', 'w']]
            all_valid_moves.append(vm)
            
            fig, _ = trainer.env.plot(trail=frame_positions[-3:-1] if len(frame_positions) >= 3 else None)
            all_plots.append(base64EncodeFigure(fig))
            all_obs.append(obs)
            
            action_idx, _, _, done = trainer.step(feedback=[[]], update_Cest=False)
            all_actions.append(action_idx)
            all_status.append(done)
            if done: break

        valid_indices = []
        seen_in_traj = set()
        for i, done in enumerate(all_status):
            if not done and all_obs[i] not in labelled_memory and all_obs[i] not in seen_in_traj:
                valid_indices.append(i)
                seen_in_traj.add(all_obs[i])

        if valid_indices: break
    
    if not valid_indices:
        session['selected_indices'] = []
    elif BATCH_MODE:
        U = np.array([get_active_utility(all_obs[i], all_actions[i], trainer.agent, mode=ACTIVE_LEARNING_MODE) for i in valid_indices])
        if np.std(U) > 0: U = (U - np.mean(U)) / np.std(U)
        N_fb = min(len(valid_indices), FEEDBACKS_PER_EPISODE)
        idx_in_valid = np.arange(len(U)) if N_fb == len(U) else np.argpartition(-U + np.random.randn(len(U))*0.1, N_fb)[:N_fb]
        session['selected_indices'] = sorted([valid_indices[i] for i in idx_in_valid])
    else:
        session['selected_indices'] = valid_indices

    # Store essential UI data only in session
    session['all_plots'], session['all_obs'], session['all_actions'], session['all_status'], session['all_valid_moves'] = all_plots, all_obs, all_actions, all_status, all_valid_moves
    session['current_queue_idx'] = 0 
    idx = session['selected_indices'][0] if session['selected_indices'] else 0
    session['current_environment_idx'] = idx
    session['current_environment'] = all_plots[idx] if all_plots else ""
    if 'Ce_list' not in session: session['Ce_list'] = [0.5]
    session['graph_data'] = base64EncodeFigure(plot_confidence(session['Ce_list'])[0])
    session.modified = True

def new_environment_plots():
    idx = session['current_environment_idx']
    data = session['all_plots'][idx]
    valid_moves = session['all_valid_moves'][idx]
    
    font_path = Path("static/styles/fonts/emulogic-font/Emulogic-zrEw.ttf")
    custom_font = FontProperties(fname=str(font_path))
    fig2, ax = plt.subplots(figsize=(5, 5),edgecolor = c_bg2)
    ax.axis('off')
    ax.text(0.5, 0.5, f"Reviewing Frame {session['current_queue_idx'] + 1} of {len(session['selected_indices'])}",
             ha='center', va='center',c = '#FFFF00', fontsize=10, color='red',fontproperties=custom_font)
    data2 = base64EncodeFigure(fig2)

    fig3, _ = plot_confidence(session['Ce_list'])
    data3 = base64EncodeFigure(fig3)
    return data, data2, data3, valid_moves


@app.route('/')
def index():
    # Session/DB init is now handled by before_request hook
    if 'labelled_states' not in session:
        session['labelled_states'] = []
        
    initialise_environment()
    if not session['selected_indices']:
        return "No unlabelled states found in recent episodes. Refresh or wait for agent to explore more."
    
    data, data2, data3, valid_moves = new_environment_plots()
    session['graph_data'] = data3
    return render_template(get_template(),
                           img1=data2,
                           img2=data,
                           graph=data3,
                           valid_moves=valid_moves,
                           session_status=session['all_status'][session['current_environment_idx']])


@app.route('/newepisode', methods=['POST'])
def new_episode():
    initialise_environment()
    data, data2, data3, valid_moves = new_environment_plots()
    return render_template(get_template(),
                           img1=data2,
                           img2=data,
                           graph=data3,
                           valid_moves=valid_moves,
                           session_status=session['all_status'][session['current_environment_idx']])


@app.route('/next', methods=['POST'])
def nextFrame():
    if session['current_queue_idx'] + 1 < len(session['selected_indices']):
        session['current_queue_idx'] += 1
        idx = session['selected_indices'][session['current_queue_idx']]
        session['current_environment_idx'] = idx
        session['current_environment'] = session['all_plots'][idx]
    
    data, data2, data3, valid_moves = new_environment_plots()
    return render_template(get_template(),
                           img1=data2,
                           img2=session['current_environment'],
                           graph=session['graph_data'],
                           valid_moves=valid_moves,
                           session_status=session['all_status'][session['current_environment_idx']])


@app.route('/previous', methods=['POST'])
def previousFrame():
    if session['current_queue_idx'] > 0:
        session['current_queue_idx'] -= 1
        idx = session['selected_indices'][session['current_queue_idx']]
        session['current_environment_idx'] = idx
        session['current_environment'] = session['all_plots'][idx]
        
    data, data2, data3, valid_moves = new_environment_plots()
    return render_template(get_template(),
                           img1=data2,
                           img2=session['current_environment'],
                           graph=session['graph_data'],
                           valid_moves=valid_moves,
                           session_status=session['all_status'][session['current_environment_idx']])


@app.route('/submit', methods=['POST'])
def submit():
    # 1. Reconstruct Trainer and load brain
    trainer = PacmanTrainer(
        algID='tabQL_Cest_vi_t2',
        env_size='small',
        active_feedback_type=ACTIVE_LEARNING_MODE,
    )
    if os.path.exists(session['brain_path']):
        with open(session['brain_path'], 'rb') as f:
            trainer.load_brain(pickle.load(f))

    idx = session['current_environment_idx']
    valid_moves = session['all_valid_moves'][idx] # [n, s, e, w]
    
    data = request.get_json()
    arrow_dict = {'arrowup': 'n', 'arrowdown': 's', 'arrowleft': 'w', 'arrowright': 'e'}
    action_list = trainer.action_list
    taken_action = session['all_actions'][idx]
    invalid_indices = [i for i, valid in enumerate(valid_moves) if not valid]

    if FEEDBACK_TYPE == 'binary-feedback':
        feedback_string = arrow_dict[data['arrow']]
        action = action_list.index(feedback_string)
        bad_actions = [i for i in invalid_indices if i != action]
        good_actions = [action] if action == taken_action else []
        if action != taken_action: bad_actions.append(action)
        fb_obj = Feedback(state=session['all_obs'][idx], good_actions=good_actions, conf_good_actions=1.0, bad_actions=bad_actions, conf_bad_actions=1.0)
        db_feedback = feedback_string

    elif FEEDBACK_TYPE == 'ranked-feedback':
        ranking = data['ranking']
        Q_synthetic = np.zeros(len(action_list))
        for rank_0, arrow_str in enumerate(ranking):
            Q_synthetic[action_list.index(arrow_dict[arrow_str])] = len(action_list) - 1 - rank_0
        for inv_idx in invalid_indices: Q_synthetic[inv_idx] = -1
        sorted_idx = np.argsort(-Q_synthetic).tolist()
        fb_obj = Feedback(state=session['all_obs'][idx], good_actions=sorted_idx[:2], bad_actions=sorted_idx[2:], conf_good_actions=1.0, conf_bad_actions=1.0)
        db_feedback = json.dumps(ranking)

    elif FEEDBACK_TYPE == 'ordinal-feedback':
        values = data['values']
        Q_synthetic = np.zeros(len(action_list))
        for arrow_str, val in values.items():
            Q_synthetic[action_list.index(arrow_dict[arrow_str])] = float(val)
        for inv_idx in invalid_indices: Q_synthetic[inv_idx] = 0.0
        fb_obj = Feedback(state=session['all_obs'][idx], good_actions=Q_synthetic.tolist(), conf_good_actions=1.0)
        db_feedback = json.dumps(values)

    # 2. Update Agent
    trainer.agent.act(taken_action, session['all_obs'][idx], 0, False, [[fb_obj]], 0.5, update_Cest=True)
    session['Ce_list'].append(trainer.agent.Ce[0])
    session['graph_data'] = base64EncodeFigure(plot_confidence(session['Ce_list'])[0])

    # 3. Save DB entry
    conn = get_db_connection()
    if conn:
        conn.execute('''INSERT INTO annotations (sessionID, env, action, feedback, ce) 
                        VALUES (?, ?, ?, ?, ?)''', 
                     (str(session['uid']), int(session['all_obs'][idx]), 
                      action_list[taken_action], db_feedback, float(trainer.agent.Ce[0])))
        conn.commit()
        conn.close()

    # 4. Save updated brain to file
    with open(session['brain_path'], 'wb') as f:
        pickle.dump(trainer.get_brain(), f)
    
    # Update local memory
    if 'labelled_states' not in session: session['labelled_states'] = []
    session['labelled_states'].append(int(session['all_obs'][idx]))
    session.modified = True

    # 5. Move to next or regenerate
    if session['current_queue_idx'] + 1 < len(session['selected_indices']):
        session['current_queue_idx'] += 1
        idx = session['selected_indices'][session['current_queue_idx']]
        session['current_environment_idx'] = idx
        session['current_environment'] = session['all_plots'][idx]
    else:
        initialise_environment()
    
    if not session['selected_indices']:
        return "No more unlabelled states found. Refresh to try again."

    data, data2, data3, valid_moves = new_environment_plots()
    return render_template(get_template(),
                           img1=data2,
                           img2=session['current_environment'],
                           graph=data3,
                           valid_moves=valid_moves,
                           session_status=session['all_status'][session['current_environment_idx']])


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)