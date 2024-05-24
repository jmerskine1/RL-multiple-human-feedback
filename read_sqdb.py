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
plt.style.use('ggplot')
import matplotlib
matplotlib.use('agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import numpy as np

from envPacMan import env
from agent import agent

store = RedisStore(redis.StrictRedis())
app = Flask(__name__)
KVSessionExtension(store, app)
app.secret_key = "test2"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:////data/result_database.sqlite"
db = SQLAlchemy(app)

print(dir(db.session[0]['Ce_list']))