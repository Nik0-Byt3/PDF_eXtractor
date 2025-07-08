from flask import Flask
from config import BaseConfig as Config

App = Flask(__name__)
App.config.from_object(Config)
