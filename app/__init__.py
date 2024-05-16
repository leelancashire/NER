from flask import Flask
from .config import Config

def create_app():
    """
    Create a Flask application using the app factory pattern.

    Returns:
        app (Flask): Flask application instance.
    """
    # see: https://python-adv-web-apps.readthedocs.io/en/latest/flask.html#test-flask
    app = Flask(__name__) 
    app.config.from_object(Config)

    # see: https://realpython.com/flask-blueprint/
    from .main import bp as main_bp
    app.register_blueprint(main_bp)

    return app
