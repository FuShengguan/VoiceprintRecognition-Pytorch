from flask import Flask

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello Wo66576rld!'


if __name__ == '__main__':
    app.run()
