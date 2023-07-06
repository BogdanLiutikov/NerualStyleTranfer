import base64
from io import BytesIO

import flask
from PIL import Image as PILImage
from flask import Flask, render_template, request
from flask_socketio import SocketIO

from NST_main_script import main

app = Flask(__name__)
socketio = SocketIO(app,
                    async_mode='eventlet',
                    logger=True,
                    # engineio_logger=True
                    )


@app.route('/', methods=['GET', 'POST'])
def main_page():
    if request.method == 'GET':
        return render_template('main.html')
    else:
        content = PILImage.open(request.files['content'])
        style = PILImage.open(request.files['style'])
        iterations = int(request.form.get('iterations', '100'))
        #
        style_weight = float(request.form.get('style_weight', '6'))
        script = main(content, style, iterations=iterations, style_weight=10 ** style_weight, socketio=socketio)
        buffered = BytesIO()
        script.save(buffered, format="JPEG")
        encoded = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return flask.jsonify(encoded)


if __name__ == '__main__':
    socketio.run(app)
