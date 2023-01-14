from flask import Flask, json
from werkzeug.exceptions import HTTPException
from common import request_parse
import os
import flask_config
import argparse
import jsonify
import time

__version__ = "2023.01.14.01"

from yolo_detection import ObjectDetectionAndClassification

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, required=True, help='config file path')
args = parser.parse_args()

def version():
    return "version:" + __version__

app = Flask(__name__)



@app.errorhandler(HTTPException)
def handle_exception(e):
    """Return JSON instead of HTML for HTTP errors."""
    # start with the correct headers and status code from the error
    response = e.get_response()
    # replace the body with JSON
    response.data = json.dumps({
        "code": e.code,
        "name": e.name,
        "description": e.description,
    })
    response.content_type = "application/json"
    return response



@app.route("/api/batch-to-mask", methods=["GET"])
def batch_to_mask():
    req = request_parse()
    imagePath = req["imagePath"]
    start = time.time()
    print("{} rev req: {}".format(start, imagePath))
    path = os.path.join(config.image_folder, imagePath)
    result = model.detection_model_eval(path)
    if len(result) == 0:
        return jsonify()
    f_result = model.classification_model_eval(result)
    end = time.time()
    print("{} succ class: {}, spend: {}", end, len(f_result), end-start)
    return jsonify(f_result)

@app.route("/")
def hello_world():
    return "<p>Hello, This Is Unet Model Server! </p>"


if __name__ == '__main__':
    configFilePath = args.config
    print("configFilePath: ", configFilePath)
    config = flask_config.Config(configFilePath)
    print(config.image_folder)
    model = ObjectDetectionAndClassification(config.d_pth, config.c_pth, "")
    app.run(port=8080, debug=True, host="0.0.0.0")
