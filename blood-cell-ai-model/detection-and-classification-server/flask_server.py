from flask import Flask, json
from werkzeug.exceptions import HTTPException
from common import request_parse
import os
import flask_config
import argparse
from flask import jsonify
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



@app.route("/api/yolo-detection-class/get-blood-cell-class", methods=["GET"])
def batch_to_mask():
    req = request_parse()
    imagePath = req["imagePath"]
    start = time.time()
    print("{} rev req: {}".format(start, imagePath))
    path = os.path.join(config.image_folder, imagePath)
    fileExsit = os.path.exists(path)
    if fileExsit is False:
        print("{} file not exsit", imagePath)
        return jsonify()
    result = model.detection_model_eval(path)
    if len(result) == 0:
        return jsonify()
    f_result = model.classification_model_eval(result)
    end = time.time()
    print("{} succ class: {}, spend: {}".format(end, len(f_result), end-start))
    return jsonify(convert_class_result(f_result))

def convert_class_result(data: list)->dict:
    if data is None or len(data) == 0:
        return {}
    results = []
    for item in data:
        ans = {
            "x": -1,
            "y": -1,
            "w": -1,
            "h": -1,
            "class": "-1"
        }.copy()
        if len(item) == 2:
            rect = item[0]
            ans["x"] = rect[0]
            ans["y"] = rect[1]
            ans["w"] = rect[2]
            ans["h"] = rect[3]
            ans["class"] = item[1]
            results.append(ans)
    return results

@app.route("/")
def hello_world():
    return "<p>Hello, This Is Yolo Model Server! </p>"


if __name__ == '__main__':
    configFilePath = args.config
    print("configFilePath: ", configFilePath)
    config = flask_config.Config(configFilePath)
    print(config.image_folder)
    model = ObjectDetectionAndClassification(config.d_pth, config.c_pth, "")
    app.run(port=8080, debug=True, host="0.0.0.0")
