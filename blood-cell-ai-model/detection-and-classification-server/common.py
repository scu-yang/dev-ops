from flask import request

def request_parse():
    if request.method == 'POST':
        data =request.json
    elif request.method == 'GET':
        data = request.args
    return data