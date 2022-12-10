# -*- coding: UTF-8 -*-
# @File    : server.py
import argparse
import functools
import os
import shutil
import sys
import json
import time
import pymysql
import pandas as pd
from flask import Flask, jsonify, make_response
from flask import request

import numpy as np
import torch

sys.path.append("C://jupyterNoteBook/VoiceprintRecognition-Pytorch/")
from data_utils.reader import load_audio, CustomDataset
from modules.ecapa_tdnn import EcapaTdnn, SpeakerIdetification
from utils.record import RecordAudio
from utils.utility import add_arguments, print_arguments
from flask_restful import Api, Resource
app = Flask(__name__, static_url_path="")
api = Api(app)
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('use_model', str, 'ecapa_tdnn', '所使用的模型')
add_arg('threshold', float, 0.48, '判断是否为同一个人的阈值')
add_arg('audio_db', str, 'audio_db', '音频库的路径')
add_arg('feature_method', str, 'melspectrogram', '音频特征提取方法', choices=['melspectrogram', 'spectrogram'])
add_arg('resume', str, 'C://jupyterNoteBook/VoiceprintRecognition-Pytorch//models//ecapa_mel_noise_visu',
        '模型文件夹路径')
args = parser.parse_args()
print_arguments(args)

dataset = CustomDataset(data_list_path=None, feature_method=args.feature_method)
# 获取模型
if args.use_model == 'ecapa_tdnn':
    ecapa_tdnn = EcapaTdnn(input_size=dataset.input_size)
    model = SpeakerIdetification(backbone=ecapa_tdnn)
else:
    raise Exception(f'{args.use_model} 模型不存在！')
# 指定使用设备
device = torch.device("cuda")
model.to(device)
# 加载模型
model_path = os.path.join(args.resume, args.use_model, 'model.pth')
model_dict = model.state_dict()
param_state_dict = torch.load(model_path)
for name, weight in model_dict.items():
    if name in param_state_dict.keys():
        if list(weight.shape) != list(param_state_dict[name].shape):
            param_state_dict.pop(name, None)
model.load_state_dict(param_state_dict, strict=False)
print(f"成功加载模型参数和优化方法参数：{model_path}")
model.eval()

person_feature = []
person_name = []
user_id_list = []


# 执行识别
def infer(audio_path):
    data = load_audio(audio_path, mode='infer', feature_method=args.feature_method)
    data = data[np.newaxis, :]
    data = torch.tensor(data, dtype=torch.float32, device=device)
    # 执行预测
    feature = model.backbone(data)
    return feature.data.cpu().numpy()


# 声纹识别
def recognition(path):
    name3 = ''
    pro = 0
    i1 = 0
    feature = infer(path)[0]
    # 遍历数据库计算余弦相似度
    for i, person_f in enumerate(person_feature):
        dist = np.dot(feature, person_f) / (np.linalg.norm(feature) * np.linalg.norm(person_f))
        if dist > pro:
            pro = dist
            name3 = person_name[i]
            i1 = i
    return name3, pro, i1


# 声纹注册
def register(path, user_name):
    save_path = os.path.join(args.audio_db, user_name + os.path.basename(path)[-4:])
    shutil.move(path, save_path)
    feature = infer(save_path)[0]
    person_name.append(user_name)
    person_feature.append(feature)


# 统一调用方法
def request_parse(req_data):
    # '''解析请求数据并以json形式返回'''
    if req_data.method == 'POST':
        data = req_data.json
    elif req_data.method == 'GET':
        data = req_data.args
    return data


@api.representation('application/json')
def output_json(data, code, headers=None):
    result = {}
    if app.specs_url == request.url:
        result = data
    elif code in (200, 201, 204):
        result['code'] = 200
        result['data'] = data
    else:
        result['code'] = data.get('code') or 500
        result['message'] = data['message']
    response = make_response(json.dumps(result), code)
    response.headers.extend(headers or {})
    return response


class BaseException(Exception):
    http_code = 500
    business_code = 200000

    def __init__(self, message=None):
        if message:
            self.message = message

    def to_dict(self):
        data = {'code': self.business_code, 'message': self.message}
        return data


class InsufficientPrivilege(BaseException):
    http_code = 403
    business_code = 200001
    message = '权限不足'


@app.errorhandler(BaseException)
def handle_base_exception(error):
    return error.to_dict(), error.http_code


# 加载要识别的音频库
def load_audio_db(audio_db_path):
    conn = pymysql.connect(host='192.168.28.66', user='root', password='fsg199710020', port=3306,
                           db='iot_saas_info')
    sql = "Select name,user_id,feature,audio_path,tenant_id FROM base_feature"
    cur = conn.cursor()
    cur.execute(sql)
    u = cur.fetchall()
    print(u)
    print("name", "user_id", "audio_path", "tenant_id")
    for row in u:
        name1 = row[0]
        user_id = row[1]
        a = list(map(float, row[2].split(' ')))
        feature = np.array(a)
        audio_path = row[3]
        tenant_id = row[4]
        person_name.append(name1)
        person_feature.append(feature)
        user_id_list.append(user_id)
        print(name1, user_id, audio_path, tenant_id)
    # audios = os.listdir(audio_db_path)
    # for audio in audios:
    #     path = os.path.join(audio_db_path, audio)
    #     name = audio[:-4]
    #     feature = qw(path)[0]
    #     person_name.append(name)
    #     person_feature.append(feature)
    #     print(f"Loaded {name} audio.")


# 参数例子：
# {
#     "audio": "C:/jupyterNoteBook/VoiceprintRecognition-Pytorch/audio_db/z.wav"
# }
@app.route("/get_feature", methods=['POST', 'GET'])
def get_feature():
    data = request_parse(request)
    audio_path = data.get("audio")
    if isinstance(audio_path, list):
        audio_path = audio_path[0]
    audio_path.replace("/", "\\")
    feature = infer(audio_path)[0]
    return " ".join(str(i) for i in feature.tolist())


@app.route("/distinguish", methods=['POST', 'GET'])
def distinguish():
    user_id = -1
    data = request_parse(request)
    audio_path = data.get("audio")
    if isinstance(audio_path, list):
        audio_path = audio_path[0]
    audio_path.replace("/", "\\")
    name2, p, i = recognition(audio_path)
    if p > args.threshold:
        print(f"识别说话的为：{name2}，相似度为：{p}")
        user_id = user_id_list[i]
    else:
        print(f"音频库没有该用户的语音{name2}，相似度为：{p}")
        user_id = 0
    return str(user_id)


@app.route("/load_db", methods=['POST', 'GET'])
def load_db():
    load_audio_db(args.audio_db)
    return str(1)


@app.route('/Hello')
def hello_world():
    return 'Hello World!'


if __name__ == '__main__':
    load_audio_db(args.audio_db)
    # record_audio = RecordAudio()

    app.run(host="127.0.0.1", port=9001, debug=True)
