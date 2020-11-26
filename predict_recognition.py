import argparse
import os
import time
import numpy as np
import pyaudio
import tensorflow
from record_demo import get_voice

import random
from audio import read_mfcc
from batcher import sample_from_mfcc
from constants import SAMPLE_RATE, NUM_FRAMES
from conv_models import DeepSpeakerModel
from test import batch_cosine_similarity

np.random.seed(123)
random.seed(123)


parser = argparse.ArgumentParser()
# # set up training configuration.
# parser.add_argument('--n_classes', default=5994, type=int, help='class dim number')
parser.add_argument('--audio_db', default='audio_db/', type=str, help='person audio database')
# parser.add_argument('--resume', default=r'pretrained/weights.h5', type=str, help='resume model path')
# # set up network configuration.
# parser.add_argument('--net', default='resnet34s', choices=['resnet34s', 'resnet34l'], type=str)
# parser.add_argument('--ghost_cluster', default=2, type=int)
# parser.add_argument('--vlad_cluster', default=8, type=int)
# parser.add_argument('--bottleneck_dim', default=512, type=int)
# parser.add_argument('--aggregation_mode', default='gvlad', choices=['avg', 'vlad', 'gvlad'], type=str)
# # set up learning rate, training loss and optimizer.
# parser.add_argument('--loss', default='softmax', choices=['softmax', 'amsoftmax'], type=str)
args = parser.parse_args()

person_feature = []
person_name = []

# 減少显存占用
config = tensorflow.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
_ = tensorflow.compat.v1.Session(config=config)
# ==================================
#       Get Model
# ==================================
# construct the data generator.
# params = {'dim': (257, None, 1),
#           'nfft': 512,
#           'spec_len': 250,
#           'win_length': 400,
#           'hop_length': 160,
#           'n_classes': args.n_classes,
#           'sampling_rate': 16000,
#           'normalize': True}
#
# network_eval = model.vggvox_resnet2d_icassp(input_dim=params['dim'],
#                                             num_class=params['n_classes'],
#                                             mode='eval', args=args)

# ==> load pre-trained model
# network_eval.load_weights(os.path.join(args.resume), by_name=True)
model = DeepSpeakerModel()
model.m.load_weights('/home/ubuntu/PycharmProjects/deep-speaker/checkpoints/ResCNN_triplet_training_checkpoint_265.h5', by_name=True)
print('==> successfully loading model {}.')


# 预测获取声纹特征
def predict(input_filename):
    mfcc = sample_from_mfcc(read_mfcc(input_filename, SAMPLE_RATE), NUM_FRAMES)
    predict_fea = model.m.predict(np.expand_dims(mfcc, axis=0))
    return predict_fea


# 加载要识别的音频库
def load_audio_db(audio_db_path):
    start = time.time()
    audios = os.listdir(audio_db_path)
    for audio in audios:
        # path = os.path.join(audio_db_path, audio)
        name = audio[:-4]
        predict_fea = predict('audio_db/' + audio)
        person_name.append(name)
        person_feature.append(predict_fea)
        print("Loaded %s audio." % name)
    end = time.time()
    print('加载音频库完成，消耗时间：%fms' % (round((end - start) * 1000)))


# 识别声纹
def recognition(audio):
    name = ''
    pro = 0
    predict_fea = predict(audio)
    for i, person_f in enumerate(person_feature):
        # 计算相识度
        dist = batch_cosine_similarity(predict_fea, person_f)
        # dist = np.dot(feature, person_f.T)
        if dist > pro:
            pro = dist
            name = person_name[i]
    return name, pro


def start_recognition():
    # 录音参数
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 16000
    RECORD_SECONDS = 4
    WAVE_OUTPUT_FILENAME = "infer_audio.wav"

    while True:
        i = input("按下回车键开机录音，录音%s秒中：" % RECORD_SECONDS)
        print("开始录音......")
        get_voice()
        """
        # 打开录音
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
        i = input("按下回车键开机录音，录音%s秒中：" % RECORD_SECONDS)
        print("开始录音......")
        frames = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)

        print("录音已结束!")

        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        """
        print("录音已结束!")

        os.system("~/PycharmProjects/Kersa-Speaker-Recognition/noisered.sh");
        WAVE_OUTPUT_FILENAME_CLEAN = "infer_audio_clean.wav"

        # 识别对比音频库的音频
        start = time.time()
        name, p = recognition(WAVE_OUTPUT_FILENAME_CLEAN)
        end = time.time()
        # print("预测时间为：%d，识别说话的为：%s，相似度为：%f" % (round((end - start) * 1000), name, p))
        if p > 0.8:
            print("预测时间为：%d，识别说话的为：%s，相似度为：%f" % (round((end - start) * 1000), name, p))
        else:
            print("预测时间为：%d，音频库没有该用户的语音，相似度为：%f" % (round((end - start) * 1000), p))


if __name__ == '__main__':
    load_audio_db(args.audio_db)
    start_recognition()
