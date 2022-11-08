# coding=utf-8
import sys

sys.path.append("C://jupyterNoteBook/VoiceprintRecognition-Pytorch/")
from recognition_judge import infer


def add1():

    sum1 = infer(sys.argv[2])
    print(sum1)
    return sum1


add1()
