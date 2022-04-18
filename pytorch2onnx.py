# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import os
import argparse
from models.pfld import PFLDInference
from torch.autograd import Variable
import torch
import onnxsim

parser = argparse.ArgumentParser(description='pytorch2onnx')
parser.add_argument('--torch_model',
                    default="./checkpoint/snapshot/checkpoint.pth.tar")
parser.add_argument('--onnx_model', default="./output/pfld.onnx")
parser.add_argument('--onnx_model_sim',
                    help='Output ONNX model',
                    default="./output/pfld-sim.onnx")
args = parser.parse_args()

print("=====> load pytorch checkpoint...")
checkpoint = torch.load(args.torch_model, map_location=torch.device('cpu'))
pfld_backbone = PFLDInference()
pfld_backbone.load_state_dict(checkpoint['pfld_backbone'])
print("PFLD bachbone:", pfld_backbone)

print("=====> convert pytorch model to onnx...")
dummy_input = Variable(torch.randn(1, 3, 112, 112))
input_names = ["input_1"]
output_names = ["output_1"]
torch.onnx.export(pfld_backbone,
                  dummy_input,
                  args.onnx_model,
                  verbose=True,
                  input_names=input_names,
                  output_names=output_names)

print("====> check onnx model...")
import onnx
model = onnx.load(args.onnx_model)
onnx.checker.check_model(model)


print("====> Simplifying...")
# model_opt = onnxsim.simplify(args.onnx_model)
model_opt, check = onnxsim.simplify(args.onnx_model)
# print("model_opt", model_opt)
onnx.save(model_opt, args.onnx_model_sim)
print("onnx model simplify Ok!")

import cv2
import numpy as np
import onnxruntime

# onnxruntime


input_img = cv2.imread('face.png').astype(np.float32)/255
# HWC to NCHW
input_img = np.transpose(input_img, [2, 0, 1])
input_img = np.expand_dims(input_img, 0)

ort_session = onnxruntime.InferenceSession("./output/pfld-sim.onnx")
input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name

ort_inputs = {input_name: input_img}
# ort_inputs = input_img
ort_output = ort_session.run(["409"], ort_inputs)[0]
landmarks = ort_output
pre_landmark = landmarks[0]

pre_landmark = pre_landmark.reshape(-1,2)*112

input_img = np.squeeze(input_img,axis=0)
input_img = np.transpose(input_img, [1, 2, 0])*255

for (x, y) in pre_landmark.astype(np.int32):
    cv2.circle(input_img, (x, y), 1, (0, 0, 255))

cv2.imwrite('face_landmark_68.png', input_img)