import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.onnx import SamOnnxModel

import onnxruntime
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic

checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=checkpoint)
sam.to(device='mps')
predictor = SamPredictor(sam)

image = cv2.imread('demo/src/assets/data/oc-harley.jpeg')
predictor.set_image(image)
image_embedding = predictor.get_image_embedding().cpu().numpy()

try:
    np.save("demo/src/assets/data/oc-harley.npy", image_embedding)
    print("Saved image embedding.")
except Exception as e:
    print("Error saving image embedding:", e)