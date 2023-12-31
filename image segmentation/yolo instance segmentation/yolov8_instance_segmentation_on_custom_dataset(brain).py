# -*- coding: utf-8 -*-
"""yolov8-instance-segmentation-on-custom-dataset(brain).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1qYP3mLPSGuChfzN3vmsLBWpWk1pt4pz1
"""

!nvidia-smi

import os
HOME = os.getcwd()
print(HOME)

# Pip install method (recommended)

!pip install ultralytics==8.0.28

from IPython import display
display.clear_output()

import ultralytics
ultralytics.checks()

# Git clone method (for development)

# %cd {HOME}
# !git clone github.com/ultralytics/ultralytics
# %cd {HOME}/ultralytics
# !pip install -e .

# from IPython import display
# display.clear_output()

# import ultralytics
# ultralytics.checks()

from ultralytics import YOLO

from IPython.display import display, Image

from google.colab import drive
drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
!mkdir {HOME}/datasets
# %cd {HOME}/datasets

!pip install roboflow --quiet
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="dLRZAgJmFiZR3V2thQxr")
project = rf.workspace("rmk-engineering-college-xfiwz").project("segmentation-of-tumor")
dataset = project.version(2).download("yolov5")

"""## Custom Training"""

# Commented out IPython magic to ensure Python compatibility.
# %cd {HOME}

!yolo task=segment mode=train model=yolov8s-seg.pt data={dataset.location}/data.yaml epochs=10 imgsz=640

!ls {HOME}/runs/segment/train/

from PIL import Image
import requests
from io import BytesIO
import numpy as np

# Commented out IPython magic to ensure Python compatibility.
# %cd {HOME}
display(Image(filename=f'{HOME}/runs/segment/train/confusion_matrix.png', width=600))

from IPython.display import Image, display

# Assuming the HOME variable contains the path to the home directory
image_path = f'{HOME}/runs/segment/train/results.png'

# Display the image
display(Image(filename=image_path, width=600))

# Commented out IPython magic to ensure Python compatibility.
# %cd {HOME}
Image(filename=f'{HOME}/runs/segment/train/BoxF1_curve.png', width=600)

"""## Validate Custom Model"""

# Commented out IPython magic to ensure Python compatibility.
# %cd {HOME}

!yolo task=segment mode=val model={HOME}/runs/segment/train/weights/best.pt data={dataset.location}/data.yaml

"""## Inference with Custom Model"""

# Commented out IPython magic to ensure Python compatibility.
# %cd {HOME}
!yolo task=segment mode=predict model={HOME}/runs/segment/train/weights/best.pt conf=0.25 source={dataset.location}/test/images save=true

import glob
from IPython.display import Image, display

for image_path in glob.glob(f'{HOME}/runs/segment/predict/*.jpg')[:3]:
      display(Image(filename=image_path, height=200))
      print("\n")

!yolo task=segment mode=predict model='/content/runs/segment/train/weights/best.pt' conf=0.25 source='/content/Hirnmetastase_MRT-T1_KM.jpg' save= true

from IPython.display import Image, display

# Assuming the HOME variable contains the path to the home directory
image_path = f'/content/Hirnmetastase_MRT-T1_KM.jpg'
print("TEST IMAGE")
# Display the image
display(Image(filename=image_path, width=200))

from IPython.display import Image, display

# Assuming the HOME variable contains the path to the home directory
image_path = f'/content/runs/segment/predict2/Hirnmetastase_MRT-T1_KM.jpg'
print("SEGMENTED IMAGE")
# Display the image
display(Image(filename=image_path, width=200))

import torch
model = torch.load('/content/best.pt')

import os, random
# test_set_loc = dataset.location + "/content/datasets/cheque-1/test"
# random_test_image = random.choice(os.listdir(test_set_loc))
random_test_image = "/content/Hirnmetastase_MRT-T1_KM.jpg"
print("running inference on " + random_test_image)
pred = model.predict(random_test_image)
pred

print(model.predict("/content/Hirnmetastase_MRT-T1_KM.jpg"))


# save an image annotated with your predictions

import torch
model = YOLO('/content/best.pt')

