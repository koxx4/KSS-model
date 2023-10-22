import os
import dotenv
import comet_ml
import torch
from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model
from roboflow import Roboflow
from ultralytics import YOLO

dotenv.load_dotenv()

ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY")
COMETML_API_KEY = os.environ.get("COMETML_API_KEY")

DATASET_VERSION = 7
EXPERIMENT_NAME = f'YOLOV5NU-V{DATASET_VERSION}-PCM-2-q'
DATA_YAML_FILE = f'data-kss-{DATASET_VERSION}.yaml'
DATASET_FOLDER = f'Kitchen-Safety-System-{DATASET_VERSION}'

rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace("diploma-2023").project("kitchen-safety-system")
dataset = project.version(DATASET_VERSION).download("yolov8", location=f"./datasets/{DATASET_FOLDER}", overwrite=False)

if not torch.cuda.is_available():
    exit(1)
torch.cuda.get_device_name(0)

comet_ml.init(api_key=COMETML_API_KEY, project_name="KSS")
experiment = Experiment()
experiment.set_name(EXPERIMENT_NAME)

# Load a model
model = YOLO("yolov5nu.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data=DATA_YAML_FILE, epochs=300, batch=136, imgsz=640, optimizer="auto", lr0=0.03)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
path = model.export(format="onnx", imgsz=640, half=True, simplify=True)  # export the model to ONNX format

log_model(experiment, model, model_name=EXPERIMENT_NAME)
