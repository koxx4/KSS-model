import os
import dotenv
import comet_ml
import torch
import torch.utils.data
from roboflow import Roboflow
from torch import optim
import torch.utils.data
import torchvision
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights, fasterrcnn_mobilenet_v3_large_320_fpn, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, ModelPruning
from pytorch_lightning.loggers import CometLogger
from torchmetrics.detection import MeanAveragePrecision
import statistics
from PIL import Image
from pycocotools.coco import COCO

dotenv.load_dotenv()

ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY")
COMETML_API_KEY = os.environ.get("COMETML_API_KEY")

inputsz = 640
num_classes = 12  # 11 + background
num_epochs = 100
lr = 0.001
momentum = 0.997
weight_decay = 0.0005

rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace("diploma-2023").project("kitchen-safety-system")
dataset = project.version(7).download(
    "coco", location="./datasets/Kitchen-Safety-System-7-COCO", overwrite=False)

EXPERIMENT_NAME = 'FASTERRCNN.MOBILENETV3-320-V7-PCM-1'

comet_logger = CometLogger(api_key=COMETML_API_KEY,
                           experiment_name=EXPERIMENT_NAME, project_name="KSS")

MAX_OBJ_IN_IMG = 300


class RipoDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.test_img_next_iterator = 0

    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]['file_name']
        # open the input image
        img = Image.open(os.path.join(self.root, path))

        # number of objects in the image
        num_objs = min(len(coco_annotation), MAX_OBJ_IN_IMG)

        # Annotation is in dictionary format
        my_annotation = {}

        if num_objs > 0:

            boxes = []
            labels = []
            areas = []

            for i in range(num_objs):
                xmin = coco_annotation[i]['bbox'][0]
                ymin = coco_annotation[i]['bbox'][1]
                xmax = xmin + coco_annotation[i]['bbox'][2]
                ymax = ymin + coco_annotation[i]['bbox'][3]
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(coco_annotation[i]['category_id'])

            for i in range(num_objs):
                areas.append(coco_annotation[i]['area'])

            areas = torch.as_tensor(areas, dtype=torch.float32)
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            img_id = torch.tensor([img_id])
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

            my_annotation["boxes"] = boxes
            my_annotation["labels"] = labels
            my_annotation["image_id"] = img_id
            my_annotation["area"] = areas
            my_annotation["iscrowd"] = iscrowd

        else:
            # If there are no objects in the photo
            my_annotation = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.int64),
                "image_id": torch.tensor([img_id]),
                "area": torch.zeros(0, dtype=torch.float32),
                "iscrowd": torch.zeros((0,), dtype=torch.int64)
            }

        if self.transforms is not None:
            img = self.transforms(img)

        img = torch.as_tensor(img, dtype=torch.float32)

        return img, my_annotation, path

    def __len__(self):
        return len(self.ids)


TRAIN_BATCH_SIZE = 128
VALIDATION_BATCH_SIZE = 128
TRAIN_DATA_DIR = 'datasets/Kitchen-Safety-System-7-COCO/train'
TRAIN_COCO = 'datasets/Kitchen-Safety-System-7-COCO/train/_annotations.coco.json'
VAL_DATA_DIR = 'datasets/Kitchen-Safety-System-7-COCO/valid/'
VAL_COCO = 'datasets/Kitchen-Safety-System-7-COCO/valid/_annotations.coco.json'
PRED_COCO = "./predictions.json"

comet_logger.log_hyperparams({
    "train_batch_size": TRAIN_BATCH_SIZE,
    "val_batch_size": VALIDATION_BATCH_SIZE,
    "max_obj_in_img": MAX_OBJ_IN_IMG,
    "lr": lr,
    "weight_decay": weight_decay,
    "momentum": momentum,
    "imgsz": inputsz,
    "num_classes": num_classes,
    "num_epochs": num_epochs
})


def collate_fn(batch):
    return tuple(zip(*batch))

device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

if torch.cuda.is_available():
    print(f'Masz kompatybilne GPU: {torch.cuda.get_device_name(0)}')
    torch.cuda.empty_cache()

def get_model_instance():
    # Step 1: Initialize model with the best available weights
    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1
    model = fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=weights, box_score_thresh=0.8)

    return model, weights.transforms()

class FasterRCNNMobileNetV3(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.save_hyperparameters()
        self.mean_ap_metric_50 = MeanAveragePrecision(
            iou_type="bbox", iou_thresholds=[0.5])
        self.mean_ap_metric_75 = MeanAveragePrecision(
            iou_type="bbox", iou_thresholds=[0.75])
        self.mean_ap_metric_50_95 = MeanAveragePrecision(iou_type="bbox", iou_thresholds=[
                                                         0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
        self.automatic_optimization = True

    def forward(self, inputs, target):

        input_images = list(img.to(device) for img in inputs)
        labels = [{k: v.to(device) for k, v in t.items()} for t in target]

        return self.model(input_images, labels)

    def training_step(self, batch, batch_idx):

        inputs, labels, _ = batch

        outputs = self(inputs=inputs, target=labels)

        loss_cls = outputs['loss_classifier'].item()
        loss_box_reg = outputs['loss_box_reg'].item()
        loss_obj = outputs['loss_objectness'].item()
        loss_rpn_box_reg = outputs['loss_rpn_box_reg'].item()
        loss = self.sum_losses(outputs)

        self.log("train/box_loss", loss_box_reg,
                 prog_bar=True, batch_size=TRAIN_BATCH_SIZE)
        self.log("train/cls_loss", loss_cls, prog_bar=True,
                 batch_size=TRAIN_BATCH_SIZE)
        self.log("train/obj_loss", loss_obj, prog_bar=True,
                 batch_size=TRAIN_BATCH_SIZE)
        self.log("train/rpn_loss", loss_rpn_box_reg,
                 prog_bar=True, batch_size=TRAIN_BATCH_SIZE)
        self.log("train/loss", loss, prog_bar=True,
                 batch_size=TRAIN_BATCH_SIZE)

        metrics = {
            "train/box_loss": loss_box_reg,
            "train/cls_loss": loss_cls,
            "train/obj_loss": loss_obj,
            "train/rpn_loss": loss_box_reg,
            "train/loss": loss
        }

        self.logger.log_metrics(metrics, step=batch_idx)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels, _ = batch
        input_images = list(img.to(device) for img in inputs)
        labels = [{k: v.to(device) for k, v in t.items()} for t in labels]

        with torch.no_grad():
            preds = self(inputs=input_images, target=labels)
            self.mean_ap_metric_50(preds, labels)
            self.mean_ap_metric_75(preds, labels)
            self.mean_ap_metric_50_95(preds, labels)

        outputs = self(inputs=input_images, target=labels)
        scores = [{score for score in img['scores']} for img in outputs]
        scores = list(filter(lambda x: (len(x) > 0), scores))

        if scores:
            scores = [s.item() for s in scores[0]]

        score_mean_val = statistics.mean(scores) if len(scores) > 0 else 0.0

        self.logger.log_metrics(
            {"val/score_mean_val": score_mean_val}, step=batch_idx)

        self.log("val/score_mean_val", score_mean_val, prog_bar=True)

    def on_validation_epoch_end(self):
        mAP_50 = self.mean_ap_metric_50.compute()['map_50']
        mAP_75 = self.mean_ap_metric_75.compute()['map_75']
        mAP_50_95 = self.mean_ap_metric_50_95.compute()['map']

        self.log("metrics/mAP50(B)", mAP_50, prog_bar=True,
                 batch_size=VALIDATION_BATCH_SIZE)
        self.log("metrics/mAP75(B)", mAP_75, prog_bar=True,
                 batch_size=VALIDATION_BATCH_SIZE)
        self.log("metrics/mAP50-95(B)", mAP_50_95,
                 prog_bar=True, batch_size=VALIDATION_BATCH_SIZE)

        metrics = {
            "metrics/mAP50(B)": mAP_50,
            "metrics/mAP75(B)": mAP_75,
            "metrics/mAP50-95(B)": mAP_50_95
        }

        self.logger.log_metrics(metrics=metrics, step=self.global_step)

        # Resetowanie metryki na początek nowej epoki
        self.mean_ap_metric_50.reset()
        self.mean_ap_metric_75.reset()
        self.mean_ap_metric_50_95.reset()

    def sum_losses(self, output):
        return sum(loss for loss in output.values())

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=lr,
                                weight_decay=weight_decay)
        return optimizer


torch.set_float32_matmul_precision('low')
model_instance, model_transforms = get_model_instance()

autoencoder = FasterRCNNMobileNetV3(model=model_instance)

img_transforms = torchvision.transforms.Compose([
    model_transforms
])

# Creating datasets for training and validation
ripo_train_dataset = RipoDataset(
    TRAIN_DATA_DIR,
    TRAIN_COCO,
    transforms=img_transforms)

ripo_validation_dataset = RipoDataset(
    VAL_DATA_DIR,
    VAL_COCO,
    transforms=img_transforms)

train_data_loader = torch.utils.data.DataLoader(ripo_train_dataset,
                                                batch_size=TRAIN_BATCH_SIZE,
                                                shuffle=False,
                                                num_workers=2,
                                                collate_fn=collate_fn)

validation_data_loader = torch.utils.data.DataLoader(ripo_validation_dataset,
                                                     batch_size=VALIDATION_BATCH_SIZE,
                                                     shuffle=False,
                                                     num_workers=2,
                                                     collate_fn=collate_fn)

checkpoint_callback = ModelCheckpoint(
    monitor='train/loss',
    filename='kss-frcnnmnv3-320-v7-{epoch:02d}-{train_loss:.2f}',
    save_top_k=3,
    mode='min',
)

trainer = pl.Trainer(enable_progress_bar=True, max_epochs=num_epochs, logger=comet_logger, callbacks=[checkpoint_callback])

trainer.fit(model=autoencoder, train_dataloaders=train_data_loader,
            val_dataloaders=validation_data_loader)
