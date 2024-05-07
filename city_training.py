import torch
from torchvision.datasets import Cityscapes
import torchvision.models.segmentation as segmentation
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.classification import MulticlassJaccardIndex as IoU

import city_config
import city_utils


class cityscape_deeplab(pl.LightningModule):
    def __init__(self, n_classes):
        super(cityscape_deeplab, self).__init__()

        self.whole_model = segmentation.deeplabv3_resnet50(weight="default", num_classes=n_classes)
        self.criterion = smp.losses.DiceLoss(mode='multiclass')
        self.metrics = IoU(num_classes=n_classes)

    def forward(self, x):
        # print("\nin forward\n")
        return self.whole_model(x)

    def training_step(self, batch, batch_idx):
        # print("\nin training step start\n")
        images, semantic_masks = batch # float32, float32
        
        # Forward pass
        outputs = self(images) # float32
        outputs = outputs["out"] # (b, nclass, w, h)
        
        loss = self.criterion(outputs, semantic_masks.long())
        iou = self.metrics(outputs, semantic_masks)

        self.log_dict({'train_loss': loss, 'train_iou': iou})
        # print("\nin training step end\n")
        return {"loss": loss, "score": iou}

    # define what happens for testing here

    def train_dataloader(self):
        # print("\nin train_dataloader start\n")
        train_dataset = Cityscapes(city_config.dataset_path, split='train', mode='fine',
                                   target_type='semantic', transform=city_utils.convert_input_images,
                                   target_transform=city_utils.convert_input_masks)
        # Data loader
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=city_config.batch_size,
                                                   num_workers=4, shuffle=True, persistent_workers=True)
        
        # print("\nin train_dataloader end\n")
        return train_loader
    
    def validation_step(self, batch, batch_idx):
        # print("\nin validation_step start\n")
        images, semantic_masks = batch

        # Forward pass
        outputs = self(images) # float32
        outputs = outputs["out"] # (b, nclass, w, h)

        val_loss = self.criterion(outputs, semantic_masks.long())
        val_iou = self.metrics(outputs, semantic_masks)

        self.log_dict({"val_loss": val_loss, "val_iou": val_iou})

        # print("\nin validation_step end\n")
        return {"val_loss": val_loss, "val_iou": val_iou}

    def val_dataloader(self):
        # print("\nin val_dataloader start\n")
        val_dataset = Cityscapes(city_config.dataset_path, split='val', mode='fine',
                                 target_type='semantic', transform=city_utils.convert_input_images,
                                 target_transform=city_utils.convert_input_masks)

        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=city_config.batch_size,
                                                 num_workers=4, shuffle=False, persistent_workers=True)
        
        # print("\nin val_dataloader end\n")
        return val_loader
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=city_config.learning_rate)



if __name__ == '__main__':
    logger = TensorBoardLogger("tb_logs", name="cityscape_deeplab")
    model = cityscape_deeplab(city_config.n_classes)

    # fast_dev_run=True -> runs single batch through training and validation
    if torch.cuda.is_available():
        trainer = Trainer(logger=logger, max_epochs=city_config.num_epochs, precision='16-mixed')
    else:
        trainer = Trainer(logger=logger, max_epochs=city_config.num_epochs)
    # trainer = Trainer(fast_dev_run=True)

    print("\n------Cityscapes training started------\n")
    trainer.fit(model)
    print("\n------Cityscapes training completed------\n")
