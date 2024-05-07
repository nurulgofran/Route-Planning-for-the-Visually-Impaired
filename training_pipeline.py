import torch
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.classification import MulticlassJaccardIndex as IoU

from city_training import cityscape_deeplab
from map_dataset_loader import Mapillary
import map_config
import map_utils

class mapillary_deeplab(pl.LightningModule):
    def __init__(self, n_classes, cityscape_model):
        super(mapillary_deeplab, self).__init__()

        self.whole_model = cityscape_model
        self.criterion = smp.losses.DiceLoss(mode='multiclass')
        self.metrics = IoU(num_classes=n_classes)

    def forward(self, x):
        return self.whole_model(x)

    def training_step(self, batch, batch_idx):
        images, semantic_masks = batch # float32, float32
        
        # Forward pass
        outputs = self(images) # float32
        outputs = outputs["out"] # (b, nclass, w, h)
        
        loss = self.criterion(outputs, semantic_masks.long())
        iou = self.metrics(outputs, semantic_masks)

        self.log_dict({'train_loss': loss, 'train_iou': iou})

        return {"loss": loss, "score": iou}

    def train_dataloader(self):
        train_dataset = Mapillary(map_config.mapillary_train_path, transform=map_utils.convert_input_images,
                                   target_transform=map_utils.convert_input_masks)
        # Data loader
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=map_config.batch_size,
                                                   num_workers=4, shuffle=True, persistent_workers=True)

        return train_loader
    
    def validation_step(self, batch, batch_idx):
        images, semantic_masks = batch

        # Forward pass
        outputs = self(images) # float32
        outputs = outputs["out"] # (b, nclass, w, h)

        val_loss = self.criterion(outputs, semantic_masks.long())
        val_iou = self.metrics(outputs, semantic_masks)

        self.log_dict({"val_loss": val_loss, "val_iou": val_iou})

        return {"val_loss": val_loss, "val_iou": val_iou}

    def val_dataloader(self):
        val_dataset = Mapillary(map_config.mapillary_val_path, transform=map_utils.convert_input_images,
                                 target_transform=map_utils.convert_input_masks)

        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=map_config.batch_size,
                                                 num_workers=4, shuffle=False, persistent_workers=True)
        
        return val_loader

    def test_step(self, batch, batch_idx):
        images, semantic_masks = batch

        # Forward pass
        outputs = self(images) # float32
        outputs = outputs["out"] # (b, nclass, w, h)

        test_loss = self.criterion(outputs, semantic_masks.long())
        test_iou = self.metrics(outputs, semantic_masks)

        self.log_dict({"test_loss": test_loss, "test_iou": test_iou})

        return {"test_loss": test_loss, "test_iou": test_iou}
    
    def test_dataloader(self):
        test_dataset = Mapillary(map_config.mapillary_test_path, transform=map_utils.convert_input_images,
                                 target_transform=map_utils.convert_input_masks)

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=map_config.batch_size,
                                                 num_workers=4, shuffle=False, persistent_workers=True)
        
        return test_loader


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=map_config.learning_rate)



if __name__ == '__main__':
    logger = TensorBoardLogger("tb_logs", name="mapillary_deeplab")

    if torch.cuda.is_available():
        checkpoint = torch.load(map_config.city_ckpt_path)
    else:
        checkpoint = torch.load(map_config.city_ckpt_path, map_location=torch.device('cpu'))

    cityscape_model = cityscape_deeplab(map_config.n_classes)
    cityscape_model.load_state_dict(checkpoint["state_dict"])


    model = mapillary_deeplab(n_classes=map_config.n_classes, cityscape_model=cityscape_model)

    # fast_dev_run=True -> runs single batch through training and validation
    # trainer = Trainer(fast_dev_run=True)

    if torch.cuda.is_available():
        trainer = Trainer(logger=logger, max_epochs=map_config.num_epochs, precision='16-mixed')
    else:
        trainer = Trainer(logger=logger, max_epochs=map_config.num_epochs)


    print("\n------Mapillary training started------\n")
    trainer.fit(model)
    print("\n------Mapillary training completed------\n")
