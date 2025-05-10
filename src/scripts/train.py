import albumentations as A
import rich.progress as rprogress
import torch
from rich.traceback import install

import wandb
wandb.login()

from src.callbacks import ModelCheckpoint
from src.datasets import MultiDomainDataset, DomainRole, PreprocessingPipeline
from src.models import MLDG, BaseLearner, Encoder, Classifier

install(show_locals=False)

import argparse

parser = argparse.ArgumentParser(description="Training script")

parser.add_argument("--num_classes", type=int, default=7, help="Number of classes in dataset")
parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs to train for")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size")


parser.add_argument("--nonlinear_classifier", type=bool, default=False, help="Classifier architecture")
parser.add_argument("--dropout", type=float, default=0., help="Dropout rate to use in classifier head")

parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=0., help="Weight decay")
parser.add_argument("--beta", type=float, default=1.0, help="Controls importance of meta-test gradients")

args = parser.parse_args()

run = wandb.init(
    project="MetaLearning_ComputerVisionProject",
    config=vars(args),
)

# Backend: albumentations
transform = A.Compose([
    A.Resize(height=224, width=224),
    A.Normalize(
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    ),
    A.ToTensorV2(),
])

augment_transform = A.Compose([
    A.RandomResizedCrop(size=(224, 224), scale=(0.7, 1.0), p=1.0),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3, p=0.5),
    A.ToGray(num_output_channels=3, p=0.10),
    A.Normalize(
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    ),
    A.ToTensorV2(),
])


def wrapper_albumentations_transform(transform: A.Compose):
    def transform_(img):
        img_ = img.permute(1, 2, 0).numpy()
        try:
            return transform(image=img_).get("image")
        except Exception as error:
            print(f"Failed to apply transform at image {img_.shape}, {img_.dtype}")
            raise error

    return transform_

pipeline = PreprocessingPipeline(
    source_transform=wrapper_albumentations_transform(augment_transform),
    target_transform=wrapper_albumentations_transform(transform),
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device selected: {device}")

domains = ["art_painting", "cartoon", "photo", "sketch"]
roles = [DomainRole.SOURCE, DomainRole.SOURCE, DomainRole.SOURCE, DomainRole.TARGET]

dataset = MultiDomainDataset(
    root="data/PACS/",
    domains=domains,
    roles=roles,
    transforms=[pipeline] * len(domains),
    seed=0,
    split_ratio=0.80,
)

model = MLDG(
    network=BaseLearner(
        encoder=Encoder(hparams={}),
        classifier=Classifier(hparams={
            "num_feats": 2048,
            "num_classes": args.num_classes,
            "dropout": args.dropout,
            "nonlinear": args.nonlinear_classifier,
        })
    ),
    device=device,
    num_classes=args.num_classes,
    num_domains=len([role for role in roles if role == DomainRole.SOURCE]),
    hparams={
        "num_meta_test": 1,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "beta": args.beta,
        "lr_clone": args.lr,  # TODO: for now, using same value for both
        "weight_decay_clone": args.weight_decay,
    }
)

steps_per_epoch, loaders = dataset.build_source_dataloaders(
    use_splits=False, batch_size=16, num_workers=8,
    shuffle=True, persistent_workers=True
)
zipped = zip(*loaders)

checkpoint_callback = ModelCheckpoint(
    save_name=wandb.run.name,
    save_path="dump",
    mode="max",
)

logging_interval = 1
steps = 0
epochs = 0
progress_bar = rprogress.Progress(
    rprogress.SpinnerColumn(),
    *rprogress.Progress.get_default_columns(),
    rprogress.TimeElapsedColumn(),
    rprogress.TextColumn("{task.fields[metrics]}", justify="right"),
)

# Train
model.train()

task = progress_bar.add_task(
    description="Training",
    total=args.num_epochs,
    metrics=f"loss: -- • acc: --",
)

max_steps = steps_per_epoch * args.num_epochs
with progress_bar:
    while steps < max_steps:
        for batch in zipped:
            steps += 1
            epochs += 1 / steps_per_epoch

            loss = model.update(batch)
            acc = model.log_metrics(batch)

            progress_bar.update(
                task,
                advance=1 / steps_per_epoch,
                metrics=f"loss: {loss:.4f} • acc: {acc:.2%}",
            )

            wandb.log({"train/loss": loss, "train/acc": acc})

            checkpoint_callback.eval(
                module=model,
                score=acc,
            )

            if steps >= max_steps:
                break

