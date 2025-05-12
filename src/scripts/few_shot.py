import json

import albumentations as A
import rich.progress as rprogress
import torch
import wandb
from rich.traceback import install
from rich import print

wandb.login()

from src.callbacks import ModelCheckpoint
from src.datasets import MultiDomainDataset, DomainRole, PreprocessingPipeline
from src.models import MLDG, BaseLearner, Encoder, Classifier, ERM

install(show_locals=False)

import argparse

parser = argparse.ArgumentParser(description="Training script")

parser.add_argument("--checkpoint_baseline", type=str, default="dump/debug_model.ckpt", help="Checkpoint to load")
parser.add_argument("--checkpoint_model", type=str, default="dump/debug_model.ckpt", help="Checkpoint to load")

parser.add_argument("--seed", type=int, default=0, help="Seed")
parser.add_argument("--num_classes", type=int, default=7, help="Number of classes in dataset")
parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs to train for")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
parser.add_argument("--source_domains", type=int, nargs="+", default=[3], help="Source domains to use")

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

torch.manual_seed(args.seed)
print(f"A seed was set! Seed: {args.seed}")

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

# Flush gpu
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device selected: {device}")
print(f"Found processes running with GPU {device}: {torch.cuda.list_gpu_processes(device)}")

domains = ["art_painting", "cartoon", "photo", "sketch"]
roles = [
    DomainRole.SOURCE if i in args.source_domains else DomainRole.TARGET
    for i in range(len(domains))
]
print(f"Domain roles: {list(zip(domains, roles))}")
# roles = [DomainRole.SOURCE, DomainRole.SOURCE, DomainRole.SOURCE, DomainRole.TARGET]

dataset = MultiDomainDataset(
    root="data/PACS/",
    domains=domains,
    roles=roles,
    transforms=[pipeline] * len(domains),
    seed=0,
    split_ratio=0.80,
)


def init_baseline(device, args):
    baseline = ERM(
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
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        }
    )

    baseline.load_state_dict(
        state_dict=torch.load(
            args.checkpoint_baseline,
        )
    )

    return baseline


steps_per_epoch, loaders = dataset.build_source_dataloaders(
    use_splits=False, batch_size=args.batch_size, num_workers=8,
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

few_shot_samples = [0, 1, 2, 4 , 8, 16, 32, 64, 128]
few_shot_results = {
    "K": [],
    "ACC": [],
    "BASELINE": [],
}

for K in few_shot_samples:

    task = progress_bar.add_task(
        description=f"Training (K={K})",
        total=K,
        metrics=f"loss: -- • acc: --",
    )
    few_shot_counter = 0

    baseline = init_baseline(device, args)
    baseline.train()

    with progress_bar:
        for batch in zipped:
            if few_shot_counter == K:
                break
            if few_shot_counter > K:
                raise ValueError(f"Trained with more samples than expected."
                                 f" K: {K} -> Counter: {few_shot_counter}")

            assert len(batch) == 1, "Please, only a single source domain for few-shot training"

            # print(len(batch), type(batch))
            # print(len(batch[0]), type(batch[0]))

            remaining = K - few_shot_counter
            if remaining < args.batch_size:
                batch = batch[0]
                idx, x, y = batch[0], batch[1], batch[2]
                x = x[:remaining]
                y = y[:remaining]
                batch = (idx, x, y)
            else:
                batch = batch[0]

            few_shot_counter += batch[1].shape[0]  # track num samples used for training

            # print(len(batch), type(batch))
            # print(batch[0].shape, batch[1].shape, batch[2].shape)

            loss = baseline.update((batch, ) )  # TODO: check if tuple wrapper is needed
            acc = baseline.log_metrics((batch, ))

            progress_bar.update(
                task,
                completed=few_shot_counter,
                metrics=f"loss: {loss:.4f} • acc: {acc:.2%}",
            )

        baseline = baseline.to("cpu")
        baseline.eval()
        few_shot_results["BASELINE"].append(baseline)
        few_shot_results["K"].append(K)

# reverse roles
roles = [
    DomainRole.TARGET if i in args.source_domains else DomainRole.SOURCE
    for i in range(len(domains))
]
for i in range(len(domains)):
    dataset.set_role(i, roles[i])

loaders = dataset.build_target_dataloaders(
    use_splits=False, batch_size=args.batch_size, num_workers=8,
    shuffle=False, persistent_workers=True
)

assert len(loaders) == 1, "Please, only a single target domain for few-shot testing"

progress_bar = rprogress.Progress(
    rprogress.SpinnerColumn(),
    *rprogress.Progress.get_default_columns(),
    rprogress.TimeElapsedColumn(),
    rprogress.TextColumn("{task.fields[metrics]}", justify="right"),
)
progress_bar.start()

tasks = [
    progress_bar.add_task(
        description=f"Testing baseline (K={few_shot_results["K"][i]})",
        total=len(loaders[0]),
        metrics=f"acc: --",
    )
    for i in range(len(few_shot_samples))
]

for task, baseline in zip(tasks, few_shot_results["BASELINE"]):
    acc = 0
    count = 0
    baseline = baseline.to(device)
    baseline.device = device
    baseline.eval()
    for batch in loaders[0]:
        acc_ = baseline.log_metrics((batch, ))

        count += 1
        acc += acc_

        progress_bar.update(
            task,
            advance=1,
            metrics=f"acc: {acc_:.2%}",
        )

    acc /= count
    del baseline

    progress_bar.update(
        task,
        metrics=f"acc: {acc:.2%}",
    )

    few_shot_results["ACC"].append(acc.item())
few_shot_results.pop("BASELINE")

with open("few_shot_results.json", "w") as f:
    json.dump(few_shot_results, f)

# Test

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

model.eval()
model.load_state_dict(
    state_dict=torch.load(
        args.checkpoint_model,
    )
)

task = progress_bar.add_task(
    description=f"Testing model",
    total=len(loaders[0]),
    metrics=f"acc: --",
)

acc = 0
count = 0
for batch in loaders[0]:
    acc_ = model.log_metrics((batch, ))

    count += 1
    acc += acc_

    progress_bar.update(
        task,
        advance=1,
        metrics=f"acc: {acc_:.2%}",
    )

acc /= count

progress_bar.update(
    task,
    metrics=f"acc: {acc:.2%}",
)

few_shot_results["ACC_MODEL"] = acc.item()


with open(f"few_shot_temp/few_shot_results_complete_{args.seed}.json", "w") as f:
    json.dump(few_shot_results, f)

progress_bar.stop()
