import argparse
import os
import sys
import time
import datetime
import json
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from pathlib import Path
from dataset import build_dataset
from engine import train_one_epoch, evaluate
from model.resnext import ReXNetV1
from model.rexnetv2 import ReXNetV2
from timm.utils import ModelEma
from timm.data import Mixup
from utils import NativeScalerWithGradNormCount as NativeScaler
import utils
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

# Import RFMiD preprocessing
rfmid_preprocess_dir = "/mnt/home/thakurim/Healthcare/scripts"
sys.path.append(rfmid_preprocess_dir)
from rfmid_preprocess import preprocess_rfmid  
from optim_factory import create_optimizer, LayerDecayValueAssigner


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def preprocess_data(args):
    """Preprocess RFMiD dataset: Extracts and structures dataset before training."""
    base_dir = args.data_path
    output_dir = os.path.join(base_dir, "processed/RFMiD")
    preprocess_rfmid(base_dir, output_dir)

    # Set correct dataset paths
    args.data_path = "/mnt/projects/zhuangyo_project/processed/RFMiD/all_classes"
    args.train_images = os.path.join(args.data_path, "1. Original Images", "a. Training Set")
    args.val_images = os.path.join(args.data_path, "1. Original Images", "b. Validation Set")
    args.train_labels = os.path.join(args.data_path, "2. Groundtruths", "training_labels.csv")
    args.val_labels = os.path.join(args.data_path, "2. Groundtruths", "validation_labels.csv")

    # Debugging prints
    print(f"DEBUG: Train Images Path -> {args.train_images}")
    print(f"DEBUG: Train Labels Path -> {args.train_labels}")
    print(f"Training images directory: {args.train_images}")
    print(f"Validation images directory: {args.val_images}")
    print(f"Training labels file: {args.train_labels}")
    print(f"Validation labels file: {args.val_labels}")

    # Ensure paths exist
    assert os.path.exists(args.train_images), f" Error: Training images not found at {args.train_images}"
    assert os.path.exists(args.val_images), f" Error: Validation images not found at {args.val_images}"
    assert os.path.exists(args.train_labels), f" Error: Training labels not found at {args.train_labels}"
    assert os.path.exists(args.val_labels), f" Error: Validation labels not found at {args.val_labels}"
    print(" Dataset paths successfully verified!")


def main(args):
    utils.init_distributed_mode(args)
    print(args)

    # Ensure update_freq is set
    if not hasattr(args, "update_freq") or args.update_freq is None:
        args.update_freq = 1  # Default value to prevent division errors

    device = torch.device(args.device)




    # Fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # Preprocess dataset
    preprocess_data(args)

    
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)
    
    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size

    if num_training_steps_per_epoch == 0 or num_training_steps_per_epoch is None:
        num_training_steps_per_epoch = 1 

    start_steps = args.start_epoch * num_training_steps_per_epoch


    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    if args.disable_eval:
        args.dist_eval = False
        dataset_val = None
    else:
        dataset_val, _ = build_dataset(is_train=False, args=args)

    if args.distributed:
        sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, seed=args.seed)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if args.dist_eval:
        if dataset_val and len(dataset_val) % num_tasks != 0:
            print("⚠️ Warning: Distributed evaluation dataset not divisible by process count!")
        sampler_val = torch.utils.data.DistributedSampler(dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    print("Sampler_train = %s" % str(sampler_train))

    # Logging setup
    log_writer = utils.TensorboardLogger(log_dir=args.log_dir) if global_rank == 0 and args.log_dir else None
    wandb_logger = utils.WandbLogger(args) if global_rank == 0 and args.enable_wandb else None

    # ✅ **Ensure DataLoader is included**
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size), num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False
    )


    for batch in data_loader_train:
        print(f"DEBUG: Batch Shape = {batch[0].shape}")  
        break


    # Initialize model
    if args.model == "rexnetv1":
        model = ReXNetV1(input_ch=3, width_mult=args.width_mult, classes=args.nb_classes, dropout_path=args.drop_path)
    elif args.model == "rexnetv2":
        model = ReXNetV2(input_ch=3, width_mult=args.width_mult, classes=args.nb_classes, drop_path=args.drop_path)
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    
    print(f"Model initialized: {model}")
    print("DEBUG: Model First Layer Configuration ->", model.features[0])
    weight_path = os.path.join(args.weights_dir, "rexnet_3.0.pth")
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=device), strict=False)
        print(f"Loaded weights from {weight_path}")
    else:
        print(f"Warning: Pretrained weights not found at {weight_path}. Training from scratch.")

    model.to(device)
    model_without_ddp = model

    # ✅ **Correctly handle DistributedDataParallel (DDP)**
    if args.distributed:
        print(f"Using DistributedDataParallel (GPU {args.gpu})")
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module  

    # ✅ **EMA**
    model_ema = None
    if args.model_ema:
        model_ema = ModelEma(model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else '', resume='')
        print(f"Using EMA with decay = {args.model_ema_decay}")

    print(f"Optimizer selected: {args.opt}")
    optimizer = create_optimizer(args, model_without_ddp, skip_list=None)

    loss_scaler = NativeScaler()

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    max_accuracy = 0.0
    for epoch in range(args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        mixup_active = args.mixup > 0 or args.cutmix > 0 or args.cutmix_minmax is not None
        if mixup_active:
            print("Mixup is activated!")
            mixup_fn = Mixup(
                mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
                prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
                label_smoothing=args.smoothing, num_classes=args.nb_classes
            )
        else:
            mixup_fn = None

        if mixup_fn is not None:
            criterion = SoftTargetCrossEntropy()
        elif args.smoothing > 0.:
            criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        else:
            criterion = torch.nn.CrossEntropyLoss()

        train_stats = train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch, loss_scaler, args.clip_grad, model_ema, mixup_fn, log_writer, wandb_logger)
        test_stats = evaluate(data_loader_val, model, device)

        if max_accuracy < test_stats["accuracy_score"]:
            max_accuracy = test_stats["accuracy_score"]
            utils.save_model(args=args, model=model, optimizer=optimizer, loss_scaler=loss_scaler, epoch="best", model_ema=model_ema)

        print(f"Max accuracy: {max_accuracy:.4f}%")

    print(f"Training time: {str(datetime.timedelta(seconds=int(time.time() - start_time)))}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser("RFMiD Training Script", add_help=False)
    parser.add_argument("--data_path", default="/mnt/projects/zhuangyo_project/data/RFMiD", type=str)
    parser.add_argument("--weights_dir", default="/mnt/home/thakurim/Healthcare/scripts/checkpoints", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--input_size", default=512, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--data_set", default="rfmid", type=str)
    parser.add_argument("--model", default="rexnetv1", type=str, choices=["rexnetv1", "rexnetv2"])
    parser.add_argument("--width_mult", default=3.0, type=float)
    parser.add_argument("--drop_path", default=0.2, type=float)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument("--dist_backend", default="nccl", type=str)
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument('--disable_eval', type=str2bool, default=False, help='Disable evaluation after training')
    parser.add_argument('--log_dir', default='Experiment/3/log', help='path where to tensorboard log')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', type=str2bool, default=False,
                        help='Perform evaluation only')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--pin_mem', type=str2bool, default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    # distributed training parameters
    parser.add_argument('--dist_eval', type=str2bool, default=True,
                        help='Enabling distributed evaluation')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', type=str2bool, default=False)
    parser.add_argument('--main_eval', default='f1', type=str,
                        help="The Save the model weight besed on which metric.")
    parser.add_argument('--use_amp', type=str2bool, default=False, 
                        help="Use PyTorch's AMP (Automatic Mixed Precision) or not")

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    
    # EMA related parameters
    parser.add_argument('--model_ema', type=str2bool, default=True)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', type=str2bool, default=False, help='')
    parser.add_argument('--model_ema_eval', type=str2bool, default=True, help='Using ema to eval during training.')
    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Optimization parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

# Weights and Biases arguments
    parser.add_argument('--enable_wandb', type=str2bool, default=False,
                        help="enable logging to Weights and Biases")
    parser.add_argument('--project', default='convnext', type=str,
                        help="The name of the W&B project where you're sending the new run.")
    parser.add_argument('--wandb_ckpt', type=str2bool, default=False,
                        help="Save model checkpoints as W&B Artifacts.")
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)









