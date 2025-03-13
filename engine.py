import math
from typing import Iterable, Optional
import torch
from timm.data import Mixup
from timm.utils import accuracy, ModelEma
import numpy as np
from sklearn import metrics
import utils


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    wandb_logger=None, start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, use_amp=False):

    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    print_freq = 10

    optimizer.zero_grad()

    for data_iter_step, (samples, _, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // (update_freq if update_freq is not None else 1)
        if num_training_steps_per_epoch is None:
            num_training_steps_per_epoch = 1
        if step >= num_training_steps_per_epoch:
            continue
        it = (start_steps if start_steps is not None else 0) + (step if step is not None else 0)

        # Update learning rate and weight decay
        if lr_schedule_values is not None or (wd_schedule_values is not None and data_iter_step % update_freq == 0):
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast(enabled=use_amp):
            output = model(samples)
            loss = criterion(output, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training.")
            assert math.isfinite(loss_value)

        loss /= update_freq
        loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.parameters(), create_graph=False)
        if (data_iter_step + 1) % update_freq == 0:
            optimizer.step()
            optimizer.zero_grad()
            if model_ema is not None:
                model_ema.update(model)

        torch.cuda.synchronize()

        if mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, use_amp=False):

    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    ground_truths_multiclass = []
    predictions_class = []

    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        targets = batch[-1]

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, targets)

        _, predicted_class = torch.max(outputs.data, 1)
        ground_truths_multiclass.extend(targets.cpu().numpy())
        predictions_class.extend(outputs.cpu().numpy())

        metric_logger.update(loss=loss.item())

    metric_logger.synchronize_between_processes()

    ground_truths_multiclass = np.asarray(ground_truths_multiclass)
    predictions_class = np.asarray(predictions_class)
    preds = np.argmax(predictions_class, axis=1)

    # Metrics
    accuracy_score = metrics.accuracy_score(ground_truths_multiclass, preds)
    kappa_score = metrics.cohen_kappa_score(ground_truths_multiclass, preds, weights='quadratic')
    f1_score = metrics.f1_score(ground_truths_multiclass, preds, average='micro')

    metric_logger.update(accuracy=accuracy_score)
    metric_logger.update(kappa=kappa_score)
    metric_logger.update(f1=f1_score)

    print(f"Accuracy: {accuracy_score:.4f}, Kappa: {kappa_score:.4f}, F1 Score: {f1_score:.4f}")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

