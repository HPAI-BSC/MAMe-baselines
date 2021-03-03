import time
import torch

from trainer.utils.consts import Split
from trainer.utils.utils import current_memory_usage, squeeze_generic
from trainer.utils.saver import load_checkpoint


def train_epoch(train_loader, model, loss_function, optimizer, device, epoch):
    model.train()

    correct = 0
    total = 0
    losses = 0
    t0 = time.time()
    for idx, (batch_images, batch_labels) in enumerate(train_loader):
        # Loading tensors in the used device
        step_images, step_labels = batch_images.to(device), batch_labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        step_output = model(step_images)
        step_output = squeeze_generic(step_output, axes_to_keep=[0])
        step_labels = squeeze_generic(step_labels, axes_to_keep=[0])
        loss = loss_function(step_output, step_labels)
        loss.backward()
        optimizer.step()

        step_total = step_labels.size(0)
        step_loss = loss.item()
        losses += step_loss * step_total
        total += step_total

        step_preds = torch.max(step_output.data, 1)[1]
        step_correct = (step_preds == step_labels).sum().item()
        correct += step_correct

    train_loss = losses / total
    train_acc = correct / total
    format_args = (epoch, train_acc, train_loss, time.time() - t0, current_memory_usage())
    print('EPOCH {} :: train accuracy: {:.4f} - train loss: {:.4f} at {:.1f}s  [{} MB]'.format(*format_args))


def val_epoch(val_loader, model, loss_function, device, epoch):
    model.eval()

    correct = 0
    total = 0
    losses = 0
    t0 = time.time()
    with torch.no_grad():
        for batch_images, batch_labels in val_loader:
            # Loading tensors in the used device
            step_images, step_labels = batch_images.to(device), batch_labels.to(device)

            step_output = model(step_images)
            step_output = squeeze_generic(step_output, axes_to_keep=[0])
            step_labels = squeeze_generic(step_labels, axes_to_keep=[0])
            loss = loss_function(step_output, step_labels)

            step_total = step_labels.size(0)
            step_loss = loss.item()
            losses += step_loss * step_total
            total += step_total

            step_preds = torch.max(step_output.data, 1)[1]
            step_correct = (step_preds == step_labels).sum().item()
            correct += step_correct

    val_loss = losses / total
    val_acc = correct / total
    format_args = (epoch, val_acc, val_loss, time.time() - t0, current_memory_usage())
    print('EPOCH {} :: val accuracy: {:.4f} - val loss: {:.4f} at {:.1f}s  [{} MB]'.format(*format_args))


def training(
        input_pipeline,
        model,
        loss_function,
        optimizer,
        device,
        saver,
        retrain,
        max_epochs=10000,
):
    initial_epoch = 0

    if saver and retrain:
        model, optimizer, last_epoch = load_checkpoint(saver.model_path, model, optimizer)
        initial_epoch = last_epoch + 1

    for epoch in range(initial_epoch, max_epochs):
        train_epoch(input_pipeline[Split.TRAIN], model, loss_function, optimizer, device, epoch)

        if saver:
            saver.save_checkpoint(model, optimizer, epoch)

        val_epoch(input_pipeline[Split.VAL], model, loss_function, device, epoch)
