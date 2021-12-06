"""
Filename: runAudioVisual.py
Description: This is the main execution file that starts the training using the both the audio and video model
"""

# Python Standard Libraries
import os
import time

# Third Party Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

# Project Module
from src.utility.ConfigUtil import ConfigUtil
from src.utility.FileUtil import FileUtil
from src.utility.TorchUtil import TorchUtil
from src.utility.LoggerUtil import LoggerUtil
from src.utility.VideoUtil import VideoUtil
from src.model.AudioVisualModel import ConcatGRU, AudioRecognition, LipReading
from src.dataset.LRW.LRW_AudioVisualDataset import LRW_AudioVisualDataset
from src.lr_scheduler.AdjustLR import AdjustLR


# Source Code
def get_model_save_path(config):
    base_save_path = "../saved_models/audioVisualModels"
    isEveryFrame = config["every-frame"]
    mode = config["mode"]

    if isEveryFrame and mode != 'temporalConv':
        save_path = FileUtil.joinPath(base_save_path, mode, "_every_frame")
    elif not isEveryFrame and mode != 'temporalConv':
        save_path = FileUtil.joinPath(base_save_path, mode, "_last_frame")
    elif mode == 'temporalConv':
        save_path = FileUtil.joinPath(base_save_path, mode)
    else:
        raise Exception(f"Unknown mode '{mode}' as specified in configuration")

    actual_save_path = FileUtil.resolvePath(save_path)

    if not FileUtil.directoryExists(actual_save_path):
        LoggerUtil.warning(f"Models save_path at '{actual_save_path}' didn't exist.")
        FileUtil.makeDirRecursively(actual_save_path)
        LoggerUtil.warning(f"Successfully created save_path at '{actual_save_path}'.")

    return actual_save_path


def get_loss_function():
    criterion = nn.CrossEntropyLoss()
    return criterion


def get_optimizer(audio_model, video_model, concat_model, config):
    mode = config["mode"]
    lr = config["lr"]

    if mode == 'finetuneGRU':
        optimizer = optim.Adam([{'params': video_model.parameters(), 'lr': lr, 'weight_decay': 0.},
                                {'params': audio_model.parameters(), 'lr': lr, 'weight_decay': 0.},
                                {'params': concat_model.parameters(), 'lr': lr, 'weight_decay': 0.}],
                               lr=0., weight_decay=0.)
    elif mode == 'backendGRU':
        for param in audio_model.parameters():
            param.requires_grad = False
        for param in video_model.parameters():
            param.requires_grad = False
        for param in concat_model.parameters():
            param.requires_grad = True
        optimizer = optim.Adam([
            {'params': concat_model.parameters(), 'lr': lr}
            ], lr=0., weight_decay=0.)
    else:
        raise Exception(f"Unknown mode '{mode}' as specified in configuration")

    return optimizer


def get_data_loader(config, logger):
    audio_dataset_path = config["audio-dataset"]
    video_dataset_path = config["video-dataset"]
    sorted_labels_path = config["sorted-labels-path"]
    batch_size = config["batch-size"]
    workers_num = config["workers"]

    logger.info("")
    logger.info("========================")
    logger.info("** Start Loading Data **")
    logger.info("========================")

    dsets = dict()
    dset_sizes = dict()
    dset_loaders = dict()

    folds = ['train', 'val', 'test']
    for fold in folds:
        dsets[fold] = LRW_AudioVisualDataset(fold, audio_dataset_path, video_dataset_path, logger, sorted_labels_path)
        dset_sizes[fold] = len(dsets[fold])
        dset_loaders[fold] = torch.utils.data.DataLoader(dsets[fold],
                                                         batch_size=batch_size, num_workers=workers_num, shuffle=True)

    logger.info(f"Dataset Statistics - train: {dset_sizes['train']}, " +
                f"val: {dset_sizes['val']}, test: {dset_sizes['test']}")

    return dset_loaders, dset_sizes


def _prepare_targets(targets, use_gpu):
    return Variable(targets.cuda()) if use_gpu else Variable(targets)


def _prepare_audio_inputs(audio_inputs, use_gpu, training=True):
    audio_inputs = audio_inputs.float()
    if training:
        audio_inputs = Variable(audio_inputs.cuda()) if use_gpu else Variable(audio_inputs)
    else:
        audio_inputs = Variable(audio_inputs.cuda(), volatile=True) if use_gpu else Variable(audio_inputs, volatile=True)

    return audio_inputs


def _prepare_video_inputs(video_inputs, use_gpu, training=True):
    if (training):
        batch_img = VideoUtil.RandomCrop(video_inputs.numpy(), (88, 88))
        batch_img = VideoUtil.ColorNormalize(batch_img)
        batch_img = VideoUtil.HorizontalFlip(batch_img)
    else:
        batch_img = VideoUtil.CenterCrop(video_inputs.numpy(), (88, 88))
        batch_img = VideoUtil.ColorNormalize(batch_img)

    batch_img = np.reshape(batch_img,
                           (batch_img.shape[0], batch_img.shape[1], batch_img.shape[2], batch_img.shape[3], 1))
    video_inputs = torch.from_numpy(batch_img)
    video_inputs = video_inputs.float().permute(0, 4, 1, 2, 3)

    if (training):
        video_inputs = Variable(video_inputs.cuda()) if use_gpu else Variable(video_inputs)
    else:
        video_inputs = Variable(video_inputs.cuda(), volatile=True) if use_gpu else Variable(video_inputs, volatile=True)

    return video_inputs


def train(audio_model, video_model, concat_model, data_loader, criterion, epoch, optimizer, config, logger, use_gpu, save_path):
    totalEpochs = config["epochs"]
    isEveryFrame = config["every-frame"]
    mode = config["mode"]
    printInterval = config["print-stat-interval"]

    # switch to training mode
    audio_model.train()
    video_model.train()
    concat_model.train()

    logger.info("")
    logger.info("====================")
    logger.info("** Start Training **")
    logger.info("====================")
    logger.info(f'Epoch {epoch}/{totalEpochs - 1}')
    logger.info(f'Current Learning rate: {TorchUtil.getLearningRates(optimizer)}')

    running_loss, running_corrects, running_all = 0., 0., 0.

    beginTime = time.time()
    for batch_idx, (audio_inputs, video_inputs, targets) in enumerate(data_loader):
        # prepare inputs and targets
        audio_inputs = _prepare_audio_inputs(audio_inputs, use_gpu, training=True)
        video_inputs = _prepare_video_inputs(video_inputs, use_gpu, training=True)
        targets = _prepare_targets(targets, use_gpu)

        # make predictions using the model
        audio_outputs = audio_model(audio_inputs)
        video_outputs = video_model(video_inputs)
        merged_inputs = torch.cat((audio_outputs, video_outputs), dim=2)
        outputs = concat_model(merged_inputs)
        if isEveryFrame:
            outputs = torch.mean(outputs, 1)
        _, preds = torch.max(F.softmax(outputs, dim=1).data, 1)

        # training & learning
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # statistics
        ### LOSS stats: running_loss += loss.data[0] * inputs.size(0)
        running_loss += loss.data * merged_inputs.size(0)
        running_corrects += torch.sum(preds == targets.data)
        running_all += len(merged_inputs)

        ### print all statistics
        if (batch_idx is not 0 and batch_idx % printInterval == 0 or (batch_idx == len(data_loader)-1)):
            LoggerUtil.printAllStat(logger, beginTime, batch_idx, running_loss, running_corrects, running_all, data_loader)

    # finished the entire epoch for this training
    # print statistics for this epoch
    logger.info('(training) Epoch:\t{:2}\tLoss: {:.4f}\tAcc:{:.4f}'.format(
        epoch,
        running_loss / len(data_loader.dataset),
        running_corrects / len(data_loader.dataset))
    )

    # save models
    audio_model_save_path = FileUtil.joinPath(save_path, f"audio_model_{mode}_{epoch+1}.pt")
    video_model_save_path = FileUtil.joinPath(save_path, f"video_model_{mode}_{epoch + 1}.pt")
    concat_model_save_path = FileUtil.joinPath(save_path, f"concat_model_{mode}_{epoch + 1}.pt")

    torch.save(audio_model.state_dict(), audio_model_save_path)
    torch.save(video_model.state_dict(), video_model_save_path)
    torch.save(concat_model.state_dict(), concat_model_save_path)

    return audio_model, video_model, concat_model


def test(audio_model, video_model, concat_model, data_loader, criterion, epoch, config, logger, use_gpu):
    totalEpochs = config["epochs"]
    isEveryFrame = config["every-frame"]
    mode = config["mode"]
    printInterval = config["print-stat-interval"]

    # switch to validation mode
    audio_model.eval()
    video_model.eval()
    concat_model.eval()

    logger.info("")
    logger.info("===================")
    logger.info("** Start Testing **")
    logger.info("===================")

    running_loss, running_corrects, running_all = 0., 0., 0.

    beginTime = time.time()
    for batch_idx, (audio_inputs, video_inputs, targets) in enumerate(data_loader):
        # prepare inputs and targets
        audio_inputs = _prepare_audio_inputs(audio_inputs, use_gpu, training=False)
        video_inputs = _prepare_video_inputs(video_inputs, use_gpu, training=False)
        targets = _prepare_targets(targets, use_gpu)

        # make predictions using the model
        audio_outputs = audio_model(audio_inputs)
        video_outputs = video_model(video_inputs)
        merged_inputs = torch.cat((audio_outputs, video_outputs), dim=2)
        outputs = concat_model(merged_inputs)
        if isEveryFrame:
            outputs = torch.mean(outputs, 1)
        _, preds = torch.max(F.softmax(outputs, dim=1).data, 1)

        # calculating loss
        loss = criterion(outputs, targets)

        # statistics
        ### LOSS stats: running_loss += loss.data[0] * inputs.size(0)
        running_loss += loss.data * merged_inputs.size(0)
        running_corrects += torch.sum(preds == targets.data)
        running_all += len(merged_inputs)

        ### print all statistics
        if (batch_idx is not 0 and batch_idx % printInterval == 0 or (batch_idx == len(data_loader) - 1)):
            LoggerUtil.printAllStat(logger, beginTime, batch_idx, running_loss, running_corrects, running_all,
                                    data_loader)

    # finished the entire epoch for this training
    # print statistics for this epoch
    logger.info('(training) Epoch:\t{:2}\tLoss: {:.4f}\tAcc:{:.4f}\n'.format(
        epoch,
        running_loss / len(data_loader.dataset),
        running_corrects / len(data_loader.dataset))
    )
    return None


def run_model(config, use_gpu):
    # create the directory for saving trained models and log files
    save_path = get_model_save_path(config)
    log_file_path = FileUtil.joinPath(save_path, f'{config["mode"]}_{config["lr"]}.txt')
    logger = LoggerUtil.getLogger(log_file_path, "audioVisualModelLogger")
    logger.info("\n" + "-" * 75)
    LoggerUtil.logCurrentTime(logger)

    # create a new model
    nClasses = config["nClasses"]
    mode = config["mode"]
    isEveryFrame = config["every-frame"]

    audio_model = AudioRecognition(mode=mode, inputDim=512, hiddenDim=512, nClasses=nClasses, frameLen=29, every_frame=isEveryFrame)
    video_model = LipReading(mode=mode, inputDim=256, hiddenDim=512, nClasses=nClasses, frameLen=29, every_frame=isEveryFrame)
    if (mode in ["backendGRU", "finetuneGRU"]):
        concat_model = ConcatGRU(inputDim=2048, hiddenDim=512, nLayers=2, nClasses=nClasses, every_frame=isEveryFrame)
    else:
        raise ValueError(f"Unknown mode {mode} for concat_model")

    # if config["audio-path"], config["video-path"], config["concat-path"] is specified, reload those models instead
    TorchUtil.reloadModel(audio_model, logger, config["audio-path"])
    TorchUtil.reloadModel(video_model, logger, config["video-path"])
    TorchUtil.reloadModel(concat_model, logger, config["concat-path"])

    # define loss function and optimizer
    criterion = get_loss_function()
    optimizer = get_optimizer(audio_model, video_model, concat_model, config)

    # get data set
    dset_loaders, dset_sizes = get_data_loader(config, logger)

    # learning rate scheduler
    lr_scheduler = AdjustLR(optimizer, [config["lr"]], sleep_epochs=5, half=5, verbose=1)

    # perform either training or testing as specified in config
    isTest = config["test"]
    if (isTest):
        # perform testing of model
        test(audio_model, video_model, concat_model, dset_loaders["val"], criterion, 0, config, logger, use_gpu)
        test(audio_model, video_model, concat_model, dset_loaders["test"], criterion, 0, config, logger, use_gpu)
        return
    else:
        # perform training of model
        for epoch in range(config["epochs"]):
            lr_scheduler.step(epoch)
            audio_model, video_model, concat_model = train(audio_model, video_model, concat_model,
                                                           dset_loaders["train"], criterion, epoch, optimizer,
                                                           config, logger, use_gpu, save_path)
            test(audio_model, video_model, concat_model, dset_loaders["val"], criterion, epoch, config, logger, use_gpu)


def main():
    # reading configuration
    configFilePath = ConfigUtil.getConfigPath()
    audioVisualConfig = ConfigUtil.readAudioVisualConfig(configFilePath)

    # setup GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # setup random seed for training data shuffling
    SEED = 1
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)

    # run
    run_model(audioVisualConfig, use_gpu)



# Main Execution Block
if __name__ == "__main__":
    main()