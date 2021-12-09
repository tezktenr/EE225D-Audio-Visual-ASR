"""
Filename: runAudioOnly.py
Description: This is the main execution file that starts the training using the Audio Only model
"""

# Python Standard Libraries
import os
import time
import argparse

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
from src.audio_visual_asr.model.AudioModel import AudioRecognition
from src.audio_visual_asr.dataset.LRW.LRW_AudioDataset import LRW_AudioDataset
from src.audio_visual_asr.lr_scheduler.AdjustLR import AdjustLR

# Global Constants
CURR_SOURCE_DIR_PATH = FileUtil.getDirectoryOfFile(FileUtil.resolvePath(__file__))

# Source Code
def get_model_save_path(config):
    base_save_path = FileUtil.joinPath(CURR_SOURCE_DIR_PATH, "_SAVED_MODELS", "AudioOnly")
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


def get_optimizer(model, config):
    mode = config["mode"]
    lr = config["lr"]

    if mode == 'temporalConv' or mode == 'finetuneGRU':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.)
    elif mode == 'backendGRU':
        for param in model.parameters():
            param.requires_grad = False
        for param in model.gru.parameters():
            param.requires_grad = True
        optimizer = optim.Adam([
                {'params': model.gru.parameters(), 'lr': lr}
            ], lr=0., weight_decay=0.)
    else:
        raise Exception(f"Unknown mode '{mode}' as specified in configuration")

    return optimizer


def get_data_loader(config, logger):
    dataset_path = config["dataset-path"]
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
        dsets[fold] = LRW_AudioDataset(fold, dataset_path, logger, sorted_labels_path)
        dset_sizes[fold] = len(dsets[fold])
        dset_loaders[fold] = torch.utils.data.DataLoader(dsets[fold],
                                                         batch_size=batch_size, num_workers=workers_num, shuffle=True)

    logger.info(f"Dataset Statistics - train: {dset_sizes['train']}, " +
                f"val: {dset_sizes['val']}, test: {dset_sizes['test']}")

    return dset_loaders, dset_sizes


def train(model, data_loader, criterion, epoch, optimizer, config, logger, use_gpu, save_path):
    totalEpochs = config["epochs"]
    isEveryFrame = config["every-frame"]
    mode = config["mode"]
    printInterval = config["print-stat-interval"]

    model.train()   # switch to training mode

    logger.info("")
    logger.info("====================")
    logger.info("** Start Training **")
    logger.info("====================")
    logger.info(f'Epoch {epoch}/{totalEpochs - 1}')
    logger.info(f'Current Learning rate: {TorchUtil.getLearningRates(optimizer)}')

    running_loss, running_corrects, running_all = 0., 0., 0.

    beginTime = time.time()
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        # prepare inputs and targets/labels
        inputs = inputs.float()
        inputs = Variable(inputs.cuda()) if use_gpu else Variable(inputs)
        targets = Variable(targets.cuda()) if use_gpu else Variable(targets)

        # make predictions using the model
        outputs = model(inputs)
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
        running_loss += loss.data * inputs.size(0)
        running_corrects += torch.sum(preds == targets.data)
        running_all += len(inputs)

        ### print all statistics
        if (batch_idx != 0 and batch_idx % printInterval == 0 or (batch_idx == len(data_loader)-1)):
            LoggerUtil.printAllStat(logger, beginTime, batch_idx, running_loss, running_corrects, running_all, data_loader)

    # finished the entire epoch for this training
    # print statistics for this epoch
    logger.info('(training) Epoch:\t{:2}\tLoss: {:.4f}\tAcc:{:.4f}\n'.format(
        epoch,
        running_loss / len(data_loader.dataset),
        running_corrects / len(data_loader.dataset))
    )

    # save models
    model_save_path = FileUtil.joinPath(save_path, f"{mode}_{epoch+1}.pt")
    torch.save(model.state_dict(), model_save_path)
    return model


def test(model, data_loader, criterion, epoch, config, logger, use_gpu):
    totalEpochs = config["epochs"]
    isEveryFrame = config["every-frame"]
    mode = config["mode"]
    printInterval = config["print-stat-interval"]

    model.eval()    # switch to validation mode

    logger.info("")
    logger.info("===================")
    logger.info("** Start Testing **")
    logger.info("===================")

    running_loss, running_corrects, running_all = 0., 0., 0.

    beginTime = time.time()
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        # prepare inputs and targets/labels
        inputs = inputs.float()
        inputs = Variable(inputs.cuda(), volatile=True) if use_gpu else Variable(inputs, volatile=True)
        targets = Variable(targets.cuda()) if use_gpu else Variable(targets)

        # make predictions using the model
        outputs = model(inputs)
        if isEveryFrame:
            outputs = torch.mean(outputs, 1)
        _, preds = torch.max(F.softmax(outputs, dim=1).data, 1)

        # calculating loss
        loss = criterion(outputs, targets)

        # statistics
        ### LOSS stats: running_loss += loss.data[0] * inputs.size(0)
        running_loss += loss.data * inputs.size(0)
        running_corrects += torch.sum(preds == targets.data)
        running_all += len(inputs)

        ### print all statistics
        if (batch_idx != 0 and batch_idx % printInterval == 0 or (batch_idx == len(data_loader) - 1)):
            LoggerUtil.printAllStat(logger, beginTime, batch_idx, running_loss, running_corrects, running_all, data_loader)

    # finished the entire epoch for this test
    # print statistics for this epoch
    logger.info('(testing) Epoch:\t{:2}\tLoss: {:.4f}\tAcc:{:.4f}\n'.format(
        epoch,
        running_loss / len(data_loader.dataset),
        running_corrects / len(data_loader.dataset))
    )

    return None


def run_model(config, use_gpu):
    # create the directory for saving trained models and log files
    save_path = get_model_save_path(config)
    log_file_path = FileUtil.joinPath(save_path, f'{config["mode"]}_{config["lr"]}.txt')
    logger = LoggerUtil.getLogger(log_file_path, "audioModelLogger")
    logger.info("\n" + "-" * 75)
    LoggerUtil.logCurrentTime(logger)

    # create a new model
    model = AudioRecognition(mode=config["mode"],
                             inputDim=512,
                             hiddenDim=512,
                             nClasses=config["nClasses"],
                             frameLen=29,
                             every_frame=config["every-frame"])

    # if config["model-path"] is specified, reload that model instead
    TorchUtil.reloadModel(model, logger, config["model-path"])

    # define loss function and optimizer
    criterion = get_loss_function()
    optimizer = get_optimizer(model, config)

    # get data set
    dset_loaders, dset_sizes = get_data_loader(config, logger)

    # learning rate scheduler
    lr_scheduler = AdjustLR(optimizer, [config["lr"]], sleep_epochs=5, half=5, verbose=1)

    # perform either training or testing as specified in config
    isTest = config["test"]
    if (isTest):
        # perform testing of model
        test(model, dset_loaders["val"], criterion, 0, config, logger, use_gpu)
        test(model, dset_loaders["test"], criterion, 0, config, logger, use_gpu)
        return
    else:
        # perform training of model
        for epoch in range(config["epochs"]):
            lr_scheduler.step(epoch)
            model = train(model, dset_loaders["train"], criterion, epoch, optimizer, config, logger, use_gpu, save_path)
            test(model, dset_loaders["val"], criterion, epoch, config, logger, use_gpu)


def getAudioConfig():
    # read audio visual section in "train_config.json"
    parser = argparse.ArgumentParser(description='Train Audio Only ASR Model')
    _DEFAULT_CONFIG_PATH = FileUtil.joinPath(CURR_SOURCE_DIR_PATH, "train_config.json")
    parser.add_argument('--config', default=_DEFAULT_CONFIG_PATH, help='path to "train_config.json" file')
    args =parser.parse_args()
    config = ConfigUtil.readAudioTrainConfig(args.config)
    return config


def trainAudioOnly():
    # get audio visual train_config
    audioConfig = getAudioConfig()

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
    run_model(audioConfig, use_gpu)


# Main Execution Block
if __name__ == "__main__":
    trainAudioOnly()