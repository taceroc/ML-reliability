import tatiana_model
import sys
sys.path.append("../BOGLC/code")
import data_utils
import os
import random
import time
import torch
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler, TensorDataset, WeightedRandomSampler
import glob
from torch.optim import Adam, SGD
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import pickle
from torchsummary import summary
import seaborn as sns
import matplotlib.pyplot as plt
import signal
import utils_data
import DataSetLoad as data_load
import callbacks as callbacks
from functools import partial
import tempfile
from pathlib import Path
from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle
os.environ['RAY_PICKLE_VERBOSE_DEBUG'] = '1'
name = 'norm_raytune'
file_model = 'DIA_model_gpu_'+name+'.pkl'
print(file_model)
def handler(signum, frame):
    print('Signal handler called with signal', signum)
    exit(0)

signal.signal(signal.SIGUSR1, handler)
signal.signal(signal.SIGTERM, handler)
signal.signal(signal.SIGINT, handler)




def train_DIA(config, data_loc, num_samples_tr=200000, num_samples_te=20000):
    os.environ['PYTHONHASHSEED'] = str(76)
    random.seed(45)
    np.random.seed(1)
    # torch.backends.cudnn.enabled = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    seed = 42
    torch.manual_seed(seed)
    test_model = tatiana_model.CNN()
    test_model = test_model.to(device)
    criterion = torch.nn.BCELoss()
    optimizer = Adam(test_model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    last_epoch = 0
    batch_size = int(config["batch_size"])

    train_dataloader = utils_data.create_DIA_norm_dataset(batch_size, size=num_samples_tr, split='train')
    test_dataloader = utils_data.create_DIA_norm_dataset(batch_size, size=num_samples_te, split='test')
    
    num_epochs = 50
    print("Starting training")
    training_loss = []
    for epoch in range(last_epoch, last_epoch+num_epochs):
        print(epoch)
        loss_tr_by = []
        loss_te_by = []
        test_model.train()
        inf_time = 0
        total = 0
        # Initialize counters for TP, TN, FP, FN
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        t_init = time.process_time()
        torch.manual_seed(25)
        for i, data in enumerate(train_dataloader, 0):
            # Get the inputs and labels and set device
            diff, srch, tmpl, labels, ids = data
            train_data = torch.cat([diff, srch, tmpl], 3)
            train_data = train_data.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = test_model(train_data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print statistic
            inf_time = time.process_time() - t_init
            predicted = (outputs >= 0.5).float()
            total += labels.size(0)
            
            # Calculate TP, TN, FP, FN
            TP += ((predicted == 0) & (labels == 0)).sum().item()
            TN += ((predicted == 1) & (labels == 1)).sum().item()
            FP += ((predicted == 0) & (labels == 1)).sum().item()
            FN += ((predicted == 1) & (labels == 0)).sum().item()
            
            loss_tr_by.append(loss.item())

        accuracy_tr = 100 * (TN + TP) / total
        purity = TP / (TP + FP) if TP + FP != 0 else 0
        completness = TP / (TP + FN) if TP + FN != 0 else 0
        f1 = TP / (TP + 0.5*(FP + FN)) if (TP + 0.5*(FP + FN)) != 0 else 0
        print(f'Epoch: {epoch + 1}, training: acc: {accuracy_tr}, purity: {purity}, completness: {completness}, f1: {f1}')
        print(f'Epoch: {epoch + 1}, training loss: {loss_tr_by[-1]}')
        
        # Evaluate the network on the test set
        test_model.eval()   # Set the network to evaluation mode
        total = 0
        # Initialize counters for TP, TN, FP, FN
        TP = 0
        TN = 0
        FP = 0
        FN = 0

        with torch.no_grad():   # Disable gradient tracking
            torch.manual_seed(25)
            for i, data in enumerate(test_dataloader,0):
                # Move the inputs and labels to the device (GPU or CPU)
                # inf_time = 0
                # t_init = time.process_time()
                diff, srch, tmpl, labels, ids = data
                test_data = torch.cat([diff, srch, tmpl], 3)
                test_data = test_data.to(device)
                labels = labels.to(device)
                # Forward pass
                outputs = test_model(test_data)
                losste = criterion(outputs, labels)
                predicted = (outputs >= 0.5).float()
                total += labels.size(0)
            
                # Calculate TP, TN, FP, FN
                TP += ((predicted == 0) & (labels == 0)).sum().item()
                TN += ((predicted == 1) & (labels == 1)).sum().item()
                FP += ((predicted == 0) & (labels == 1)).sum().item()
                FN += ((predicted == 1) & (labels == 0)).sum().item()
                
                loss_te_by.append(losste.item())

        accuracy = 100 * (TN + TP) / total
        purity = TP / (TP + FP) if TP + FP != 0 else 0
        completness = TP / (TP + FN) if TP + FN != 0 else 0
        f1 = TP / (TP + 0.5*(FP + FN)) if (TP + 0.5*(FP + FN)) != 0 else 0
        print(f'Epoch: {epoch + 1}, testing: acc: {accuracy}, purity: {purity}, completness: {completness}, f1: {f1}')
        print(f'Epoch: {epoch + 1}, testing loss: {loss_te_by[-1]}')

        checkpoint_data = {
            "epoch": epoch+1,
            "net_state_dict": test_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / file_model
            with open(data_path, "wb") as fp:
                pickle.dump(checkpoint_data, fp)

            checkpoint = Checkpoint.from_directory(checkpoint_dir)
            train.report(
                {"loss": np.mean(np.array(loss_te_by)), 
                 "loss_tr": np.mean(np.array(loss_tr_by)),
                 "accuracy": accuracy, "purity": purity, "completness": completness},
                checkpoint=checkpoint,
            )
    print("Finished Training")

def test_acc(data_loc, test_model, device="cpu"):
    batch_size = 256
    test_dataloader = utils_data.create_DIA_norm_dataset(batch_size, size=21522, split='test')
    total = 0
    # Initialize counters for TP, TN, FP, FN
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    with torch.no_grad():   # Disable gradient tracking
        torch.manual_seed(25)
        for i, data in enumerate(test_dataloader,0):
            # Move the inputs and labels to the device (GPU or CPU)
            diff, srch, tmpl, labels, ids = data
            test_data = torch.cat([diff, srch, tmpl], 3)
            test_data = test_data.to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = test_model(test_data)
            predicted = (outputs >= 0.5).float()
            total += labels.size(0)
        
            # Calculate TP, TN, FP, FN
            TP += ((predicted == 0) & (labels == 0)).sum().item()
            TN += ((predicted == 1) & (labels == 1)).sum().item()
            FP += ((predicted == 0) & (labels == 1)).sum().item()
            FN += ((predicted == 1) & (labels == 0)).sum().item()
            
    accuracy = 100 * (TN + TP) / total
    purity = TP / (TP + FP) if TP + FP != 0 else 0
    completness = TP / (TP + FN) if TP + FN != 0 else 0
    f1 = TP / (TP + 0.5*(FP + FN)) if (TP + 0.5*(FP + FN)) != 0 else 0
    print(f'testing: acc: {accuracy}, purity: {purity}, completness: {completness}, f1: {f1}')
    
    return accuracy, purity, completness

def main(nseed, num_samples=10, max_num_epochs=50):
    print(num_samples, nseed)
    np.random.seed(nseed)
    config = {
        "lr": tune.loguniform(1e-4, 1e-3),
        "weight_decay": tune.loguniform(1e-3, 1e-1),
        "batch_size": tune.choice([128])
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=10,
        reduction_factor=2,
    )
    ROOT = os.path.dirname(os.path.abspath("../"))
    data_loc = os.path.join(ROOT, "taceroc/BOGLC/data/data_split_3s/")
    result = tune.run(
        tune.with_parameters(train_DIA, data_loc=data_loc),
        resources_per_trial={"cpu": 32, "gpu": 1},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        storage_path="/pscratch/sd/t/taceroc/BOGLC/raytune")
    
    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")

    best_trained_model = tatiana_model.CNN()
    best_trained_model.to("cpu")

    best_checkpoint = result.get_best_checkpoint(trial=best_trial, metric="accuracy", mode="max")
    with best_checkpoint.as_directory() as checkpoint_dir:
        data_path = Path(checkpoint_dir) / file_model
        with open(data_path, "rb") as fp:
            best_checkpoint_data = pickle.load(fp)

        best_trained_model.load_state_dict(best_checkpoint_data["net_state_dict"])
        test_accuracy = test_acc(data_loc, best_trained_model)
        print("Best trial test set accuracy: {}".format(test_accuracy))

if __name__ == "__main__":
    params = int(sys.argv[1])
    params2 = int(sys.argv[2])
    
    # def generate_configs(num_samples, config):
    #     analysis = tune.run(
    #         lambda config: None,  # Dummy function
    #         config=config,
    #         num_samples=num_samples,
    #         verbose=0
    #     )
    #     return analysis.get_all_configs()
    num_samples=params
    main(nseed=params2, num_samples=num_samples, max_num_epochs=50)