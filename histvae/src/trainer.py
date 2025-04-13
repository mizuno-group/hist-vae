# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:09:08 2019

trainer

@author: tadahaya
"""
import os, time
from typing import List, Union, Any
import torch
from torch.nn.utils import clip_grad_norm_

from .utils import save_experiment, save_checkpoint, calc_elapsed_time

class BaseTrainer:
    def __init__(self):
        self.callbacks: List[Any] = []

    def train(self):
        """ train the model """
        raise NotImplementedError
    
    def train_epoch(self):
        """ train the model for one epoch """
        raise NotImplementedError

    def evaluate(self):
        """ evaluate the model """
        raise NotImplementedError

    def set_callbacks(self, callbacks: Union[List[Any], Any]):
        """
        Parameters
        ----------
        callbacks: list, instance
            list of callback instances or a single callback instance

        """
        if not isinstance(callbacks, list):
            callbacks = [callbacks]
        for callback in callbacks:
            if not callable(callback):
                raise ValueError("!! Callbacks must be callable instances. !!")
            else:
                self.callbacks.append(callback)

    def run_callbacks(self, **kwargs):
        """
        Parameters
        ----------
        kwargs: dict
            keyword arguments to pass to the callbacks

        """        
        for callback in self.callbacks:
            callback(**kwargs)


class BaseLogger:
    def __init__(self):
        self.items = []

    def __call__(self, **kwargs):
        """ Args: kwargs (dict): keyword arguments """
        self.items.append(kwargs)

    def get_items(self):
        """ get items as a dict """
        items = {}
        for item in self.items:
            for k, v in item.items():
                if k not in items:
                    items[k] = []
                items[k].append(v)
        return items


class EarlyStopping:
    def __init__(self, patience=10, mode="min", restore_best_model=True, verbose=True):
        """
        Parameters
        ----------
        patience: int
            number of epochs with no improvement after which training will be stopped

        mode: str
            one of {min, max}.

        restore_best_model: bool
            whether to restore model weights from the epoch with the best score

        """
        self.patience = patience
        self.restore_best_model = restore_best_model
        self.verbose = verbose
        self.best_score = None
        self.best_epoch = None
        self.counter = 0
        self.early_stop = False
        self.best_model_state = None
        self._monitor_fxn = {
            "min": lambda a, b: a < b,
            "max": lambda a, b: a > b
        }[mode]


    def __call__(self, model, score, epoch):
        """
        Parameters
        ----------
        model: torch.nn.Module
            current model

        score: float
            current score (loss or accuracy)

        epoch: int
            current epoch

        """
        if self.best_score is None or self._monitor_fxn(score, self.best_score):
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            if self.restore_best_model:
                self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                # store the best model state on CPU
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(">> EarlyStopping triggered")
                if self.restore_best_model and self.best_model_state:
                    model.load_state_dict(self.best_model_state)


class PreTrainer(BaseTrainer):
    def __init__(
            self, config, model, optimizer=None, callbacks=[], outdir:str=""
            ):
        super().__init__()
        # arguments
        self.config = config
        self.device = config.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
            )
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.logger = BaseLogger()
        self.callbacks = callbacks
        self.callbacks.append(self.logger)
        self.outdir = outdir
        # config contents
        self.exp_name = config["exp_name"]
        self.save_model_every = config["save_model_every"]
        self.log_every = config["log_every"]
        # I/O
        self.resdir = os.path.join(self.outdir, self.exp_name)
        os.makedirs(self.resdir, exist_ok=True)
        # early stopping
        self.early_stopping = None
        if (config["patience"] > 0) & (config["patience"] is not None):
            self.early_stopping = EarlyStopping(patience=config["patience"], mode="min")
        # loggings
        self.history = {
            "best_score": None,
            "early_stop_epoch": None,
            "elapsed_time": None
        }


    def train(self, trainloader, testloader):
        """
        train the model for the specified number of epochs.
        
        """
        start_time = time.time()
        # training
        for i in range(self.config["epochs"]):
            train_loss, train_recon, train_kl = self.train_epoch(trainloader)
            test_loss, test_recon, test_kl = self.evaluate(testloader)
            # logging
            self.run_callbacks(
                epoch=i + 1,
                train_loss=train_loss,
                test_loss=test_loss,
                train_recon=train_recon,
                train_kl=train_kl,
                test_recon=test_recon,
                test_kl=test_kl
                )
            if (i + 1) % self.log_every == 0:
                print(
                    f"Epoch: {i + 1}, Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}"
                    )
            # early stopping
            if self.early_stopping is not None:
                self.early_stopping(self.model, test_loss, i + 1)
                if self.early_stopping.early_stop:
                    self.history["early_stop_epoch"] = i + 1 # record the epoch
                    break
            # save the model
            if self.save_model_every > 0 and (i + 1) % self.save_model_every == 0:
                save_checkpoint(model=self.model, optimizer=self.optimizer, name=f"epoch_{i + 1}", outdir=self.resdir)
        # save the experiment
        elapsed_time = calc_elapsed_time(start_time)
        self.history["elapsed_time"] = elapsed_time
        self.history["best_score"] = self.early_stopping.best_score if self.early_stopping is not None else None
        self.history["best_epoch"] = self.early_stopping.best_epoch if self.early_stopping is not None else None
        self.history.update(self.logger.get_items())
        save_experiment(
            config=self.config, model=self.model, optimizer=self.optimizer, history=self.history, outdir=self.resdir
            )


    def train_epoch(self, trainloader):
        """ train the model for one epoch """
        self.model.train()
        self.optimizer.train()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        total_samples = 0 # for averaging the loss
        # initialize the gradients
        self.optimizer.zero_grad()
        for i, (data, label) in enumerate(trainloader):
            # data = (original hist, noisy hist)
            hist0, hist1 = (x.to(self.device) for x in data)
            label = label.to(self.device)
            # forward
            recon, mu, logvar = self.model(hist1) # output, mu, logvar
            # loss calculation
            loss, recon_loss, kl_loss = self.model.vae_loss(
                recon, hist0, mu, logvar, beta=self.config["beta"]
                )
            # note: loss is averaged over the batch
            # backpropagation
            loss.backward()
            # clip the gradients
            if self.config["clip_grad"] > 0:
                clip_grad_norm_(self.model.parameters(), self.config["clip_grad"])
            # update the parameters
            if (i + 1) % self.config["accum_grad"] == 0 or (i + 1) == len(trainloader):
                self.optimizer.step()  # Perform the parameter update
                self.optimizer.zero_grad()  # Reset gradients for the next accumulation
            batch_size = hist0.shape[0]
            total_loss += loss.detach().item() * batch_size
            total_recon_loss += recon_loss.detach().item() * batch_size
            total_kl_loss += kl_loss.detach().item() * batch_size
            total_samples += batch_size
        return total_loss / total_samples, total_recon_loss / total_samples, total_kl_loss / total_samples


    def evaluate(self, testloader):
        self.model.eval()
        self.optimizer.eval()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        total_samples = 0 # for averaging the loss
        with torch.no_grad():
            for data, label in testloader:
                hist0, hist1 = (x.to(self.device) for x in data)
                label = label.to(self.device)
                # forward
                recon, mu, logvar = self.model(hist1) # output, mu, logvar
                # loss calculation
                loss, recon_loss, kl_loss = self.model.vae_loss(
                    recon, hist0, mu, logvar, beta=self.config["beta"]
                    )
                # Loss accumulation
                batch_size = hist0.shape[0]
                total_loss += loss.item() * batch_size # detach() is not necessary
                total_recon_loss += recon_loss.item() * batch_size
                total_kl_loss += kl_loss.item() * batch_size
                total_samples += batch_size
        return total_loss / total_samples, total_recon_loss / total_samples, total_kl_loss / total_samples


class FineTuner(BaseTrainer):
    def __init__(self, config, model, optimizer=None, loss_fn=None, callbacks=[], outdir=""):
        super().__init__()
        # arguments
        self.config = config
        self.device = config.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
            )
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.logger = BaseLogger()
        self.callbacks = callbacks
        self.callbacks.append(self.logger)
        self.outdir = outdir
        # config contents
        self.exp_name = config["exp_name"]
        self.save_model_every = config["save_model_every"]
        self.log_every = config["log_every"]
        if config["frozen"]:
            self.use_pretrain_loss = False # if the model is frozen, pretrain loss is never used
        else:
            self.use_pretrain_loss = config["use_pretrain_loss"]
        # I/O
        self.resdir = os.path.join(self.outdir, self.exp_name)
        os.makedirs(self.resdir, exist_ok=True)
        # early stopping
        self.early_stopping = None
        if (config["patience"] > 0) & (config["patience"] is not None):
            self.early_stopping = EarlyStopping(patience=config["patience"], mode="min")
        # loggings
        self.history = {
            "best_score": None,
            "early_stop_epoch": None,
            "elapsed_time": None
        }


    def train(self, trainloader, testloader):
        """
        train the model for the specified number of epochs.
        
        """
        start_time = time.time()
        # training
        for i in range(self.config["epochs"]):
            train_loss, train_recon, train_kl, train_acc = self.train_epoch(trainloader)
            test_loss, test_recon, test_kl, test_acc = self.evaluate(testloader)
            # logging
            self.run_callbacks(
                epoch=i + 1,
                train_loss=train_loss,
                test_loss=test_loss,
                train_recon=train_recon,
                train_kl=train_kl,
                test_recon=test_recon,
                test_kl=test_kl,
                train_accuracy=train_acc,
                test_accuracy=test_acc,
                )
            if (i + 1) % self.log_every == 0:
                print(f"Epoch: {i + 1}")
                print(f"  Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}")
                print(f"  Train accuracy: {train_acc:.4f}, Test accuracy: {test_acc:.4f}")
            # early stopping
            if self.early_stopping is not None:
                self.early_stopping(self.model, test_loss, i + 1)
                if self.early_stopping.early_stop:
                    self.history["early_stop_epoch"] = i + 1 # record the epoch
                    break
            # save the model
            if self.save_model_every > 0 and (i + 1) % self.save_model_every == 0:
                save_checkpoint(model=self.model, optimizer=self.optimizer, name=f"epoch_{i + 1}", outdir=self.resdir)
        # save the experiment
        elapsed_time = calc_elapsed_time(start_time)
        self.history["elapsed_time"] = elapsed_time
        self.history["best_score"] = self.early_stopping.best_score if self.early_stopping is not None else None
        self.history["best_epoch"] = self.early_stopping.best_epoch if self.early_stopping is not None else None
        self.history.update(self.logger.get_items())
        save_experiment(
            config=self.config, model=self.model, optimizer=self.optimizer, history=self.history, outdir=self.resdir
            )


    def train_epoch(self, trainloader):
        """ train the model for one epoch """
        self.model.train()
        self.optimizer.train()
        total_loss = 0.0
        total_pt_loss = 0.0
        total_ft_loss = 0.0
        total_samples = 0 # for averaging the loss
        correct = 0
        # initialize the gradients
        self.optimizer.zero_grad()
        for i, (data, label) in enumerate(trainloader):
            # data = (original hist, noisy hist)
            hist0, hist1 = (x.to(self.device) for x in data)
            label = label.to(self.device)
            # forward/loss calculation
            if self.use_pretrain_loss:
                logits, recon, mu, logvar = model(hist1) # use noisy hist for pretraining
                pt_loss, _, _ = self.model.vae_loss(
                    recon, hist0, mu, logvar, beta=self.config["beta"]
                    )
                # note: ignore the reconstruction loss and kl_loss in logging
                ft_loss = self.loss_fn(logits, label)
                loss = pt_loss + ft_loss
            else:
                logits, recon, mu, logvar = model(hist0) # use original hist
                ft_loss = self.loss_fn(logits, label)
                pt_loss = 0
                loss = ft_loss
            # backpropagation
            loss.backward()
            # clip the gradients
            if self.config["clip_grad"] > 0:
                clip_grad_norm_(self.model.parameters(), self.config["clip_grad"])
            # update the parameters
            if (i + 1) % self.config["accum_grad"] == 0 or (i + 1) == len(trainloader):
                self.optimizer.step()  # Perform the parameter update
                self.optimizer.zero_grad()  # Reset gradients for the next accumulation
            # Loss accumulation
            batch_size = hist0.shape[0]
            total_loss += loss.detach().item() * batch_size
            total_samples += batch_size
            # Accuracy calculation (disable gradients for efficiency)
            with torch.no_grad():
                predictions = torch.argmax(output, dim=1)
                correct += (predictions == label).sum().item()
        return total_loss / total_samples, total_pt_loss / total_samples, total_ft_loss / total_samples, correct / total_samples
            

    def evaluate(self, testloader):
        """Evaluate the model on the test set"""
        self.model.eval()
        self.optimizer.eval()
        total_loss = 0.0
        total_pt_loss = 0.0
        total_ft_loss = 0.0
        total_samples = 0
        correct = 0
        with torch.no_grad():
            for data, label in testloader:
                # Move data to device
                hist0, hist1 = (x.to(self.device) for x in data)
                label = label.to(self.device)
                # forward/loss calculation
                if self.use_pretrain_loss:
                    logits, recon, mu, logvar = model(hist1) # use noisy hist for pretraining
                    pt_loss, _, _ = self.model.vae_loss(
                        recon, hist0, mu, logvar, beta=self.config["beta"]
                        )
                    # note: ignore the reconstruction loss and kl_loss in logging
                    ft_loss = self.loss_fn(logits, label)
                    loss = pt_loss + ft_loss
                else:
                    logits, recon, mu, logvar = model(hist0) # use original hist
                    ft_loss = self.loss_fn(logits, label)
                    pt_loss = 0
                    loss = ft_loss
                # Loss accumulation
                batch_size = hist0.shape[0]
                total_loss += loss.item() * batch_size # detach() is not necessary
                total_samples += batch_size
                # Accuracy calculation
                predictions = torch.argmax(output, dim=1)
                correct += int((predictions == label).sum())
        return total_loss / total_samples, total_pt_loss / total_samples, total_ft_loss / total_samples, correct / total_samples
