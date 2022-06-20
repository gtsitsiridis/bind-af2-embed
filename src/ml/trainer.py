from ml.assess_performance import PerformanceEpochs, PerformanceAssessment, PerformanceEpoch
from data.dataset import Dataset
import torch
from tools import EarlyStopping, MyWorkerInit
from pathlib import Path
from method import Method
from protein_results import ProteinResults
import numpy as np


class MLTrainer(object):

    def __init__(self, dataset: Dataset, method: Method):
        if torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'

        self.dataset = dataset
        self.method = method

    def train(self, train_ids: list, validation_ids: list, output_dir: Path,
              epochs: int = 200, is_early_stopping: bool = True, batch_size: int = 406, verbose=True):
        """
        Train & validate predictor for one set of parameters and ids
        :param method:
        :param output_dir:
        :param train_ids:
        :param validation_ids:
        :param batch_size:
        :param epochs:
        :param is_early_stopping:
        :param verbose:
        :return:
        """
        method = self.method

        train_set = method.get_dataset(ids=train_ids, dataset=self.dataset)
        validation_set = method.get_dataset(ids=validation_ids, dataset=self.dataset)
        model = method.init_model()
        optimizer = method.init_optimizer(model=model)
        loss_fun = method.init_loss()
        early_stopping = self._init_early_stopping(output_dir=output_dir)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True,
                                                   worker_init_fn=MyWorkerInit())
        validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=True,
                                                        pin_memory=True, worker_init_fn=MyWorkerInit())

        train_performance = []
        validation_performance = []
        num_epochs = 0

        for epoch in range(epochs):
            if verbose:
                print("Epoch {}".format(epoch))

            train_performance_epoch = PerformanceEpoch("Train")
            val_performance_epoch = PerformanceEpoch("Validation")

            # training
            model.train()
            # feature_batch.shape=(B, 2*T + 1025, T)
            # target_batch.shape=(B, 3, T)
            # loss_mask_batch.shape=(B, 3, T)
            for feature_batch, target_batch, loss_mask_batch, _ in train_loader:
                optimizer.zero_grad()
                feature_batch = feature_batch.to(self.device)
                feature_1024_batch = feature_batch[:, :-1, :]  # feature_1024_batch.shape=(B, 2*T + 1024, T)
                target_batch = target_batch.to(self.device)
                loss_mask_batch = loss_mask_batch.to(self.device)

                pred_batch = model.forward(feature_1024_batch)  # pred_batch.shape=(B, 3, T)

                # don't consider padded positions for loss calculation
                # normalize based on length
                loss_el_batch = loss_fun(pred_batch, target_batch)  # loss_el_batch.shape=(B, 3, T)
                loss_el_masked_batch = loss_el_batch * loss_mask_batch  # loss_el_masked_batch.shape=(B, 3, T)
                loss_norm = torch.sum(loss_el_masked_batch)  # loss_norm.shape=(1)

                # add to performance object
                train_performance_epoch.loss += loss_norm.item()
                train_performance_epoch.loss_count += feature_batch.shape[0]
                train_performance_epoch.add_batch(feature_batch=feature_batch, pred_batch=pred_batch,
                                                  target_batch=target_batch)

                loss_norm.backward()
                optimizer.step()

            # validation
            model.eval()
            with torch.no_grad():
                for feature_batch, target_batch, loss_mask_batch, _ in validation_loader:
                    feature_batch = feature_batch.to(self.device)
                    feature_1024_batch = feature_batch[:, :-1, :]
                    target_batch = target_batch.to(self.device)
                    loss_mask_batch = loss_mask_batch.to(self.device)

                    pred_batch = model.forward(feature_1024_batch)

                    # don't consider padded position for loss calculation
                    loss_el_batch = loss_fun(pred_batch, target_batch)
                    loss_el_masked_batch = loss_el_batch * loss_mask_batch

                    # add to performance object
                    val_performance_epoch.loss += torch.sum(loss_el_masked_batch).item()
                    val_performance_epoch.loss_count += feature_batch.shape[0]
                    val_performance_epoch.add_batch(feature_batch=feature_batch, target_batch=target_batch,
                                                    pred_batch=pred_batch)

            train_performance_epoch.normalize()
            val_performance_epoch.normalize()

            if verbose:
                print(str(train_performance_epoch))
                print(str(val_performance_epoch))

            # append average performance for this epoch
            train_performance.append(train_performance_epoch)
            validation_performance.append(val_performance_epoch)

            num_epochs += 1

            # stop training if F1 score doesn't improve anymore
            if is_early_stopping:
                eval_val = val_performance_epoch.f1 * (-1)
                # eval_val = val_loss
                early_stopping(eval_val, model, verbose)
                if early_stopping.early_stop:
                    break

        model = torch.load(early_stopping.checkpoint_file)

        print(str(train_performance[-1]))
        print(str(validation_performance[-1]))

        return model

    def predict(self, ids: list, model: torch.nn.Module):
        validation_set = self.method.get_dataset(ids=ids, dataset=self.dataset)
        validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=1, shuffle=True, pin_memory=True)
        sigm = torch.nn.Sigmoid()

        proteins = dict()
        for features, target, mask, prot_id in validation_loader:
            prot_id = prot_id[0]

            model.eval()
            with torch.no_grad():
                features = features.to(self.device)
                target = target.to(self.device)

                features_1024 = features[..., :-1, :]
                pred = model.forward(features_1024)
                pred = sigm(pred)

                pred = pred.squeeze()
                target = target.squeeze()
                features = features.squeeze()

                pred_i, target_i = self._remove_padded_positions(pred, target, features)
                pred_i = pred_i.detach().cpu()

                prot = ProteinResults(name=prot_id, target=target_i, predictions=pred_i)
                proteins[prot_id] = prot

        return proteins

    @staticmethod
    def _remove_padded_positions(pred, target, feature):
        indices = (feature[feature.shape[0] - 1, :] != 0).nonzero()

        pred_i = pred[:, indices].squeeze()
        target_i = target[:, indices].squeeze()

        return pred_i, target_i

    @staticmethod
    def _init_early_stopping(output_dir: Path) -> EarlyStopping:
        checkpoint_file = output_dir / "checkpoint_early_stopping.pt"
        return EarlyStopping(patience=10, delta=0.01, checkpoint_file=checkpoint_file)
