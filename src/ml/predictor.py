from __future__ import annotations

from data.dataset import Dataset
import torch
from ml.method import Method
from logging import getLogger
from ml.common import General
from ml.summary_writer import MySummaryWriter
from ml.common import Results, ProteinResult

logger = getLogger('app')


class MLPredictor(object):

    def __init__(self, dataset: Dataset, method: Method):
        if torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'

        self.dataset = dataset
        self.method = method

    def run(self, ids: list, writer: MySummaryWriter = None) -> Results:
        validation_set = self.method.get_dataset(ids=ids, dataset=self.dataset)
        validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=1, shuffle=True, pin_memory=True)
        sigm = torch.nn.Sigmoid()
        method = self.method

        results = Results()
        for features, padding, target, mask, prot_id in validation_loader:
            prot_id = prot_id[0]

            method.model.eval()
            with torch.no_grad():
                pred = method.forward(feature_batch=features)
                target = target.to(self.device)

                pred = pred.squeeze()
                target = target.squeeze()
                padding = padding.squeeze()

                pred, pred = General.remove_padded_positions(pred, target, padding)
                pred = sigm(pred)

                pred = pred.detach().cpu()

                prot = ProteinResult(prot_id=prot_id, target=pred.detach().numpy(),
                                     predictions=pred.detach().numpy())
                results[prot.prot_id] = prot

        return results
