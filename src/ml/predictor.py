from __future__ import annotations

from data.dataset import Dataset
import torch
from ml.method import Method
from logging import getLogger
from ml.common import General
from ml.summary_writer import MySummaryWriter
from ml.common import Results, ProteinResult
from pathlib import Path

from ml.template import RunTemplate

logger = getLogger('app')


class MLPredictor(object):

    def __init__(self, dataset: Dataset, method: Method, template: RunTemplate, tag: str,
                 writer: MySummaryWriter = None,
                 results_file_path: Path = None):
        if torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'

        self._dataset = dataset
        self._method = method
        self._writer = writer
        self._train_params = template.train_params
        self._tag = tag
        self._results_file_path = results_file_path

    def __call__(self, ids: list) -> Results:
        validation_set = self._method.get_dataset(ids=ids, writer=self._writer)
        validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=1, shuffle=True, pin_memory=True)
        method = self._method
        writer = self._writer
        train_params = self._train_params

        results = Results()
        i = 0
        # batch size is 1
        for features, padding, target, loss_mask, prot_id in validation_loader:
            prot_id = prot_id[0]
            method.model.eval()
            with torch.no_grad():
                if i == 0 and writer is not None:
                    writer.add_model(model=method.model, feature_batch=features)
                i += 1
                target = target.to(self.device)
                loss_mask = loss_mask.to(self.device)
                pred = method.forward(feature_batch=features)
                loss = method.loss(pred_batch=pred, loss_mask_batch=loss_mask, target_batch=target)

                # create protein result
                prot = ProteinResult(prot_id=prot_id, target=target.squeeze(),
                                     predictions=pred.squeeze(),
                                     padding=padding.squeeze(),
                                     loss=loss.squeeze(),
                                     tag=self._tag)
                results[prot.prot_id] = prot

        # log protein results and performance
        writer.add_protein_results(protein_results=results,
                                   cutoff=train_params['cutoff'])
        performance = results.get_performance(cutoff=train_params['cutoff'])
        writer.add_performance_figures(performance)
        writer.add_performance_scalars(performance=performance)
        if self._results_file_path is not None:
            General.to_csv(df=results.to_df(cutoff=train_params['cutoff']),
                           filename=self._results_file_path)
        return results
