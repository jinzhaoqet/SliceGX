import os
import random

import torch_geometric.transforms as T
import hydra
import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

from dataset import get_dataset, get_dataloader
from gnnNets import get_gnnNets

from utils import check_dirs, set_seed
import warnings
from torch.optim import Adam
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
import os
import torch


class TrainModel(object):
    def __init__(
        self,
        model,
        dataset_name,
        dataset,
        device,
        graph_classification=True,
        save_dir=None,
        save_name="model",
        train_mask=None,
        valid_mask=None,
        test_mask=None,
        **kwargs,
    ):
        self.model = model
        self.dataset_name = dataset_name
        self.dataset = dataset  # train_mask, eval_mask, test_mask
        self.loader = None
        self.device = device
        self.graph_classification = graph_classification
        self.node_classification = not graph_classification

        self.optimizer = None
        self.save = save_dir is not None
        self.save_dir = save_dir
        self.save_name = save_name

        self.train_mask = train_mask
        self.valid_mask = valid_mask
        self.test_mask = test_mask
        check_dirs(self.save_dir)

        if self.graph_classification:
            dataloader_param = kwargs.get("dataloader_param")
            self.loader = get_dataloader(dataset, **dataloader_param)

    def __loss__(self, logits, labels):
        return F.cross_entropy(logits, labels)

    def _train_batch(self, data, labels):
        if self.graph_classification:
            logits = self.model(data=data)
            logits = F.softmax(logits, -1)
            loss = self.__loss__(logits, labels)
        else:
            logits = self.model(x=data.x, edge_index=data.edge_index)
            if self.dataset_name in ['CS','Physics','Facebook']:
                loss = self.__loss__(logits[self.train_mask], labels[self.train_mask])
            else:
                loss = self.__loss__(logits[data.train_mask], labels[data.train_mask])

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=2.0)
        self.optimizer.step()
        return loss.item()

    def _eval_batch(self, data, labels, **kwargs):
        self.model.eval()
        if self.graph_classification:
            logits = self.model(data=data)
            logits = F.softmax(logits, -1)
            loss = self.__loss__(logits, labels)
        else:
            logits = self.model(x=data.x, edge_index=data.edge_index)
            mask = kwargs.get("mask")
            if mask is None:
                warnings.warn("The node mask is None")
                mask = torch.ones(labels.shape[0])
            loss = self.__loss__(logits[mask], labels[mask])
        loss = loss.item()
        preds = logits.argmax(-1)
        return loss, preds

    def eval(self):
        self.model.to(self.device)
        self.model.eval()
        if self.graph_classification:
            losses, accs = [], []
            for batch in self.loader["eval"]:
                batch = batch.to(self.device)
                loss, batch_preds = self._eval_batch(batch, batch.y)
                losses.append(loss)
                accs.append(batch_preds == batch.y)
            eval_loss = torch.tensor(losses).mean().item()
            eval_acc = torch.cat(accs, dim=-1).float().mean().item()
        else:
            data = self.dataset.to(self.device)
            if self.dataset_name in ['CS','Physics','Facebook']:
                eval_loss, preds = self._eval_batch(data, data.y, mask=self.valid_mask)
            else:
                eval_loss, preds = self._eval_batch(data, data.y, mask=data.val_mask)
            eval_acc = (preds == data.y).float().mean().item()
        return eval_loss, eval_acc

    def test(self):
        state_dict = torch.load(
            os.path.join(self.save_dir, f"{self.save_name}_best.pth")
        )["net"]
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()
        if self.graph_classification:
            losses, preds, accs = [], [], []
            for batch in self.loader["test"]:
                batch = batch.to(self.device)
                loss, batch_preds = self._eval_batch(batch, batch.y)
                losses.append(loss)
                preds.append(batch_preds)
                accs.append(batch_preds == batch.y)
            test_loss = torch.tensor(losses).mean().item()
            preds = torch.cat(preds, dim=-1)
            test_acc = torch.cat(accs, dim=-1).float().mean().item()
        else:
            data = self.dataset.to(self.device)
            if self.dataset_name in ['CS','Physics','Facebook']:
                test_loss, preds = self._eval_batch(data, data.y, mask=self.test_mask)
            else:
                test_loss, preds = self._eval_batch(data, data.y, mask=data.test_mask)
            test_acc = (preds == data.y).float().mean().item()
        print(f"Test loss: {test_loss:.4f}, test acc {test_acc:.4f}")
        return test_loss, test_acc, preds

    def train(self, train_param=None, optimizer_param=None):
        num_epochs = train_param["num_epochs"]
        num_early_stop = train_param["num_early_stop"]
        milestones = train_param["milestones"]
        gamma = train_param["gamma"]

        if optimizer_param is None:
            self.optimizer = Adam(self.model.parameters())
        else:
            self.optimizer = Adam(self.model.parameters(), **optimizer_param)

        if milestones is not None and gamma is not None:
            lr_schedule = MultiStepLR(
                self.optimizer, milestones=milestones, gamma=gamma
            )
        else:
            lr_schedule = None

        self.model.to(self.device)
        best_eval_acc = 0.0
        best_eval_loss = 0.0
        early_stop_counter = 0
        for epoch in range(num_epochs):
            is_best = False
            self.model.train()
            if self.graph_classification:
                losses = []
                for batch in self.loader["train"]:
                    batch = batch.to(self.device)
                    loss = self._train_batch(batch, batch.y)
                    losses.append(loss)
                train_loss = torch.FloatTensor(losses).mean().item()

            else:
                data = self.dataset.to(self.device)
                train_loss = self._train_batch(data, data.y)

            eval_loss, eval_acc = self.eval()
            print(
                f"Epoch:{epoch}, Training_loss:{train_loss:.4f}, Eval_loss:{eval_loss:.4f}, Eval_acc:{eval_acc:.4f}"
            )
            if num_early_stop > 0:
                if eval_loss <= best_eval_loss:
                    best_eval_loss = eval_loss
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                if epoch > num_epochs / 2 and early_stop_counter > num_early_stop:
                    break
            if lr_schedule:
                lr_schedule.step()

            if best_eval_acc < eval_acc:
                is_best = True
                best_eval_acc = eval_acc
            recording = {"epoch": epoch, "is_best": str(is_best)}
            if self.save:
                self.save_model(is_best, recording=recording)

    def save_model(self, is_best=False, recording=None):
        self.model.to("cpu")
        state = {"net": self.model.state_dict()}
        for key, value in recording.items():
            state[key] = value
        latest_pth_name = f"{self.save_name}_latest.pth"
        best_pth_name = f"{self.save_name}_best.pth"
        ckpt_path = os.path.join(self.save_dir, latest_pth_name)
        torch.save(state, ckpt_path)
        if is_best:
            print("saving best...")
            # shutil.copy(ckpt_path, os.path.join(self.save_dir, best_pth_name))
            torch.save(state, os.path.join(self.save_dir, best_pth_name))
        self.model.to(self.device)

    def load_model(self):
        state_dict = torch.load(
            os.path.join(self.save_dir, f"{self.save_name}_best.pth")
        )["net"]
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config):
    config.models.gnn_savedir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    config.models.param = config.models.param[config.datasets.dataset_name]
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(device)
    if config.models.param.graph_classification == False:
        if config.datasets.dataset_name in ['CS','Physics','Facebook']:
            dataset, train_mask, valid_mask, test_mask = get_dataset(config.datasets.dataset_root, config.datasets.dataset_name)
        else:
            dataset = get_dataset(config.datasets.dataset_root, config.datasets.dataset_name)
        num_node_features=dataset.num_node_features
        num_classes=dataset.num_classes
        dataset = dataset[0]
        dataset.x = dataset.x.float()
        if config.datasets.dataset_name in ['tree_grid','ba_shapes','Cora','CS','tree_cycle']:
            pass
        else:
            dataset.y = torch.argmax(dataset.y, dim=1)
        dataset.y = dataset.y.squeeze().long()

        gnn_model = get_gnnNets(num_node_features, num_classes, config.models)
        print("gnn_model structure:\n", gnn_model)
        train_param = {
            'num_epochs': config.models.param.num_epochs,
            'num_early_stop': config.models.param.num_early_stop,
            'milestones': config.models.param.milestones,
            'gamma': config.models.param.gamma
        }
        optimizer_param = {
            'lr': config.models.param.learning_rate,
            'weight_decay': config.models.param.weight_decay
        }
        if config.datasets.dataset_name in ['CS','Physics','Facebook']:
            trainer = TrainModel(
                model=gnn_model,
                dataset_name=config.datasets.dataset_name,
                dataset=dataset,
                device=device,
                graph_classification=config.models.param.graph_classification,
                save_dir=os.path.join(config.models.gnn_savedir, config.datasets.dataset_name),
                save_name=f'{config.models.gnn_name}_{len(config.models.param.gnn_latent_dim)}l',
                train_mask=train_mask,
                valid_mask=valid_mask,
                test_mask=test_mask,
            )
        else:
            trainer = TrainModel(
                model=gnn_model,
                dataset_name=config.datasets.dataset_name,
                dataset=dataset,
                device=device,
                graph_classification=config.models.param.graph_classification,
                save_dir=os.path.join(config.models.gnn_savedir, config.datasets.dataset_name),
                save_name=f'{config.models.gnn_name}_{len(config.models.param.gnn_latent_dim)}l',
            )
        trainer.train(train_param=train_param, optimizer_param=optimizer_param)
        _, _, _ = trainer.test()
    else:
        set_seed(config.random_seed)
        dataset = get_dataset(config.datasets.dataset_root, config.datasets.dataset_name)
        if dataset.data.x is not None:
            dataset.data.x = dataset.data.x.float()
        dataset.data.y = dataset.data.y.squeeze().long()
        dataloader_param = {
            'batch_size': config.models.param.batch_size,
            'data_split_ratio': config.datasets.data_split_ratio,
            'seed': config.datasets.seed
        }
        train_param = {
            'num_epochs': config.models.param.num_epochs,
            'num_early_stop': config.models.param.num_early_stop,
            'milestones': config.models.param.milestones,
            'gamma': config.models.param.gamma
        }
        optimizer_param = {
            'lr': config.models.param.learning_rate,
            'weight_decay': config.models.param.weight_decay
        }
        model = get_gnnNets(dataset.num_node_features, dataset.num_classes, config.models)

        trainer = TrainModel(model=model,
                             dataset=dataset,
                             dataset_name=config.datasets.dataset_name,
                             device=device,
                             save_dir=os.path.join(config.models.gnn_savedir, config.datasets.dataset_name),
                             save_name=f'{config.models.gnn_name}_{len(config.models.param.gnn_latent_dim)}l',
                             dataloader_param=dataloader_param)

        trainer.train(train_param, optimizer_param)
        _, _, _ = trainer.test()

if __name__ == '__main__':
    main()