"""Standalone DIN recommender trainer (ML-1M, OOD splits) -- prep, not yet run.

Mirrors the hardened baseline_train_{mf,sasrec}_ood.py pattern: env-var paths,
warning suppression, throttled logging, checkpoint/resume with atomic saves and
explicit stop-reason logging. There was no DIN baseline script in the original
repo, so this is a new file (data plumbing follows baseline_train_sasrec.py,
which shares the `his` history-sequence format).

Model notes (see minigpt4/models/rec_base_models.py: RecEncoder_DIN):
- forward((sparse, seqs)) with sparse=(N,2) [uid, iid], seqs=(N, maxlen) history.
- The final FullyConnectedLayer has sigmoid=True, so outputs are probabilities
  in (0,1) -> use BCELoss here, NOT BCEWithLogitsLoss (SASRec/LightGCN/MF emit
  raw logits and use BCEWithLogitsLoss).
- embedding_size must be divisible by 3 (split into uid/iid/his sub-embeddings).

Default recipe (lr=1e-4, wd=1e-4, emb=192, drop=0.2, maxlen=10) is taken from the
authors' own stale-config checkpoint filename
(1010_ml_DIN_lr0.0001-dp-0.2-wd-0.0001...pt in collm_pretrain_din_ood_cc.yaml).
"""
from minigpt4.models.rec_base_models import RecEncoder_DIN
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import pandas as pd
import numpy as np
import torch
import torch.optim
from sklearn.metrics import roc_auc_score
import torch.nn as nn
import torch.nn.functional as F
import omegaconf
import random
import warnings

import os
import time

warnings.filterwarnings("ignore", message="Only one class is present in y_true")

# Kaggle paths: set COLLM_ML1M_DIR (dir with *_ood2.pkl) and COLLM_OUTPUT_DIR (dir for the .pth).
ML1M_DATA_DIR = os.environ.get("COLLM_ML1M_DIR", "/kaggle/working/ml-1m/")
DIN_SAVE_DIR = os.environ.get("COLLM_OUTPUT_DIR", "/kaggle/working/")


def uAUC_me(user, predict, label, verbose=True):
    if not isinstance(predict,np.ndarray):
        predict = np.array(predict)
    if not isinstance(label,np.ndarray):
        label = np.array(label)
    predict = predict.squeeze()
    label = label.squeeze()

    start_time = time.time()
    u, inverse, counts = np.unique(user,return_inverse=True,return_counts=True) # sort in increasing
    index = np.argsort(inverse)
    candidates_dict = {}
    k = 0
    total_num = 0
    only_one_interaction = 0
    computed_u = []
    for u_i in u:
        start_id,end_id = total_num, total_num+counts[k]
        u_i_counts = counts[k]
        index_ui = index[start_id:end_id]
        if u_i_counts ==1:
            only_one_interaction += 1
            total_num += counts[k]
            k += 1
            continue
        # print(index_ui, predict.shape)
        candidates_dict[u_i] = [predict[index_ui], label[index_ui]]
        total_num += counts[k]

        k+=1
    if verbose:
        print("only one interaction users:",only_one_interaction)
    auc=[]
    only_one_class = 0

    for ui,pre_and_true in candidates_dict.items():
        pre_i,label_i = pre_and_true
        try:
            ui_auc = roc_auc_score(label_i,pre_i)
            auc.append(ui_auc)
            computed_u.append(ui)
        except:
            only_one_class += 1
            # print("only one class")

    auc_for_user = np.array(auc)
    if verbose:
        print("computed user:", auc_for_user.shape[0], "can not users:", only_one_class)
    uauc = auc_for_user.mean()
    if verbose:
        print("uauc for validation Cost:", time.time()-start_time,'uauc:', uauc)
    return uauc, computed_u, auc_for_user


class seq_dataset(Dataset):
    # Same history-sequence format as baseline_train_sasrec.py: each row is
    # (uid, iid, his, label) with his a variable-length item-id history.
    def __init__(self,data,max_len=10):
        # super.__init__()
        self.data = data
        self.max_len = max_len
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        data_i = self.data[index]
        uid, iid, his, labels = data_i[0], data_i[1], data_i[2], data_i[3]
        if len(his) < self.max_len:
            his_ = np.zeros(self.max_len)
            his_[-len(his):] = np.array(his)
            his = his_
        elif len(his) > self.max_len:
            his = np.array(his)[-self.max_len:]
        else:
            his = np.array(his)
        return  uid, iid, his, labels


class seq_dataset_eval(Dataset):
    def __init__(self,data,max_len=10):
        # super.__init__()
        self.data = data #pd.read_pickle(data_path).values
        self.max_len = max_len
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        data_i = self.data[index]
        uid, iid, his, labels = data_i[0], data_i[1], data_i[2], data_i[3]
        if len(his) < self.max_len:
            his_ = np.zeros(self.max_len)
            his_[-len(his):] = np.array(his)
            his = his_
        elif len(his) > self.max_len:
            his = np.array(his)[-self.max_len:]
        else:
            his = np.array(his)
        return  uid, iid, his, labels


class early_stoper(object):
    def __init__(self,ref_metric='valid_auc', incerase =True,patience=20) -> None:
        self.ref_metric = ref_metric
        self.best_metric = None
        self.increase = incerase
        self.reach_count = 0
        self.patience= patience
        # self.metrics = None

    def _registry(self,metrics):
        self.best_metric = metrics

    def update(self, metrics):
        if self.best_metric is None:
            self._registry(metrics)
            return True
        else:
            if self.increase and metrics[self.ref_metric] > self.best_metric[self.ref_metric]:
                self.best_metric = metrics
                self.reach_count = 0
                return True
            elif not self.increase and metrics[self.ref_metric] < self.best_metric[self.ref_metric]:
                self.best_metric = metrics
                self.reach_count = 0
                return True
            else:
                self.reach_count += 1
                return False

    def state_dict(self):
        return {
            "best_metric": self.best_metric,
            "reach_count": self.reach_count,
            "patience": self.patience,
        }

    def load_state_dict(self, ckpt):
        self.best_metric = ckpt.get("best_metric")
        self.reach_count = ckpt.get("reach_count", 0)
        self.patience = ckpt.get("patience", self.patience)

    def is_stop(self):
        if self.reach_count>=self.patience:
            return True
        else:
            return False

def _atomic_torch_save(obj, path):
    # Atomic save (tmp file + os.replace) so a crash/session-kill can never
    # leave a half-written .pth behind. Same pattern as the CoLLM runner.
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)

def _din_forward(model, batch):
    # batch: (uid[B], iid[B], his[B, maxlen], label[B]) -> probs[B].
    # RecEncoder_DIN.forward takes ((sparse, seqs)) with sparse=(B,2).
    uid, iid, his = batch[0].long(), batch[1].long(), batch[2].long()
    sparse = torch.stack([uid, iid], dim=1)
    return model((sparse, his)).reshape(-1)

# set random seed
def run_a_trail(train_config,log_file=None, save_mode=False,save_file=None,need_train=True,warm_or_cold=None,resume=False,print_every=10,ckpt_freq=10):
    seed=2023
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # load dataset
    data_dir = ML1M_DATA_DIR
    train_data = pd.read_pickle(data_dir+"train_ood2.pkl")[['uid','iid', 'his', 'label']].values
    valid_data = pd.read_pickle(data_dir+"valid_ood2.pkl")[['uid','iid', 'his', 'label']].values
    test_data = pd.read_pickle(data_dir+"test_ood2.pkl")[['uid','iid', 'his', 'label']].values

    user_num = max(train_data[:,0].max(),valid_data[:,0].max(), test_data[:,0].max()) + 1
    item_num = max(train_data[:,1].max(),valid_data[:,1].max(), test_data[:,1].max()) + 1

    if warm_or_cold is not None:
        if warm_or_cold == 'warm':
            test_data = pd.read_pickle(data_dir+"test_ood2.pkl")[['uid','iid', 'his', 'label', 'not_cold']]
            test_data = test_data[test_data['not_cold'].isin([1])][['uid','iid','his', 'label']].values
            print("warm data size:", test_data.shape[0])
            # pass
        else:
            test_data = pd.read_pickle(data_dir+"test_ood2.pkl")[['uid','iid','his','label', 'not_cold']]
            test_data = test_data[test_data['not_cold'].isin([0])][['uid','iid','his','label']].values
            print("cold data size:", test_data.shape[0])
            # pass

    print("user nums:", user_num, "item nums:", item_num)

    assert int(train_config['embedding_size']) % 3 == 0, \
        "DIN embedding_size must be divisible by 3, got {}".format(train_config['embedding_size'])

    train_data = seq_dataset(train_data,max_len=int(train_config['maxlen']))
    valid_data = seq_dataset_eval(valid_data,max_len=int(train_config['maxlen']))
    test_data = seq_dataset_eval(test_data,max_len=int(train_config['maxlen']))

    din_config={
        "user_num": int(user_num),
        "item_num": int(item_num),
        "embedding_size": int(train_config['embedding_size']),
        "embed_size": int(train_config['embedding_size']),
        "drop": float(train_config['drop']),
        "l2_emb": 1e-4, # not used
        "maxlen": int(train_config['maxlen']),
        }
    print("din_config:\n", din_config)
    din_config = omegaconf.OmegaConf.create(din_config)

    train_data_loader = DataLoader(train_data, batch_size = train_config['batch_size'], shuffle=True)
    valid_data_loader = DataLoader(valid_data, batch_size = train_config['batch_size'], shuffle=False)
    test_data_loader = DataLoader(test_data, batch_size = train_config['batch_size'], shuffle=False)

    model = RecEncoder_DIN(din_config).cuda()
    print("model device:", next(model.parameters()).device)

    opt = torch.optim.Adam(model.parameters(),lr=train_config['lr'], weight_decay=train_config['wd'])
    early_stop = early_stoper(ref_metric='valid_auc',incerase=True,patience=train_config['patience'])
    ckpt_path = os.path.splitext(save_file)[0] + "_resume.pth" if save_file else None
    # DIN outputs post-sigmoid probabilities -> BCELoss (cf. BCEWithLogitsLoss
    # for the logit-output MF/LightGCN/SASRec baselines).
    criterion = nn.BCELoss()

    if not need_train:
        model.load_state_dict(torch.load(save_file, weights_only=False))
        model.eval()
        pre=[]
        label = []
        users = []
        for batch_id,batch_data in enumerate(valid_data_loader):
            batch_data = [x_.cuda() for x_ in batch_data]
            ui_matching = _din_forward(model, batch_data)
            pre.extend(ui_matching.detach().cpu().numpy())
            label.extend(batch_data[-1].cpu().numpy())
            users.extend(batch_data[0].cpu().numpy())
        valid_auc = roc_auc_score(label,pre)
        valid_uauc, _, _ = uAUC_me(users,pre,label)
        label = np.array(label)
        pre = np.array(pre)
        thre = 0.1
        pre[pre>=thre] =  1
        pre[pre<thre]  =0
        val_acc = (label==pre).mean()

        pre=[]
        label = []
        users = []
        for batch_id,batch_data in enumerate(test_data_loader):
            batch_data = [x_.cuda() for x_ in batch_data]
            ui_matching = _din_forward(model, batch_data)
            pre.extend(ui_matching.detach().cpu().numpy())
            label.extend(batch_data[-1].cpu().numpy())
            users.extend(batch_data[0].cpu().numpy())
        test_auc = roc_auc_score(label,pre)
        test_uauc, _, _ = uAUC_me(users,pre,label)
        print("valid_auc:{}, valid_uauc:{}, test_auc:{}, test_uauc: {} acc: {}".format(valid_auc, valid_uauc, test_auc, test_uauc, val_acc))
        return


    start_epoch = 0
    if resume and ckpt_path is not None and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if ckpt.get("done"):
            print("resume: training already finished (epoch {}, reason: {}). Skipping.".format(ckpt["epoch"], ckpt.get("stop_reason")))
            print("resume: best result:", ckpt.get("best_metric"))
            return
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        early_stop.load_state_dict(ckpt)
        start_epoch = ckpt["epoch"] + 1
        print("resumed checkpoint at epoch {}; best_valid_auc={}; early_count={}".format(
            ckpt["epoch"], (ckpt.get("best_metric") or {}).get("valid_auc"), early_stop.reach_count))

    def _save_ckpt(epoch, done=False, reason=None):
        if ckpt_path is None:
            return
        _atomic_torch_save({
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "epoch": epoch,
            "done": done,
            "stop_reason": reason,
            "best_metric": early_stop.best_metric,
            "reach_count": early_stop.reach_count,
            "patience": early_stop.patience,
            "train_config": train_config,
        }, ckpt_path)

    last_epoch = start_epoch
    stop_reason = None
    for epoch in range(start_epoch, train_config['epoch']):
        last_epoch = epoch
        model.train()
        for bacth_id, batch_data in enumerate(train_data_loader):
            batch_data = [x_.cuda() for x_ in batch_data]
            ui_matching = _din_forward(model, batch_data)
            loss = criterion(ui_matching, batch_data[-1].float().reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()

        if epoch% train_config['eval_epoch']==0:
            model.eval()
            pre=[]
            label = []
            users = []
            for batch_id, batch_data in enumerate(valid_data_loader):
                batch_data = [x_.cuda() for x_ in batch_data]
                ui_matching = _din_forward(model, batch_data)
                pre.extend(ui_matching.detach().cpu().numpy())
                label.extend(batch_data[-1].cpu().numpy())
                users.extend(batch_data[0].cpu().numpy())
            valid_auc = roc_auc_score(label,pre)
            valid_uauc, _, _ = uAUC_me(users,pre,label, verbose=(epoch % print_every == 0))

            pre=[]
            label = []
            users = []
            for batch_id,batch_data in enumerate(test_data_loader):
                batch_data = [x_.cuda() for x_ in batch_data]
                ui_matching = _din_forward(model, batch_data)
                pre.extend(ui_matching.detach().cpu().numpy())
                label.extend(batch_data[-1].cpu().numpy())
                users.extend(batch_data[0].cpu().numpy())
            test_auc = roc_auc_score(label,pre)
            test_uauc, _, _ = uAUC_me(users,pre,label, verbose=(epoch % print_every == 0))
            updated = early_stop.update({'valid_auc':valid_auc, 'valid_uauc':valid_uauc, 'test_auc':test_auc, 'test_uauc':test_uauc, 'epoch':epoch})
            if updated and save_mode:
                _atomic_torch_save(model.state_dict(),save_file)


            if epoch % print_every == 0:
                print("epoch:{}, valid_auc:{}, test_auc:{}, early_count:{}".format(epoch, valid_auc, test_auc, early_stop.reach_count))
            if epoch % ckpt_freq == 0:
                _save_ckpt(epoch)
            if early_stop.is_stop():
                print("early stop is reached....!")
                stop_reason = "early_stop (valid_auc made no improvement for {} eval epochs)".format(early_stop.patience)
                _save_ckpt(epoch, done=True, reason=stop_reason)
                # print("best results:", early_stop.best_metric)
                break
            if epoch>500 and early_stop.best_metric[early_stop.ref_metric] < 0.52:
                print("training reaches to 500 epoch but the valid_auc is still less than 0.55")
                stop_reason = "low_auc_guard (epoch>500 and valid_auc<0.52)"
                _save_ckpt(epoch, done=True, reason=stop_reason)
                break
    if stop_reason is None:
        stop_reason = "finished all {} epochs with no early stop".format(train_config['epoch'])
        _save_ckpt(last_epoch, done=True, reason=stop_reason)
    best = early_stop.best_metric or {}
    print("Training stopped due to: {}, at epoch {}, best_valid_auc={}".format(stop_reason, last_epoch, best.get("valid_auc")))
    print("train_config:", train_config,"\nbest result:",early_stop.best_metric)
    if log_file is not None:
        print("Training stopped due to: {}, at epoch {}, best_valid_auc={}".format(stop_reason, last_epoch, best.get("valid_auc")), file=log_file)
        print("train_config:", train_config, "best result:", early_stop.best_metric, file=log_file)
        log_file.flush()


# save-best training run: authors' implied DIN recipe (lr=1e-4, wd=1e-4, d=192,
# drop=0.2, maxlen=10 -- from the 1010_ml_DIN checkpoint filename in the stale
# collm_pretrain_din_ood_cc.yaml), no warm/cold eval. Paths come from
# COLLM_ML1M_DIR / COLLM_OUTPUT_DIR.
if __name__=='__main__':
    # lr_ = [1e-1,1e-2,1e-3]
    lr_=[1e-4]
    dw_ = [1e-4]
    # embedding_size_ = [32, 64, 128, 156, 512]
    embedding_size_ = [192]
    f=None
    for lr in lr_:
        for wd in dw_:
            for embedding_size in embedding_size_:
                train_config={
                    'lr': lr,
                    'wd': wd,
                    'embedding_size': embedding_size,
                    "epoch": 5000,
                    "eval_epoch":1,
                    "patience":100,
                    "batch_size":2048,
                    "maxlen": 10,
                    "drop": 0.2,
                }
                print(train_config)
                save_file = os.path.join(DIN_SAVE_DIR, "1010_din_ml1m_best_model_d" + str(embedding_size)+ 'lr-'+ str(lr) + "wd"+str(wd) +"len"+str(train_config['maxlen']) + ".pth")
                print("save path: ", save_file)
                run_a_trail(train_config=train_config, log_file=f, save_mode=True, save_file=save_file, need_train=True, warm_or_cold=None, resume=True)
    if f is not None:
        f.close()
