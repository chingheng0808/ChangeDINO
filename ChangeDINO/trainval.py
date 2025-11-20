import torch
from option import Options
from data.cd_dataset import DataLoader
from model.create_ChangeDINO import create_model
import torch.optim as optim
from tqdm import tqdm
import math
from util.metric_tool import ConfuseMatrixMeter
import os
import json
import numpy as np
import random
from datetime import datetime
from util.util import make_numpy_grid, de_norm
import matplotlib.pyplot as plt


def setup_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False 
    torch.backends.cudnn.benchmark = True  
    torch.backends.cudnn.enabled = True  


class Trainval(object):
    def __init__(self, opt):
        self.opt = opt

        train_loader = DataLoader(opt)
        self.train_data = train_loader.load_data()
        train_size = len(train_loader)
        print("#training images = %d" % train_size)
        opt.phase = "val"
        val_loader = DataLoader(opt)
        self.val_data = val_loader.load_data()
        val_size = len(val_loader)
        print("#validation images = %d" % val_size)
        opt.phase = "train"

        self.model = create_model(opt)
        self.optimizer = self.model.optimizer
        self.schedular = self.model.schedular

        self.iters = 0
        self.total_iters = math.ceil(train_size / opt.batch_size) * opt.num_epochs
        self.previous_best = 0.0
        self.running_metric = ConfuseMatrixMeter(n_class=2)
        self.alpha = 0.5

        self.log_path = os.path.join(self.model.save_dir, "record.txt")
        self.vis_path = os.path.join(self.model.save_dir, opt.vis_path)
        os.makedirs(self.vis_path, exist_ok=True)

        if not os.path.exists(self.log_path):
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write("# Record of training/validation metrics\n")
                f.write(
                    "# name: %s | backbone: %s\n"
                    % (opt.name, getattr(opt, "backbone", "NA"))
                )
                f.write("# time,epoch,train_loss,train_focal,train_dice,lr,")
                f.write("val_metrics(json)\n")
    
    def _rescheduler(self, opt):
        self.model.optimizer = optim.AdamW(
            self.model.model.parameters(), lr=opt.lr*0.2, weight_decay=opt.weight_decay
        )
        self.model.schedular = optim.lr_scheduler.CosineAnnealingLR(
            self.model.optimizer, int(opt.num_epochs*0.1), eta_min=1e-7
        )
        self.optimizer = self.model.optimizer
        self.schedular = self.model.schedular
        

    def _append_log_line(self, epoch: int, train_stats: dict, val_scores: dict):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        line = (
            f"{ts},{epoch},"
            f"{train_stats.get('loss', float('nan')):.6f},"
            f"{train_stats.get('focal', float('nan')):.6f},"
            f"{train_stats.get('dice', float('nan')):.6f},"
            f"{train_stats.get('lr', float('nan')):.8f},"
            + json.dumps(val_scores, ensure_ascii=False)
            + "\n"
        )
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line)

    def _plot_cd_result(self, x1, x2, pred, target, epoch, stage):
        if len(pred.shape) == 4:
            pred = torch.argmax(pred, dim=1)
        vis_input = make_numpy_grid(de_norm(x1[0:8]))
        vis_input2 = make_numpy_grid(de_norm(x2[0:8]))
        vis_pred = make_numpy_grid(pred[0:8].unsqueeze(1).repeat(1, 3, 1, 1))
        vis_gt = make_numpy_grid(target[0:8].unsqueeze(1).repeat(1, 3, 1, 1))
        vis = np.concatenate([vis_input, vis_input2, vis_pred, vis_gt], axis=0)
        vis = np.clip(vis, a_min=0.0, a_max=1.0)
        file_name = os.path.join(self.vis_path, f"{stage}_" + str(epoch) + ".jpg")
        plt.imsave(file_name, vis)

    def train(self, epoch):
        tbar = tqdm(self.train_data, ncols=80)
        opt.phase = "train"
        _loss = 0.0
        _focal_loss = 0.0
        _dice_loss = 0.0
        last_lr = self.optimizer.param_groups[0]["lr"]

        for i, data in enumerate(tbar):
            self.model.model.train()
            pred, focal, dice = self.model(
                data["img1"].cuda(), data["img2"].cuda(), data["cd_label"].cuda()
            )
            
            loss = focal * self.alpha + dice
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            _loss += loss.item()
            _focal_loss += focal.item()
            _dice_loss += dice.item()
            last_lr = self.optimizer.param_groups[0]["lr"]
            del loss

            tbar.set_description(
                "Loss: %.3f, Focal: %.3f, Dice: %.3f, LR: %.6f"
                % (
                    _loss / (i + 1),
                    _focal_loss / (i + 1),
                    _dice_loss / (i + 1),
                    last_lr,
                )
            )

            if i == len(tbar) - 1:
                self._plot_cd_result(
                    data["img1"], data["img2"], pred, data["cd_label"], epoch, "train"
                )
        self.schedular.step()

        n = max(1, i + 1)
        return {
            "loss": _loss / n,
            "focal": _focal_loss / n,
            "dice": _dice_loss / n,
            "lr": last_lr,
        }

    def val(self, epoch):
        tbar = tqdm(self.val_data, ncols=80)
        self.running_metric.clear()
        opt.phase = "val"
        self.model.eval()

        with torch.no_grad():
            for i, _data in enumerate(tbar):
                val_pred = self.model.inference(
                    _data["img1"].cuda(), _data["img2"].cuda()
                )
                val_target = _data["cd_label"].detach()
                val_pred = torch.argmax(val_pred.detach(), dim=1)
                _ = self.running_metric.update_cm(
                    pr=val_pred.cpu().numpy(), gt=val_target.cpu().numpy()
                )
                if i == len(tbar) - 1:
                    self._plot_cd_result(
                        _data["img1"],
                        _data["img2"],
                        val_pred,
                        _data["cd_label"],
                        epoch,
                        "val",
                    )
            val_scores = self.running_metric.get_scores()
            message = "(phase: %s) " % (self.opt.phase)
            for k, v in val_scores.items():
                message += "%s: %.3f " % (k, v * 100)
            print(message)

        if val_scores.get("iou_1", 0.0) >= self.previous_best:
            self.model.save(self.opt.name, self.opt.backbone)
            self.previous_best = val_scores["iou_1"]

        return val_scores


if __name__ == "__main__":
    opt = Options().parse()
    trainval = Trainval(opt)
    setup_seed(seed=1)

    for epoch in range(1, opt.num_epochs + 1):
        print(
            "\n==> Name %s, Epoch %i, previous best = %.3f"
            % (opt.name, epoch, trainval.previous_best * 100)
        )
        if epoch == int(opt.num_epochs*0.9):
            trainval._rescheduler(opt)
        train_stats = trainval.train(epoch)
        val_scores = trainval.val(epoch)

        trainval._append_log_line(epoch, train_stats, val_scores)

    print("Done!")
