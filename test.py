import torch
import os
from tqdm import tqdm
from PIL import Image

from util.metric_tool import ConfuseMatrixMeter
from option import Options
from data.cd_dataset import DataLoader
from model.create_ChangeDINO import create_model

if __name__ == "__main__":
    opt = Options().parse()
    opt.phase = "test"
    test_loader = DataLoader(opt)
    test_data = test_loader.load_data()
    test_size = len(test_loader)
    print("#testing images = %d" % test_size)

    opt.load_pretrain = True
    model = create_model(opt)

    tbar = tqdm(test_data, ncols=80)
    total_iters = test_size
    running_metric = ConfuseMatrixMeter(n_class=2)
    running_metric.clear()

    test_save_path = os.path.join(opt.checkpoint_dir, opt.name, "pred")
    if opt.save_test and not os.path.exists(test_save_path):
        os.makedirs(test_save_path, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for i, _data in enumerate(tbar):
            val_pred = model.inference(_data["img1"].cuda(), _data["img2"].cuda())
            # update metric
            val_target = _data["cd_label"].detach()
            val_pred = torch.argmax(val_pred.detach(), dim=1)
            _ = running_metric.update_cm(
                pr=val_pred.cpu().detach().numpy(), gt=val_target.cpu().detach().numpy()
            )
            if opt.save_test:
                for j in range(val_pred.shape[0]):
                    pred = Image.fromarray((val_pred[j].cpu().detach().numpy()*255).astype("uint8"))
                    pred.save(
                        os.path.join(test_save_path, _data["fname"][j])
                    )
        val_scores = running_metric.get_scores()
        message = "(phase: %s) " % (opt.phase)
        for k, v in val_scores.items():
            message += "%s: %.4f " % (k, v * 100)
        print(message)
