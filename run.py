import os
from argparse import ArgumentParser

import torch
from PIL import Image
from torchvision import transforms

from model.create_ChangeDINO import create_model
from option import Options


def build_parser() -> ArgumentParser:
    """
    Reuse the training options so checkpoints/backbone configs stay in sync,
    and extend with paths for single-pair inference.
    """
    opt_builder = Options()
    opt_builder.init()
    parser = opt_builder.parser
    parser.set_defaults(name="WHU-ChangeDINO")
    parser.add_argument("--img_A", required=True, help="Path to time-A image.")
    parser.add_argument("--img_B", required=True, help="Path to time-B image.")
    parser.add_argument(
        "--output",
        type=str,
        default="./outputs/run_pred.png",
        help="Where to save the binary prediction mask (0/255).",
    )
    return parser


def parse_and_prepare() -> object:
    parser = build_parser()
    opt = parser.parse_args()

    str_ids = opt.gpu_ids.split(",")
    opt.gpu_ids = []
    for str_id in str_ids:
        gid = int(str_id)
        if gid >= 0:
            opt.gpu_ids.append(gid)
    if not opt.gpu_ids:
        raise ValueError("gpu_ids must include at least one GPU id (e.g., 0).")

    if torch.cuda.is_available():
        torch.cuda.set_device(opt.gpu_ids[0])
    else:
        raise EnvironmentError("CUDA is not available but gpu_ids were provided.")

    opt.phase = "test"
    opt.load_pretrain = True
    opt.batch_size = 1
    opt.num_workers = 0

    print("------------ Options -------------")
    for k, v in sorted(vars(opt).items()):
        print(f"{k}: {v}")
    print("-------------- End ----------------")

    return opt


def load_image(path, to_tensor, normalize):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} does not exist.")
    img = Image.open(path).convert("RGB")
    tensor = normalize(to_tensor(img)).unsqueeze(0)
    return img, tensor


def main():
    opt = parse_and_prepare()

    os.makedirs(os.path.dirname(opt.output) or ".", exist_ok=True)
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize((0.430, 0.411, 0.296), (0.213, 0.156, 0.143))

    _, img_A = load_image(opt.img_A, to_tensor, normalize)
    _, img_B = load_image(opt.img_B, to_tensor, normalize)

    img_A = img_A.cuda(non_blocking=True)
    img_B = img_B.cuda(non_blocking=True)

    model = create_model(opt)
    model.eval()

    with torch.no_grad():
        pred = model.inference(img_A, img_B)
        pred = torch.argmax(pred, dim=1)
        pred_img = Image.fromarray(
            (pred[0].cpu().detach().numpy() * 255).astype("uint8")
        )
        pred_img.save(opt.output)
        print(f"Saved prediction to {opt.output}")


if __name__ == "__main__":
    main()
