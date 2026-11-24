import os

import numpy as np
import onnxruntime as ort
import torch

from config import ARTIFACTS_DIR
from model import DominanceNet
from train import SIZE


class Wrapper(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, frames, views):
        return self.net(frames, views)


def main():
    net = DominanceNet(pretrained=False)
    net.load_state_dict(torch.load(os.path.join(ARTIFACTS_DIR, "dominance_model.pt"), map_location="cpu"))
    net.eval()
    w = Wrapper(net).eval()
    frames = torch.randn(1, 3, 4, 3, SIZE, SIZE)
    views = torch.tensor([[0, 0, 1]])
    out = os.path.join(ARTIFACTS_DIR, "dominance_model.onnx")
    torch.onnx.export(
        w, (frames, views), out, input_names=["frames", "views"], output_names=["logits"],
        dynamic_axes={"frames": {1: "videos"}, "views": {1: "videos"}}, opset_version=17, dynamo=False,
    )
    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    for v in (3, 1, 5):  # parity incl. different video counts
        f = torch.randn(1, v, 4, 3, SIZE, SIZE)
        vw = torch.randint(0, 2, (1, v))
        ref = w(f, vw).detach().numpy()
        got = sess.run(["logits"], {"frames": f.numpy(), "views": vw.numpy()})[0]
        diff = float(np.abs(ref - got).max())
        assert diff < 1e-3, f"parity failed for {v} videos: {diff}"
    print("PASS", out)


if __name__ == "__main__":
    main()
