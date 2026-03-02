import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# ----------------------------- Basic blocks -----------------------------
class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=None, groups=1, act=True):
        super().__init__()

        if padding is None:
            padding = kernel_size // 2

        groups = min(groups, in_ch)
        if out_ch % groups != 0:
            groups = 1

        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DepthwiseSeparable(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1):
        super().__init__()
        self.dw = ConvBNAct(in_ch, in_ch, kernel_size=kernel_size, stride=stride, groups=in_ch)
        self.pw = ConvBNAct(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.pw(self.dw(x))


# ----------------------------- Multi-scale stem -----------------------------
class MultiScaleStem(nn.Module):
    def __init__(self, in_ch=3, out_ch=64):
        super().__init__()

        half = out_ch // 2

        self.conv3 = ConvBNAct(in_ch, half, kernel_size=3)
        self.dw5 = ConvBNAct(in_ch, half, kernel_size=5, groups=3)
        self.conv1 = ConvBNAct(in_ch, half, kernel_size=1)

        self.project = ConvBNAct(half * 3, out_ch, kernel_size=1)

    def forward(self, x):

        a = self.conv3(x)
        b = self.dw5(x)
        c = self.conv1(x)

        cat = torch.cat([a, b, c], dim=1)

        return self.project(cat)


# ----------------------------- Encoders -----------------------------
class LocalEncoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.block = nn.Sequential(
            DepthwiseSeparable(in_ch, in_ch),
            ConvBNAct(in_ch, out_ch, kernel_size=1)
        )

    def forward(self, x):
        return self.block(x)


class GlobalEncoder(nn.Module):
    def __init__(self, dim, pool_size=7):
        super().__init__()

        self.pool_size = pool_size

        self.to_q = nn.Conv2d(dim, dim, 1)
        self.to_k = nn.Conv2d(dim, dim, 1)
        self.to_v = nn.Conv2d(dim, dim, 1)

        self.out = nn.Conv2d(dim, dim, 1)

    def forward(self, x):

        b, c, h, w = x.shape

        pooled = F.adaptive_avg_pool2d(x, (self.pool_size, self.pool_size))

        q = self.to_q(pooled).flatten(2)
        k = self.to_k(pooled).flatten(2)
        v = self.to_v(pooled).flatten(2)

        attn = torch.bmm(q.transpose(1,2), k)
        attn = attn.softmax(dim=-1)

        out = torch.bmm(v, attn.transpose(1,2)).view(b, c, self.pool_size, self.pool_size)

        out = F.interpolate(self.out(out), size=(h,w), mode='bilinear', align_corners=False)

        return out


class ChannelEncoder(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1,1,3,padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        z = self.avg(x).squeeze(-1).transpose(1,2)
        z = self.conv1d(z)
        z = self.sigmoid(z).transpose(1,2).unsqueeze(-1)

        return x * z


# ----------------------------- Cross Interaction -----------------------------
class CrossInteraction(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.modulator = nn.Sequential(
            nn.Conv2d(channels*3, channels, 1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, local, global_f, channel_f):

        cat = torch.cat([local, global_f, channel_f], dim=1)
        mod = self.modulator(cat)

        return local*mod, global_f*mod, channel_f*mod


# ----------------------------- CLG Block -----------------------------
class CLGFusionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.local = LocalEncoder(channels, channels)
        self.global_enc = GlobalEncoder(channels)
        self.channel_enc = ChannelEncoder(channels)

        self.cross = CrossInteraction(channels)

        self.fusion_logits = nn.Parameter(torch.tensor([1.0,1.0,1.0]))

    def forward(self,x):

        l = self.local(x)
        g = self.global_enc(x)
        c = self.channel_enc(x)

        l,g,c = self.cross(l,g,c)

        w = F.softmax(self.fusion_logits, dim=0)

        return w[0]*l + w[1]*g + w[2]*c + x


# ----------------------------- CLoGNet -----------------------------
class CLoGNet(nn.Module):

    def __init__(self, num_classes=2):

        super().__init__()

        self.stem = MultiScaleStem()

        self.body = nn.Sequential(
            CLGFusionBlock(64),
            nn.MaxPool2d(2),

            CLGFusionBlock(64),
            nn.MaxPool2d(2),

            CLGFusionBlock(64)
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64,32),
            nn.GELU(),
            nn.Linear(32,num_classes)
        )

    def forward(self,x):

        x = self.stem(x)
        x = self.body(x)
        x = self.head(x)

        return x


# ----------------------------- Load Model -----------------------------
device = torch.device("cpu")

model = CLoGNet()

checkpoint = torch.load("clognet_weights.pth", map_location=device)

model.load_state_dict(checkpoint["model_state"])

model.to(device)

model.eval()



# ----------------------------- Transform -----------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])


# ----------------------------- Prediction -----------------------------
def predict_image(image: Image.Image):

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(image)

        probs = torch.softmax(output, dim=1)

        confidence, pred = torch.max(probs,1)

    label = "Fake" if pred.item()==1 else "Real"

    return label, confidence.item()

