from torch import nn


class AsymProbAdaptor(nn.Module):
    def __init__(self):
        super(AsymProbAdaptor, self).__init__()
        # Text network
        mod_txt = []
        mod_txt.append(nn.Linear(512, 512))
        mod_txt.append(nn.BatchNorm1d(512))
        mod_txt.append(nn.ReLU())
        mod_txt.append(nn.Linear(512, 512))
        mod_txt.append(nn.BatchNorm1d(512))
        mod_txt.append(nn.ReLU())
        mod_txt.append(nn.Linear(512, 512))

        self.mod_txt = nn.Sequential(*mod_txt)

        self.txt_log_var = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        # Image network
        mod_img = []
        mod_img.append(nn.Linear(512, 512))
        mod_img.append(nn.BatchNorm1d(512))
        mod_img.append(nn.ReLU())
        mod_img.append(nn.Linear(512, 512))
        mod_img.append(nn.BatchNorm1d(512))
        mod_img.append(nn.Dropout(p=0.3))
        mod_img.append(nn.ReLU())
        mod_img.append(nn.Linear(512, 512))

        self.mod_img = nn.Sequential(*mod_img)

        self.img_mu = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

    def forward(self, xfT=None, xfI=None):
        if xfT is not None:
            txt_mu = xfT
            txt_log_var = self.mod_txt(self.txt_log_var(xfT))
        else:
            txt_mu, txt_log_var = None, None
        if xfI is not None:
            xfI = xfI.half()
            img_mu = xfI + self.mod_img(self.img_mu(xfI))
        else:
            img_mu = None
        return txt_mu, txt_log_var, img_mu

    def loss(self, txt_mu, txt_log_var, img_mu, image_features):
        image_features = image_features.repeat(5, 1)
        img_mu = img_mu.repeat(5, 1)
        loss1 = ((txt_mu - image_features) ** 2 / txt_log_var.exp()).sum(dim=1) / 2
        loss2 = txt_log_var.sum(dim=1) / 2
        loss3 = ((img_mu - image_features) ** 2).sum(dim=1) / 2
        lambda_loss3 = 0.5

        return (loss1 + loss2 + lambda_loss3 * loss3).mean()

