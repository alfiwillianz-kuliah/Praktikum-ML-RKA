# Arsitektur model terbaik hasil Tugas 2 Deep Learning (2_DL_5054251028.ipynb).
# Salinan definisi class pada section 3 notebook, agar bisa dipakai ulang pada tugas Deployment.

import torch.nn as nn



class CNNBatchNorm(nn.Module):
    # CNN 3.2 + BatchNorm2d setelah setiap Conv, BatchNorm1d + Dropout sebelum layer output

    def __init__(self, n_classes=10, p_dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(128, n_classes),   # output logit
        )

    def forward(self, x):
        return self.classifier(self.features(x))
