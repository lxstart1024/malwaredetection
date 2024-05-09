import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from tqdm import tqdm

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        return x

class SENet(nn.Module):
    def __init__(self):
        super(SENet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64 * 13 * 13, 64 * 13 * 13)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.view(x.size(0), 64, 13, 13)
        return x

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.resnet = ResNet()
        self.senet = SENet()
        self.fc = nn.Linear(64 * 13 * 13, 2)

    def forward(self, x):
        resnet_output = self.resnet(x)
        senet_output = self.senet(x)
        combined_output = resnet_output + senet_output
        x = combined_output.view(combined_output.size(0), -1)
        x = self.fc(x)
        return x


def evaluate_model(model, inputs, labels):
    with torch.no_grad():
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        accuracy = accuracy_score(labels.cpu().numpy(), predicted.cpu().numpy())
        precision = precision_score(labels.cpu().numpy(), predicted.cpu().numpy())
        recall = recall_score(labels.cpu().numpy(), predicted.cpu().numpy())
        f1 = f1_score(labels.cpu().numpy(), predicted.cpu().numpy())

        return accuracy, precision, recall, f1, predicted


def train_and_evaluate(file_path):
    data = pd.read_csv(file_path)
    inputs = data.iloc[:, 1:101].values
    labels = data.iloc[:, 101].values

    num_samples = len(inputs)
    inputs_13x13 = np.zeros((num_samples, 1, 13, 13))
    for i in range(num_samples):
        img = inputs[i].reshape(10, 10)
        inputs_13x13[i, 0, :10, :10] = img

    inputs_tensor = torch.tensor(inputs_13x13, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    fold = 0
    for train_index, test_index in kf.split(inputs):
        fold += 1
        print(f"Fold {fold}:")

        train_inputs, train_labels = inputs_tensor[train_index], labels_tensor[train_index]
        test_inputs, test_labels = inputs_tensor[test_index], labels_tensor[test_index]

        model = Model()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        model.train()
        for epoch in range(10):
            optimizer.zero_grad()
            outputs = model(train_inputs)
            loss = criterion(outputs, train_labels)
            loss.backward()
            optimizer.step()

        model.eval()
        accuracy, precision, recall, f1, _ = evaluate_model(model, test_inputs, test_labels)
        print(f"Test Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")