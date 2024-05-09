import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def save_to_csv(data, file_path):
    with open(file_path, 'w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerows(data)

def read_csv_file(file_path):
    data_type1 = []
    data_type2 = []
    data_type3 = []

    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            sha256_value = row[0]
            dex_pixel_values = row[1:16385]
            intent_values = row[16385:16485]
            permission_values = row[16485:16585]
            syscall_values = row[16585:16985]
            apk_type = row[16985]
            # Chimera_R_data
            type1_data = [sha256_value] + dex_pixel_values + [apk_type]
            data_type1.append(type1_data)
            # Chimera_S_data
            type2_data = [sha256_value] + intent_values + permission_values + [apk_type]
            data_type2.append(type2_data)
            # Chimera_D_data
            type3_data = [sha256_value] + syscall_values + [apk_type]
            data_type3.append(type3_data)

    return data_type1, data_type2, data_type3

def construct_data(csv_file_path):
    Chimera_R_data, Chimera_S_data, Chimera_D_data = read_csv_file(csv_file_path)
    save_to_csv(Chimera_R_data, 'Chimera_R_data.csv')
    save_to_csv(Chimera_S_data, 'Chimera_S_data.csv')
    save_to_csv(Chimera_D_data, 'Chimera_D_data.csv')
    print("Data saved successfully.")

# Chimera_S
class Chimera_S(nn.Module):
    def __init__(self):
        super(Chimera_S, self).__init__()
        self.input_layer = nn.Linear(200, 200)
        self.hidden_layer1 = nn.Linear(200, 256)
        self.hidden_layer2 = nn.Linear(256, 128)
        self.output_layer = nn.Linear(128, 2)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.relu(x)
        x = self.hidden_layer1(x)
        x = self.relu(x)
        x = self.hidden_layer2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        x = self.softmax(x)
        return x

def Chimera_S_training():
    data = pd.read_csv('Chimera_S_data.csv')
    inputs = data.iloc[:, 1:201].values
    labels = data.iloc[:, 201].values

    inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.001

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    fold = 0
    for train_index, test_index in kf.split(inputs):
        fold += 1
        print(f"Fold {fold}:")

        train_inputs, train_labels = inputs_tensor[train_index], labels_tensor[train_index]
        test_inputs, test_labels = inputs_tensor[test_index], labels_tensor[test_index]

        train_dataset = TensorDataset(train_inputs, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        test_dataset = TensorDataset(test_inputs, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Chimera_S().to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        model.train()
        for epoch in range(50):
            total_loss = 0
            for inputs, labels in tqdm(train_loader, desc=f"Training Epoch{epoch}"):
                optimizer.zero_grad()
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch [{epoch + 1}/50], Loss: {total_loss:.4f}")

        model.eval()
        with torch.no_grad():
            true_labels = []
            predicted_labels = []
            for inputs, labels in tqdm(test_loader, desc="Evaluating"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                true_labels.extend(labels.cpu().numpy())
                predicted_labels.extend(predicted.cpu().numpy())

        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels)
        recall = recall_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels)
        print(f"Test - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        print()

# Chimera_R
class Chimera_R(nn.Module):
    def __init__(self):
        super(Chimera_R, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=11, stride=2, padding=5)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=11, stride=2, padding=5)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=13, stride=2, padding=6)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=1, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, 2)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def Chimera_R_training():
    data = pd.read_csv('Chimera_R_data.csv')
    inputs = data.iloc[:, 1:16385].values
    labels = data.iloc[:, 16385].values

    inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    inputs_tensor = inputs_tensor.unsqueeze(1)

    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.001
    num_epochs = 50
    batch_size = 4

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    fold = 0
    for train_index, test_index in kf.split(inputs):
        fold += 1
        print(f"Fold {fold}:")

        train_inputs, train_labels = inputs_tensor[train_index], labels_tensor[train_index]
        test_inputs, test_labels = inputs_tensor[test_index], labels_tensor[test_index]

        train_dataset = TensorDataset(train_inputs, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = TensorDataset(test_inputs, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Chimera_R().to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for inputs, labels in tqdm(train_loader, desc=f"Training Epoch{epoch}"):
                optimizer.zero_grad()
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            true_labels = []
            predicted_labels = []
            for inputs, labels in tqdm(test_loader, desc="Evaluating"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                true_labels.extend(labels.cpu().numpy())
                predicted_labels.extend(predicted.cpu().numpy())

        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels)
        recall = recall_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels)
        print(f"Test - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        print()

# Chimera_D
class PositionalEncoder(nn.Module):
    def __init__(self, seq_length, input_size):
        super(PositionalEncoder, self).__init__()
        self.embedding = nn.Embedding(seq_length * 2, input_size)
        self.seq_length = seq_length

    def forward(self, x):
        device = x.device
        batch_size, seq_length, input_size = x.size()
        positions = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.embedding(positions)
        x = x + position_embeddings
        return x

def normalize_input_data(input_data, max_value=124):
    num_samples, seq_length = input_data.shape
    normalized_input = torch.zeros(num_samples, seq_length, max_value)
    for i, data in enumerate(input_data):
        for j, d in enumerate(data):
            normalized_input[i][j][d] = 1
    return normalized_input

class TransformerEncoderLayer(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_prob, num_heads):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=input_size, num_heads=num_heads, dropout=dropout_prob)
        self.feedforward = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )
        self.dropout = nn.Dropout(dropout_prob)
        self.layer_norm = nn.LayerNorm(input_size)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.layer_norm(x)
        x = x + self.dropout(self.feedforward(x))
        x = self.layer_norm(x)
        return x

class FullyConnected(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob):
        super(FullyConnected, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class Chimera_D(nn.Module):
    def __init__(self, seq_length, input_size, transformer_hidden_size, transformer_dropout_prob, transformer_num_heads,
                 fc_hidden_size, fc_dropout_prob):
        super(Chimera_D, self).__init__()
        self.positional_encoder = PositionalEncoder(seq_length, input_size)
        input_tensor_size = input_size * seq_length
        self.transformer_encoder_layer = TransformerEncoderLayer(input_size=input_tensor_size, hidden_size=transformer_hidden_size,
                                                                 dropout_prob=transformer_dropout_prob, num_heads=transformer_num_heads)
        self.fully_connected = FullyConnected(seq_length * input_size, fc_hidden_size, 2, fc_dropout_prob)

    def forward(self, x):
        x = self.positional_encoder(x)
        x = x.transpose(0, 1)
        x = self.transformer_encoder_layer(x)
        x = x.transpose(0, 1)
        x = x.view(x.size(0), -1)
        x = self.fully_connected(x)
        return x

def Chimera_D_training():
    data = pd.read_csv('Chimera_D_data.csv')
    inputs = data.iloc[:, 1:401].values
    inputs = normalize_input_data(inputs)
    labels = data.iloc[:, 401].values

    inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.001
    num_epochs = 30
    batch_size = 1

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    fold = 0
    for train_index, test_index in kf.split(inputs):
        fold += 1
        print(f"Fold {fold}:")

        train_inputs, train_labels = inputs_tensor[train_index], labels_tensor[train_index]
        test_inputs, test_labels = inputs_tensor[test_index], labels_tensor[test_index]

        train_dataset = TensorDataset(train_inputs, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = TensorDataset(test_inputs, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Chimera_D(seq_length=400, input_size=124, transformer_hidden_size=512, transformer_dropout_prob=0.2,
                          transformer_num_heads=4, fc_hidden_size=128, fc_dropout_prob=0.5).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for inputs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch}"):
                optimizer.zero_grad()
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss:.4f}")

        model.eval()
        with torch.no_grad():
            true_labels = []
            predicted_labels = []
            for inputs, labels in tqdm(test_loader, desc="Evaluating"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                true_labels.extend(labels.cpu().numpy())
                predicted_labels.extend(predicted.cpu().numpy())

        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels)
        recall = recall_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels)
        print(f"Test - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        print()



class Chimera_DNN(nn.Module):
    def __init__(self):
        super(Chimera_DNN, self).__init__()
        self.input_layer = nn.Linear(384, 384)
        self.hidden_layer = nn.Linear(384, 512)
        self.output_layer = nn.Linear(512, 2)
        self.dropout = nn.Dropout(p=0.3)
        self.batch_norm = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.relu(x)
        x = self.hidden_layer(x)
        x = self.relu(x)
        x = self.batch_norm(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        x = self.softmax(x)
        return x

def Chimera_training():
    data = pd.read_csv('Chimera_data.csv')
    inputs = data.iloc[:, 1:385].values
    labels = data.iloc[:, 385].values

    inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.001

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    fold = 0
    for train_index, test_index in kf.split(inputs):
        fold += 1
        print(f"Fold {fold}:")

        train_inputs, train_labels = inputs_tensor[train_index], labels_tensor[train_index]
        test_inputs, test_labels = inputs_tensor[test_index], labels_tensor[test_index]

        train_dataset = TensorDataset(train_inputs, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        test_dataset = TensorDataset(test_inputs, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Chimera_DNN().to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        model.train()
        for epoch in range(30):
            total_loss = 0
            for inputs, labels in tqdm(train_loader, desc=f"Training Epoch{epoch}"):
                optimizer.zero_grad()
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch [{epoch + 1}/30], Loss: {total_loss:.4f}")

        model.eval()
        with torch.no_grad():
            true_labels = []
            predicted_labels = []
            for inputs, labels in tqdm(test_loader, desc="Evaluating"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                true_labels.extend(labels.cpu().numpy())
                predicted_labels.extend(predicted.cpu().numpy())

        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels)
        recall = recall_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels)
        print(f"Test - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        print()