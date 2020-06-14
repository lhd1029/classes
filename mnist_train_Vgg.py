from torch import optim
import torch.nn as nn
import torch
import numpy as np
import os
import gzip


def load_data(data_folder):
    # 设置文件名
    files = [
        'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
    ]

    paths = []
    for fname in files:
        paths.append(os.path.join(data_folder, fname))
    # 打开训练集标签
    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    # 打开训练集图片
    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 1, 28, 28)
    # 打开测试集标签
    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    # 打开测试集图片
    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 1, 28, 28)

    return (x_train, y_train), (x_test, y_test)


(train_images, train_labels), (test_images, test_labels) = load_data('mnist')


device = torch.device('cpu')

num_epochs = 10  # 批数
num_classes = 10  # 分类数
batch_size = 128  # 每一批的图片数
learning_rate = 0.001  # 学习率


mode = 1  # mode=1是初始训练，mode=2是FineTune训练


class Net (nn.Module):  # VGG网络结构
    def __init__(self, num_classes=10):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1)
        )
        self.fc6 = nn.Sequential(nn.Linear(4*4*512, 4096), nn.ReLU())
        self.fc7 = nn.Sequential(nn.Linear(4096, 1000), nn.ReLU())
        self.fc8 = nn.Sequential(nn.Softmax(1000, num_classes))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc6(out)
        out = self.fc7(out)
        out = self.fc8(out)
        return out


model = Net(num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
compute_loss = nn.CrossEntropyLoss()


if mode == 2:  # FineTune训练
    test_my_images = np.array([])
    test_my_labels = np.array([], dtype=int)
    for i in range(10):
        for j in range(1, 6):
            frompath = "my_dataset_test/" + str(i) + str(j) + ".jpg"
            img = Image.open(frompath)
            test_my_images = np.append(
                test_my_images, np.array(np.asarray(img)))
            test_my_labels = np.append(test_my_labels, int(i))

    train_my_images = np.array([])
    train_my_labels = np.array([], dtype=int)
    for i in range(10):
        for j in range(1, 16):
            frompath = "my_dataset_train/" + str(i) + str(j) + ".jpg"
            img = Image.open(frompath)
            train_my_images = np.append(
                train_my_images, np.array(np.asarray(img)))
            train_my_labels = np.append(train_my_labels, int(i))
    train_my_image = np.reshape(train_my_images, (int(len(
        train_my_images) / PHOTOSIZE / PHOTOSIZE), 1, PHOTOSIZE, PHOTOSIZE))
    test_my_image = np.reshape(test_my_images, (int(len(
        test_my_images) / PHOTOSIZE / PHOTOSIZE), 1, PHOTOSIZE, PHOTOSIZE))

    train_my_image = torch.from_numpy(train_my_image) / 255.0
    train_my_label = torch.from_numpy(train_my_labels)
    train_my_image = torch.tensor(train_my_image, dtype=torch.float32)
    train_my_label = torch.tensor(train_my_label, dtype=torch.long)

    test_my_image = torch.from_numpy(test_my_image) / 255.0
    test_my_label = torch.from_numpy(test_my_labels)
    test_my_image = torch.tensor(test_my_image, dtype=torch.float32)
    test_my_label = torch.tensor(test_my_label, dtype=torch.long)

    checkpoint = torch.load(
        "D:/learning/coding/vscode/pythontest/LeNet5_model.pth.tar")
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['opt'])
    trainblae_vars = list(model.layer1.parameters())  # 只重新训练该层网络
    optimizer = optim.Adam(trainblae_vars, lr=learning_rate)


# 正则化
train_image = torch.from_numpy(train_image) / 255.0
train_label = torch.from_numpy(train_labels)
train_image = torch.tensor(train_image, dtype=torch.float32)
train_label = torch.tensor(train_label, dtype=torch.long)

test_image = torch.from_numpy(test_image) / 255.0
test_label = torch.from_numpy(test_labels)
test_image = torch.tensor(test_image, dtype=torch.float32)
test_label = torch.tensor(test_label, dtype=torch.long)


for epoch in range(num_epochs):
    for i in range(int(60000 / batch_size)):
        train_image_epoch = train_image[i *
                                        batch_size:(i + 1) * batch_size]
        train_label_epoch = train_label[i *
                                        batch_size:(i + 1) * batch_size]
        outputs = model(train_image_epoch)
        loss = compute_loss(outputs, train_label_epoch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch +
                                               1, num_epochs, loss.item()))
model.eval()
if mode == 1:
    saved_dict = {
        'model': model.state_dict(),
        'opt': optimizer.state_dict()
    }
    torch.save(saved_dict, "/Vgg16_model.pth.tar")

    with torch.no_grad():
        correct = 0
        outputs = model(test_image)
        _, prediction = torch.max(outputs.data, 1)
        correct = prediction - test_label
        correct = sum(correct == 0)
        correct = float(correct) / len(test_label)
        print("准确率：{:.4f}".format(correct))
else:
    with torch.no_grad():
        correct = 0

        outputs = model(test_my_image)
        _, prediction = torch.max(outputs.data, 1)
        correct = prediction - test_my_label
        correct = sum(correct == 0)
        correct = float(correct) / len(test_my_label)
        print("自编测试集准确率：{:.4f}".format(correct))
        # 计算混淆矩阵
        confustion_matrix = []
        for i in range(10):
            temp = []
            for j in range(10):
                temp.append(0)
            confustion_matrix.append(temp)
        for i in range(len(test_my_label)):
            confustion_matrix[test_my_label[i]][prediction[i]] += 1
        print(confustion_matrix)
