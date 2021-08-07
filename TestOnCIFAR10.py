#python3

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import sklearn.metrics as metrics
from torchvision import datasets, transforms
import DenseNet
import sys 


def train(model, device, optimizer, train_loader, epoch, lossF):
  model.train()
  total_loss = 0
  total_images = 0
  total_correct = 0
  for batch in train_loader:
    images, labels= batch
    images, labels = images.to(device), labels.to(device)
    
    pred_y = model(images)
    loss = lossF(pred_y, labels)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    output = pred_y.argmax(dim=1)

    total_loss += loss.item()
    total_images += labels.size(0)
    total_correct += output.eq(labels).sum().item()

  model_accuracy = total_correct / total_images * 100
  print('epoch {0} total_correct: {1} loss: {2:.2f} acc: {3:.2f}'.format(
          epoch,total_correct, total_loss, model_accuracy) )
        
def test(model, device, test_loader,lossF):
  model.eval()
  total = 0
  correct = 0
  with torch.no_grad():
    for images, labels in test_loader:
      images, labels = images.to(device), labels.to(device)
      output = model(images)
      
      pred_y = output.argmax(dim=1, keepdim=True)
      correct += pred_y.eq(labels.view_as(pred_y)).sum().item()
      total += labels.size(0)
    
    model_accuracy = correct / total * 100
    print('      Accuracy on {0} test images: {1:.2f}%'.format(
                                total, model_accuracy))


def main():
  loss_values = []
  acc = []

  # command-line arguments
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


  # fetch and load training data
  trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=DenseNet.transform('train'))
  train_loader = torch.utils.data.DataLoader(trainset, batch_size=DenseNet.batch_size, shuffle=False)

  # fetch and load test data
  testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=DenseNet.transform('test'))
  test_loader = torch.utils.data.DataLoader(testset, batch_size=DenseNet.batch_size, shuffle=False)

  # choose network architecture
  net = DenseNet.net.to(device)
  lossF = DenseNet.lossFunc

  if list(net.parameters()):
    # use SGD optimizer
    # training and testing loop
    for epoch in range(1, DenseNet.epochs + 1):
      lr = 0.1
      if epoch >= int(0.5*DenseNet.epochs):
        lr = 0.01
      elif epoch >= int(0.75*DenseNet.epochs):
        lr = 0.001
      optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-4)
      train(net, device, optimizer, train_loader, epoch, lossF)
    test(net, device, test_loader, lossF)

  torch.save(net.state_dict(), 'CIFAR_Model.pth')
  print("   Model saved to CIFAR_Model.pth")
  print(acc)
  print(loss_values)
  
if __name__ == '__main__':
  main()
