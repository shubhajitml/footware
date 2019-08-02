import torch

import argparse


labels = ['backstrap', 'buckle', 'hook&look', 'lace_up', 'slip_on', 'zipper']
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = torch.load("model/stage2_20.pth", map_location='cpu')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', help='image for prediction')
    args = parser.parse_args()
    print('File to predict: ',args.img)
    pred = model.predict(args.img)
    pred = pred.data.numpy().argmax()
    print('Prediction: ', labels[pred])
