import torch
from torchvision import models, transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse

labels = ['backstrap', 'buckle', 'hook&look', 'lace_up', 'slip_on', 'zipper']
idx_to_class = {0:'backstrap', 1:'buckle', 2:'hook&look', 3:'lace_up', 4:'slip_on', 5:'zipper'}
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
fynd_stats = [(0.7843, 0.7677, 0.7611), (0.3087, 0.3198, 0.3239)]

model = models.resnet50()

# model.load_state_dict(torch.load("model/stage2_20.pth", map_location=device), strict=False)
model.load_state_dict(torch.load("../../models/resnet50-stage-2_11.pth", map_location=device), strict=False)
model.eval()


def get_prediction(model, test_image_path):
     
    transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(fynd_stats[0], fynd_stats[1])])
 
    test_image = Image.open(test_image_path)
    plt.imshow(test_image)
     
    test_image_tensor = transform(test_image)
 
    if torch.cuda.is_available():
        test_image_tensor = test_image_tensor.view(1, 3, 224, 224).cuda()
    else:
        test_image_tensor = test_image_tensor.view(1, 3, 224, 224)
     
    with torch.no_grad():
        model.eval()
        # Model outputs log probabilities
        out = model(test_image_tensor)
        ps = torch.exp(out)
        topk, topclass = ps.topk(1, dim=1)
        print(" topk and topclass",topk, topclass)
        print("Output class :  ", idx_to_class[topclass.cpu().numpy()[0][0]])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', help='image for prediction')
    args = parser.parse_args()
    print('Image to predict: ',args.img)
    img_path = args.img
    get_prediction(model, img_path)