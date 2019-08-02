import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse

labels = ['backstrap', 'buckle', 'hook&look', 'lace_up', 'slip_on', 'zipper']
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
fynd_stats = [(0.7843, 0.7677, 0.7611), (0.3087, 0.3198, 0.3239)]
model = torch.load("model/stage2_20.pth", map_location=device)

def process_image(image_path):
    # Load Image
    img = Image.open(image_path)
    # Get the dimensions of the image
    width, height = img.size
    # Resize by keeping the aspect ratio, but changing the dimension
    # so the shortest size is 255px
    img = img.resize((255, int(255*(height/width))) if width < height else (int(255*(width/height)), 255))
    # Get the dimensions of the new image size
    width, height = img.size
    # Set the coordinates to do a center crop of 224 x 224
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    img = img.crop((left, top, right, bottom))
    # Turn image into numpy array
    img = np.array(img)
    # Make the color channel dimension first instead of last
    img = img.transpose((2, 0, 1))
    # Make all values between 0 and 1
    img = img/255
    # Normalize based on the preset mean and standard deviation
    img[0] = (img[0] - fynd_stats[0][0])/fynd_stats[1][0]
    img[1] = (img[1] - fynd_stats[0][1])/fynd_stats[1][1]
    img[2] = (img[2] - fynd_stats[0][2])/fynd_stats[1][2]
    # Add a fourth dimension to the beginning to indicate batch size
    img = img[np.newaxis,:]
    # Turn into a torch tensor
    image = torch.from_numpy(img)
    image = image.float()
    return image

# Using our model to predict the label
def predict(image, model):
    # Pass the image through our model
    output = model.forward(image)
    
    # Reverse the log function in our output
    output = torch.exp(output)
    
    # Get the top predicted class, and the output percentage for
    # that class
    probs, classes = output.topk(1, dim=1)
    return probs.item(), classes.item()

# Show Image
def show_image(image):
    # Convert image to numpy
    image = image.numpy()
    
    # Un-normalize the image
    image[0] = image[0] * fynd_stats[1][0] + fynd_stats[0][0]
    
    # Print the image
    fig = plt.figure(figsize=(25, 4))
    plt.imshow(np.transpose(image[0], (1, 2, 0)))    


# model.eval()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', help='image for prediction')
    args = parser.parse_args()
    print('Image to predict: ',args.img)

    # Process Image
    image = process_image(args.img)# Give image to model to predict output
    top_prob, top_class = predict(image, model)# Show the image
    show_image(image)# Print the results
    print("The model is ", top_prob*100, "% certain that the image has a predicted class of ", top_class  )
