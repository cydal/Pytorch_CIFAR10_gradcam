# Pytorch_CIFAR10_gradcam
GRADCAM for CIFAR10-Pytorch



## Pytorch GradCAM for CIFAR-10


#### Folder structure
* [models](https://github.com/cydal/Pytorch_CIFAR10_gradcam/tree/main/models) - Contains models including [resnet.py](https://github.com/cydal/Pytorch_CIFAR10_gradcam/tree/main/models/resnet.py)
* [utils.py](https://github.com/cydal/Pytorch_CIFAR10_gradcam/blob/main/utils.py) - Helper functions
* [main.py](https://github.com/cydal/Pytorch_CIFAR10_gradcam/blob/main/main.py) - Main script file to run

For end to end training and visualization of gradcam
> python main.py

If running from google colab- 
> % run main.py

### ResNet18 trained from scratch


![enter image description here](https://i.postimg.cc/KvYJT27X/image.png)


![enter image description here](https://i.postimg.cc/y87KzW0q/image.png)

# GRADCAM

## For Model Interpretability and Visualization

![enter image description here](https://i.postimg.cc/1zDHS5SF/image.png)

````python
def  gradcam_heatmap(model, results, test_images, device):

"""

Args:

model (torch.nn): Torch model

test_targets (tensor): Ground truth labels

test_images (tensor): images array

device (str): Device type

  

Returns:

heatmaps (tensor): heatmaps array

"""

results[torch.arange(len(results)),

results.argmax(dim=1)].backward(torch.ones_like(results.argmax(dim=1)))

  

gradients = model.get_activations_gradient()

  

pooled_gradients = torch.mean(gradients, dim=[2, 3])

  
  

activations = model.get_activations(test_images.to(device)).detach()

  

# weight the channels by corresponding gradients

for j in  range(activations.shape[0]):

for i in  range(512):

activations[j, i, :, :] *= pooled_gradients[j, i]

  

# average the channels of the activations

heatmaps = torch.mean(activations, dim=1).squeeze()

  
  

# relu on top of the heatmap

heatmaps = np.maximum(heatmaps.cpu(), 0)

  

# normalize the heatmap

heatmaps /= torch.max(heatmaps)

  

return(heatmaps)
````


#### Misclassified Images - T(True class) - P(Predicted)
![enter image description here](https://i.postimg.cc/s21tVWXL/image.png)


#### Correctly Classified

![enter image description here](https://i.postimg.cc/TwMQZ5Wg/image.png)

## References

- https://github.com/jacobgil/pytorch-grad-cam
-   [https://keras.io/examples/vision/grad_cam/](https://keras.io/examples/vision/grad_cam/)
-   [https://reposhub.com/python/deep-learning/gkeechin-vizgradcam.html](https://reposhub.com/python/deep-learning/gkeechin-vizgradcam.html)
