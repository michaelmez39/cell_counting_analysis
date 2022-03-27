# Automatic Cell Counting Models
Machine learning models to count cells in images.

## Files
`load_data.py` creates a Tensorflow dataset from a set of images, the class CellDataset holds the dataset in the data field. May need be altered for different datasets.

`count.py` holds the tensorflow model and the code to train the model

`viewer.py` this is used to visualize the output of the model in comparison to its input

**To change datasets, count.py and viewer.py will have to be updated with the new path to the dataset and some information like image size and width will need to be changed**

## Sources
### Convolutional Neural Networks
**Countception**
Convolutional neural network that counts cells using a smaller CNN window to the image.\
[repo](https://github.com/ieee8023/countception)\
[paper](https://arxiv.org/abs/1703.08710)

**U-Net**
Also a CNN, considers whole image as input, uses pooling in network. Can also be used for image segmentation, which we might consider if we have time. Higher training time?
[easy explanation](https://towardsdatascience.com/understanding-semantic-segmentation-with-unet-6be4f42d4b47)\
[paper segmentation](https://arxiv.org/abs/1505.04597)\
[paper counting](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8153189/#R24)\

## Datasets
**[Modified Bone Marrow (MBM) Dataset](https://github.com/ieee8023/countception/blob/master/MBM_data.zip)**

**[VGG Cells Dataset](https://github.com/ieee8023/countception/blob/master/cells.zip)**
