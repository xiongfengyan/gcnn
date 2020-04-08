# gcnn
A graph convolutional neural network for classification of building patterns using spatial vector data

This project is an open source implementation of the graph convolutional neural network for classification of building group patterns using spatial vector data.

The initial aim of our work is to introduce a novel deep learning approach to solve the traditional cognitive problems in map space. For details, one could refer to our paper 'A Graph Convolutional Neural Network for Classification of Building Patterns using Spatial Vector Data', which takes the classification of building group patterns as a case study.

For licensing reasons, we are not allowed to open the dataset used in our experiments, but the some open source dataset is available at https://www.udparty.com/index.php. Besides, OpenStreetMap(https://www.openstreetmap.org) may be a potential source of data for some geographic deep learning task, although some of the data are not well labeled and there are some quality issues to be addressed.

Please kinedly note that the implementation of neural network architecture in this project referred to the open source project contributed by Michaël Defferrard et al. (https://github.com/mdeff/cnn_graph).

However, the input in the original project is limited to a fixed graph structure, i.e., the graph structures of input samples are the same; in our project, the graph structures of input samples are different and batch training for samples with different graph structures is supported.

Please cite our paper if you use it:

Xiongfeng Yan, Tinghua Ai, Min Yang, and Hongmei Yin, 2019. A graph convolutional neural network for classification of building patterns using spatial vector data. ISPRS Journal of Photogrammetry and Remote Sensing, 150, 259-273.

Please contact me if you have further questions:
xiongfeng.yan@whu.edu.cn
