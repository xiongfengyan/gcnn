# gcnn
A graph convolutional neural network for classification of building patterns using spatial vector data

This project is an open source implementation of the graph convolutional neural network for classification of building group patterns using spatial vector data.

The initial aim of our work is to introduce a novel deep learning approach to solve the traditional cognitive problems in map space. For details, one could refer to our paper 'A Graph Convolutional Neural Network for Classification of Building Patterns using Spatial Vector Data', which takes the classification of building group patterns as a case study.

Please kinedly note that the implementation of neural network architecture in this project referred to the open source project contributed by Michaël Defferrard et al. (https://github.com/mdeff/cnn_graph). However, the input in the original project is limited to a fixed graph structure, i.e., the graph structures of input samples are the same; in our project, the graph structures of input samples are different and batch training for samples with different graph structures is supported.






-------------------------

This update provides a small sample dataset. The dataset contains 20 samples, including 7 irregular groups and 13 regular groups. They include a total of 1424 buildings and are stored in the shapefile format named “test20r.shp”. This shapefile contains 4 fields:

-- “type” represents the group type. Specifically, a value of 1 indicates that the group where the building is in an irregular pattern, and 3 is a regular pattern.

-- “inFID” represents the group ID, buildings in the same group have the same inFID value.

-- “A” represents the area of the building;

-- “Density” represents the density of the building, and its computation method please refers to the Section 3.2 of our paper.





Three steps:

1. Use the “Feature to JSON” tool in ArcMap (you can also use other tools, but I haven't tested them) to convert the test20r.shp file to a GeoJson format file named “test20r.json”.

2. Run dataProcessing.py to interpret the “test20r.json” file (i.e. calculate the descriptive characteristics for each building) and store it as a Json file named “test20ri.json”. Please change the corresponding file name on lines 364 and 367.

3. Run GCNN.py to process the test20ri.json, and also please pay attention to modify the name on line 14.




Please note:
1. Due to the small number of samples (less than a batch size), it is not advisable to train on this dataset, and an error will occur in line 93. But those codes for data preparation are passable, which may help you understand the specifications of the data.

2. After deciphering the shapefile, the test, validation, and training datasets are obtained. They are all composed of three parts: vertices, adjacecies, and labels. They are stored in numpy.array and the specifications are (printed on lines 22-37):

 -- vertices: N x S x F,
 
 -- adjacecies: N x S x S,
 
 -- labels: N x 1,
 
where, N is the number of samples, S is the number of buildings contained in each sample (that is, the number of graph nodes), and F is the feature dimension of the node.

You can also organize your data in the above specifications and feed it to the GCNN model for classification. 

Good luck!





Please cite our paper if you use it:

Xiongfeng Yan, Tinghua Ai, Min Yang, and Hongmei Yin, 2019. A graph convolutional neural network for classification of building patterns using spatial vector data. ISPRS Journal of Photogrammetry and Remote Sensing, 150, 259-273.





Please contact me if you have further questions:
xiongfeng.yan@whu.edu.cn
