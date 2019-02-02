# Master-Slave-DNNs
Master-Slave Deep Neural Network (MS-DNN) Architecture in MATLAB.

# Introduction 
The project presents a novel Master-Slave Deep Neural Network (MS-DNN) architecture in MATLAB initially used for the purpose of Gesture Recognition from surface Electromyography (sEMG) wearable sensors. The performance of the master-slave network is augmented by leveraging additional synthetic feature data generated by long short term memory networks. Performance of the proposed network is compared to that of a conventional DNN prior to and after the addition of synthetic data. Up to 14% improvement is observed in the conventional DNN and up to 9% improvement in master-slave network on addition of synthetic data with an average accuracy value of 93.5% asserting the suitability of the proposed approach. 

# Usage
Clone or Download the repository and save the files in your MATLAB directory. 'dnntrain.m' is the functioning consisting of a simple DNN
which is executed twice (once as the master and then as the slave) in the main file 'msdnn_imp.m'. Input your DNN paramters and the splitting factor of data in the main file and run it to compare the results of MS-DNN with the conventional standalone DNN.

<img src="Results\flow.png" height="300" width="600"/>

# Results
<img src="Results\master_cost.png" height="200" width="250"/><img src="Results\slave_cost.png" height="200" width="250"/><img src="Results\standalone_cost.png" height="200" width="250"/>

<img src="Results\master_iter.png" height="200" width="250"/><img src="Results\slave_iter.png" height="200" width="250"/><img src="Results\standalone_iter.png" height="200" width="250"/>

# Acknowledgment
We would like to recognize the funding support provided by the Science & Engineering Research Board, a statutory body of the Department of Science & Technology (DST), Government of India, SERB file number ECR/2016/000637. 
