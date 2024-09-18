To execute the code, please follow these steps:

Several large pre-trained model files and code are available for download from Google Drive.
https://drive.google.com/drive/folders/1udZ0SSgi0whG6BqdSyYh0M5opWu8Ta9j?usp=sharing 

The first step>
To train the model effectively, follow these steps:
1.	Set up the environment: Ensure you have the necessary libraries and dependencies installed for running the train_LayerMSA_gender2_tripletloss.py code.
2.	Choose a suitable model:  Select a high-order model from model_LayerMSA_gender_tripletloss.py for optimal performance.
3.	Adjust model parameters: Modify the parameters of the last fully connected layer in the chosen model, setting its dimension to 1024.
4.	Train the model: Utilize the training dataset to train the model for at least 200 epochs. 


The second step>
After training, utilize the trained model to infer on both the training and test datasets. Extract a 1x1024 feature vector for each image and save these vectors into CSV files: rsna_train_1024.csv and rsna_test_1024.csv.

Use the latent_vector.py script for this step, ensuring you import both the trained model and its corresponding model definition file.

The Third step>

Next, perform K-means clustering on the feature vectors stored in the two generated CSV files. Execute the kmeans.py script twice to obtain grouping results for both the training and test sets.

The Fourth step>
After obtaining the grouped CSV files, apply inter-group label smoothing. Manually copy 5-10% of labels from adjacent groups into each group's label file to generate the final training labels for the fine stage. (Results from this step are provided in the "adjacent" folder).

The Fifth step>
Rerun the code from the first step, utilizing the refined training labels, and save the fine-stage results.

The Sixth step>
Finally, calculate the overall mean absolute error (MAE) by weighting the MAEs of the fine-tuned models according to the size of their respective groups within the dataset.


For the detailed contents of each subfolder, please refer to the corresponding readme

------------------------------------------------------------
------------------------------------------------------------

If you use or refer to our code, please cite our paper. Thank you.

Title: Coarse-to-Fine Bone Age Regression by using Multi-Scale Self-Attention Mechanism
Author: Guanyu Wu, Ziming Wanga, Jian Peng, Shaobing Gao
College of Computer Science, Sichuan University, Chengdu 610065, China

Status:  Preprint submitted to Biomedical Signal Processing and Control (under review)

This repository includes preliminary code and a detailed guide for running the experiments. Please download the required public dataset as described in the associated paper. 

Any questions please contact to Mr. Guanyu Wu  email: 812997877@qq.com  
------------------------------------------------------------
------------------------------------------------------------
