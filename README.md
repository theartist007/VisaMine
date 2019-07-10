# Supervised machine learning models which predict outcome of H-1B visa application
### Dependencies

All dependencies are included in requirements.txt. pip install it on an environment with Jupyter notebook support to run the code.
Say what the step will be

```
pip install -r requirements.txt
```

## Intro

### Project Overview

Engineers around the world dream of working in the United States, especially the bay area, where coolest advancements in technology happen every day. Many talented engineers from foreign countries work hard to get temporary visas to start their American dream. H-1B is a visa issued by the U.S. government to foreign nations in special occupations. Employers who are willing to sponsor a foreign national need to file a labor condition application (LCA) and get it approved in order to be considered for a H-1B lottery for each year.

### Problem Statement

Given complex procedures and paperwork that are required, most companies delegate H-1B applications to an immigration attorney. Each application needs to adhere to a proper format and satisfy conditions required for each job category in order to be successfully certified. The whole process is usually achieved by law professionals communicating with the applicant and the employer multiple times. A denied application not only incurs additional cost for each individuals’ time, but could also potentially lower the applicant’s chance to be successfully certified for the same position. Hence, it is imperative that each filed application satisfies requirements specified by USCIS and is comparable to other applications from the same industry.
For my capstone project, I created three different supervised classifier models (Random Forest, XGBoost, and Support Vector Machine), evaluated performance of each of them, and decided the final model based on performance and feasibility. I started by figuring out the set of features that are common between data source from different years, resolving naming conflict, and discarding infeasible and least useful features. After cleaning up records with empty values, I tried initial implementation of all three models and realized all models severely overfit to a dominant class due to highly imbalanced training data. I random-sampled records from the dominant class to achieve about the same balance between two classes, applied dimensionality reduction (PCA)
to the data set, and implemented three models again with default parameters. All three models turned out to be successful, and parameter tuning was applied to each of them to further optimize performance. XGBoost classifier was selected as the final model due to its high performance and efficiency. The final model was tested with manipulated data with random noise to demonstrate it is robust enough and not significantly affected by noisy and unseen data.
The set of features is texts entered in different sections of a filed LCA. A successful model enables individuals filing an application to quickly check whether the information they enter in the application is strong enough to be considered by USCIS. This model does not replace an immigration attorney by any means, but could be used as a quick sanity check after a proper application is created.

## Source Data

For this project, official LCA disclosure data from United States Department of Labor was used. Given that the list of columns in the disclosure data changed, only data from years 2014 to 2018 was used.
```
https://www.foreignlaborcert.doleta.gov/performancedata.cfm
```

Below notebook shows quick stats for the used columns using Pandas.
```
3_FINAL_RESULT/Data Exploration.ipynb
```
## Algorithms and Techniques

I used the following algorithms to predict outcome of each LCA application.

### XGBoost
eXtreme Gradient Boosting is known to push the limits of computing power for gradient boosting by utilizing multiple threads. It is believed to take significantly less time to train than other boosting models and efficient in decreasing chance of overfitting by performing L1 and L2 regularization. It creates a strong classifier based on many weak classifiers, where “weak” and “strong” refer to how closely correlated each learner is to the actual target. Since it adds models on top of each other iteratively, errors from previous “weaker” learners could be corrected by the “stronger” predictor in the next level. This process continues until the training data is accurately predicted by the model. Since we’re dealing with a data that has a lot of noise, it is definitely worth trying.
```
3_FINAL_RESULT/XGB_final.ipynb
```

### Random Forest
By constructing multiple simple trees, random forest model is known to alleviate the tendency of overfitting for skewed data. It is also known to handle a large data set with very high dimensionality well. It is trained via bagging method, which randomly sub-samples the training data, fits a decision tree model to each smaller subset (using the best split point), and aggregates the predictions at the end. I trained and tested this model using the entire one-hot encoded dataset to compare the result with the other two models, both of which needed dimensionality reduction due to model complexity.
```
3_FINAL_RESULT/RandomForest_final.ipynb
```

### Support Vector Machine
SVC with RBF kernel is known to perform very well when working on points that are not linearly separable. As shown in figure 7, it uses a kernel trick to transform the training set into a higher dimension and finds a separation boundary between classes (called a hyperplane). Hyperplanes are identified by locating support vectors (data points that would change the dividing hyperplane if removed) and their margins. Since linearity between features is not guaranteed, SVC is definitely worth trying despite very expensive computational power. Given the expensive computational cost, dimensionality of training and test set features was reduced by applying PCA with 400 features.
```
3_FINAL_RESULT/svc_final.ipynb
```


