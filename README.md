# Classification & Time Series Analysis of Spotify Data
**Caleb Elgut - September 2020**

# Introduction

This project combines Classification & Time Series analysis to gain a deep understanding of those features of a song which would be most likely to either **predict popularity** or **predict nicheness.** After conducting a classification analysis to solve this problem, I ran a time series analysis to predict the prevalence of these features over the next five years. These predictions will help guide two groups of people: music executives and independent artists. The results will guide music executives who are looking to invest in certain elements of music while shaping their artists into further popularity. The results will also help independent artists understand which features are used the least and don't traditionally predict popularity--perhaps these artists can use the rarity of these features to their advantage in an attempt to stand out.

## Summary of DataFrame

The dataframe consists of nearly 170,000 songs released between 1921 and 2020 from Spotify's API. Each song is broken down into a myriad of factors all measured by algorithms provided by Spotify. These algorithms measure everything from reasonably-quantifiable factors such as duration, key, and loudness (measured by decibel) to factors one wouldn't initially think were quantifiable such as the level of energy in a song or its cheerfulness.

## Dictionary of DataFrame

### Dictionary:

1. **acousticness**: A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.
1. **artists**: The name(s) of the artist(s) who perform the track.
1. **danceability**: Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.
1. **duration_ms**: A track's duration in milliseconds.
1. **energy**: Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy. 
1. **explicit**: Whether the track was labeled as "explicit" due to content. 1 = Yes, 0 = No.
1. **id**: Track's unique identifier.
1. **instrumentalness**: Predicts whether a track contains no vocals. “Ooh” and “aah” sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly “vocal”. The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0. 
1. **key**: The estimated overall key of the track. Integers map to pitches using standard Pitch Class notation . E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on. If no key was detected, the value is -1. 
1. **liveness**: Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live. 
1. **loudness**: The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typical range between -60 and 0 db. **NOTE:** *Loudness is measured negatively here on purpose. This is because we are dealing with digital sound that a computer can listen to which is measured differently than what the human ear can listen to. This is sometimes why you will see negative numbers when you adjust the volume on your surround sound system. For more information on this, you can read this article: https://www.cablechick.com.au/blog/why-does-my-amplifier-use-negative-db-for-volume/*  
1. **mode**: Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0.
1. **name**: Name of Track 
1. **popularity**: The popularity of the track. The value will be between 0 and 100, with 100 being the most popular.The popularity of a track is a value between 0 and 100, with 100 being the most popular. The popularity is calculated by algorithm and is based, in the most part, on the total number of plays the track has had and how recent those plays are.
1. **release_date**: Date of track's release.
1. **speechiness**: Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks.
1. **tempo**: The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration. 
1. **valence**: A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry). 
1. **year**: Year of track's release.

# Classification Analysis
As mentioned above, the first half of this project was an effort to discover which features were most apt to predict popularity for a given track. 

## Step One: Preprocessing

The first move is to make sure our data is ready to be processed and read by our classification models. The first step of our preprocessing was to take our column 'popularity', which was originally labeled between 0 and 100, and turn it into a binary column with a class of 0 for not popular and 1 for popular. 

To accomplish this goal, we first binned popularity, splitting it between 0 and 49 & 51 and 100. Then, a new column 'popular' was created. 

If a track's popularity was between 0 and 49, the track's value for 'popular' would be denoted as 0 (**not popular**). If it was between 50 and 100 it would be denoted as a 1 (**popular**)

### Examining Class Imbalance

22.6% of the values were classified as popular and 77.35% are classified unpopular. *This is a class imbalance issue so we will need to make sure our best model has a good AUC so that we know that it can tell the difference between the two classes. We will discuss this more later.*

![class_imbalance](/readme_images/class_imbalance.jpg)

### End of Preprocessing: Clean Up Columns & Examine Correlation

After examining the class imbalance, I deleted any columns with non-numeric data as they would not be necessary to the classification analysis (their values can't be read by the models anyways). Additionally, I created a correlation table to look into how each column related to each other. 

![correlation](/readme_images/correlation.png)

### Observations from Correlation:

- Loudness and Acousticness somewhat correlate.
- Loudness and Instrumentalness somewhat correlate.
- Energy & Acousticness have a high correlation.

## Step Two: Split Data

After data was completely preprocessed, I chose the column 'popular' as the target (y) variable and the rest of the columns as the X variable. From here, it was a simple train_test_split with 80% of the data allocated for training and the remaining 20% allocated for testing. **As a reminder, it is important to make this separation so that your model will be able to verify its efficacy, otherwise--if the model is simply trained on 100% of the data--there will be no way to know that the model is making accurate predictions.

## Step Three: Model! 

For this portion of the project I put the entire data through seven models: 
1. Decision Tree Untuned
1. Decision Tree Manually Tuned
1. K-Nearest-Neighors Untuned
1. K-Nearest-Neighbors with Best K
1. Random Forest Untuned
1. Decision Tree Grid Searched
1. Random Forest Grid Searched

After these models, I trained a few more on the top 7 features most likely to predict popularity:

1. Untuned Decision Tree
1. Untuned Random Forest
1. Grid-Searched Decision Tree
1. Grid-Searched Random Forest

### What are these models? 
As a quick review I will summarize the most important of these models:

1. **Decision Tree**: A classifier which uses information gain and entropy (the opposite of information gain) to determine which features are most important in determining a target feature. A decision tree is called as such because it begins with a root node and splits from there. As each split occurs in the decision tree, entropy is measured. The split which has the lowest entropy compared to the parent node and other splits is chosen. The lesser the entropy, the better.

1. **Random Forest**: Essentially creates a series of decision trees based on the dataset with each tree being different. Each decision tree makes choices that maximize information. With a diverse series of trees, I have an opportunity to have a model that gives me even more information than the single tree I created. Random Forests are also very resilient to overfitting--our random forest of diverse decision trees are trained on different sets of data and looks at different subsets of features to make predictions--> For any given tree there is room for error but odds that every tree will make the same mistake because they looked at the same predictor is infinitesimally small!

1. **Grid-Searched**: When a classification model is grid-searched it is tuned in the best way possible. As a review, to tune a model is to adjust each of its hyperparameters individually. When a model uses Grid-Search it uses a function that tries every possible parameter combination that you feed it to find out which combination of parameters will give you the best possible score.

***Note: I do not discuss K-Nearest Neighbors here for the sake of time as it produced the least helpful results of the models here. If you are interested in learning more about K-Nearest Neighbors models, however, here is a great link: https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761***

### Workflow for Classification Modeling:

I will be showing the results from a few of my best models in a moment but before I do that I would like to share the workflow of classification modeling so that I don't need to show each and every model I worked on: 

**When not Grid-Searching**: 

Step One: Create the Classifier--as an option before this step you can choose to manually tune your model--and attach the appropriate hyperparameters.

Step Two: Fit the model to the training data

Step Three: Create a variable called "y_pred" which uses your new fit model to predict on the X_test data

Step Four: Create and print a confusion matrix and a classification report. 

- The confusion matrix shows you how many correct positive & negatives as well as how many false positives and negatives your model acquired. 

Step Five: Calculate the AUC

- The AUC is the measure of the ability of a classifier to distinguish between classes and is used as a summary of the ROC curve
  - The ROC curve shows the trade-off between sensitivity (or True Positive Rate) and specificity (1 – False Positive Rate). Classifiers that give curves closer to the top-left corner indicate a better performance.
  - Because the ROC does not depend on the distribution of classes, it is very helpful with a class imbalance problem. Because there are far more unpopular songs than popular songs in our dataframe, the ROC & AUC will be incredibly helpful in aiding me in choosing the best models for classification purposes.

Step Six: Report Important Values & Repeat
- The most important values for us when evaluating our models are: 
  - F1: The balance between Recall & Precision
  - Recall: The number of relevant items retrieved by a search divided by the total number of existing relevant items.
  - Precision: The number of relevant items retrieved by a search divided by the total number of items retrieved by that search.
  - AUC: Already explained. 
- A higher recall means we are going to assume that more tracks will turn up positive for popularity. This gives us a higher rate of false positives. This may lead our business associates to invest in the wrong features and, ultimately, the wrong artists which could cause a long-term loss for them.
- On the other hand, a higher precision means our model's threshold for what it will consider a prediction of popularity will be higher. This gives us a higher rate of false negatives as some songs with feature prevalence that would accurately predict popularity may not be noted as such by the model. This may result in an increased amount of frustrated & annoyed artists however it does not put the record label in danger of losing money. A high precision is what we will aim for--this will be particularly effective for those executives who are looking to take less risk. 

## Visualizing the Best Models:

### Grid-Searched Random Forest on All Columns: 

![confusion_matrix_gs_all](/readme_images/grid_search_ran_forest.jpg)

![gs_all_AUC](/readme_images/auc_gs_ran_for.png)

![gs_all_top_feats](/readme_images/important_feats_gs_ran_for.png)

**Analysis:**

- Scores the highest in nearly every important category:
    - Highest F1 means that we have the best balance between Recall & Precision
    - Second Highest Precision of 0.68 & Second Highest Recall of 0.45 means that our model will not have a very high rate of either false negatives or false positives. 
        - In the end, Precision is the more important factor especially when pitching to executives as we will want to ensure that our model is giving us fewer false positives however it is good to see a model with a strong balance of both.
    - Highest AUC of 0.848 means that our model is best at telling the difference between classes. Since there is a fairly strong class imbalance, here, this may be the most important component when deciding which model is the best for our predictions. 
    - Accuracy is the fraction of predictions our model got right! It isn't the best predictor for a classification model however it is worth pointing out that this model had the highest accuracy.

After the above feature ranking, I ran another series of classification analyses on the top 7 features as determined by the above model. Of these models, the grid-searched random forest was, once again, the winner! It is visualized below:

![grid_search_ran_forest_top](/readme_images/grid_search_ran_forest_top.jpg)

![grid_search_ran_forest_top_cm](/readme_images/grid_search_ran_forest_top_cm.jpg)

![auc_gs_ran_for_top](/readme_images/auc_gs_ran_for_top.jpg)

![important_feats_gs_ran_for_top](/readme_images/important_feats_gs_ran_for_top.png)

For fun, let's check out a visualization of the 5 trees from this Random Forest

![rf_5trees](/readme_images/rf_5trees.png)

Similarly to our situation with analyzing all features, the grid-searched random forest comes out on top.

- **Analysis:**
    - Nearly highest in every category except recall (The Untuned Decision Tree has that honor) however this model is still the best for showing us which features will predict popularity because the AUC in this model is much higher and can therefore be trusted when it comes to differentiating between classes (Not Popular or Popular)

# What Comes Next?

From our models, we must make a determination of which features to send out for time series analysis. I chose the top four features most likely to predict popularity as well as the fourth least likely to do so--these we will refer to as "niche" features. 

- **Our Four Most Popular Features:** 
    - Loudness
    - Acousticness
    - Energy
    - Valence
- **Our Four Most Niche Features:**
    - Mode
    - Key
    - Tempo
    - Liveness
- The Time Series Analysis of the Top 4 Features for Popularity Prediction are being sent to Record Executives for analysis
    - The Time Series Analysis will be looking at prevalence of these features over the next 5 years.
- The TSA of the Four Most Niche Features are being sent to independent artists who may be looking to break away from the mold and take a chance on investing in a certain element of music that will set them apart from peers who are looking to achieve maximum popularity.
  - **A note on the Niche Features**: I replaced key with speechiness because key was too difficult to model. An increase or decrease in the value for key simply denotes a separate key. It is not, at all, related to prevalence. This was cause for its deletion as I was looking to predict the prevalence of our features over the next few years. 

# Time Series Analysis with ARIMA Models:



