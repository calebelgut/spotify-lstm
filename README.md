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

This second half of the project dealt with Time Series Analysis. A review on Time Series Analysis would show us that Time Series Data refers to any dataset where the progress of time is an important dimension in the dataset. **Our job in this endeavor is to use the previous 100 years of data to predict the prevalence of the four most popular and most niche features that we identified during classification.** We will predict the prevalence of this data over the next five years. 

## Step One: Helper Functions

I began this section of the project by creating most of the functions that I would be using during my workflow. I would eventually create a time series model for each individual feature (8 in total) and wanted to have a streamlined method to approach this task. 

I created helper functions for: 
1. DataFrame Normalization (would only be used once)
1. Splitting the Data into Train & Test
1. Fitting the ARIMA Model
1. Finding the Train & Test RMSE for Model Validation
1. Forecasting Future Information

![helper_function_1](/readme_images/helper_function_1.jpg)
![helper_function_2](/readme_images/helper_function_2.jpg)
![helper_function_3](/readme_images/helper_function_3.jpg)
![helper_function_4](/readme_images/helper_function_4.jpg)

## Step Two: Read in the Data & Transform Into Time Series 

After reading in the data, my first task was to create two new dataframes: one that would have the yearly average of the four most popular features and one that would have the yearly average of the four most niche features.  

![new_df](/readme_images/new_df.jpg)

When I plotted the initial time series of the new dataframes, it became clear very quickly that normalization would be necessary. The loudness column had measurements at a very different scale than energy, valence, and acousticness. This disparity made the initial time series very difficult to read. Take a look at the difference between the time series with loudness and without it: 

![with_loudness](/readme_images/with_loudness.jpg)
![without_loudness](/readme_images/without_loudness.jpg)

## Step Three: Bask in the Glory of Normalized Data

With our dataframes normalized, let's take a look at them and see if we can gain preliminary understanding of their movement before we break each individual feature down into a time series: 

![normalized](/readme_images/normalized.png)

Here, it seems that loudness and energy are on the rise and have been since the 50s but have experienced a substantial rise since the 80s. This seems to correlate with the fall of acousticness which was very prevalent between 1920 and 1950. Ever since 1950, however, the prevalence of acousticness has shrunk substantially. Songs with a cheerful disposition (high **valence**) seem to have peaked in the 80s which would make sense given that decade as an era of disco & synth followed by the grunge & pop punk of the 90s. It may, however, be on the rise again.

![normalized_niche](/readme_images/normalized_niche.png)

As far as the niche features are concerned we can see, here, that songs with a *major scale* (**mode**) hit a peak between 1960 and 1970. Since that era, a majority of songs have steadily approached the minor scale with a peak in that regard in 2020. **Speechiness** had its heyday in the 1930s and hasn't reached the same peak since. It had its fall from grace post-1950s and, since then, we haven't seen too much of it until recently. **Tempo** had a major rise between the 60s and the 80s and then hit a bit of a dip between the 80s and 00s, and is now on the rise again! **Liveness** hasn't been seen too much since the late 70s & early 80s. 

## Step Four: Prepare Data for Time Series Analysis

With everything visualized and given an initial analysis, I needed to get to work on separating these features and creating a time series for each. The first step of this was to change the column 'year' into DateTime Format--a format that can be read by the time series models--and I then needed to set the index as the datetime. Setting the index as the datetime removes the datetime from analysis and, rather, allows the target feature to be analyzed by the model with the indexed DateTime keeping our data in order! 

![data_prep_ts](/readme_images/data_prep_ts.jpg)

## Step Five: Separate each column into its own time series

With our data all prepped and ready to go the final step before modelling is to separate each feature into its own time series. The first step for this is to create a list of dataframes (one list for popular features and one list for niche features) and, from there, instantiate individual time series for each of the features.

![pop_df](/readme_images/pop_df.jpg)

![instantiate](/readme_images/instantiate.jpg)

## Step Six: Establish Workflow and Get to Work

Since I repeated the same process for each of our 8 features, I will only be touching on the models that performed the best and give a conclusive analysis at the end before moving on to LSTMs. It is important to note, however, that there was a steady workflow for each model's creation. It is as follows: 

### Workflow for Our Models:

1. Visualize the Time Series
1. Use auto_arima to find the best order & seasonal order (if it exists) for the ARIMA (or SARIMA) model
1. Fit the model
1. Examine results & residual analysis
1. Use the train & test RMSE for validation
1. Forecast future values
1. Repeat for each additional time series 

### Note About the AIC:

In the models I built I received one result, in particular, that was different from my previous experiences with time series models: **The AICs I received when searching for the best order & seasonal order for my models were often negative.** Initially, this worried me as I hadn't encountered negative AICs before however, according to William H. Greene, the Robert Stansky Professor of Economics and Statistics at Stern School of Business at NYU, 

*"The sign of the AIC tells you absolutely nothing about ill conditioned parameters or whether the model is suitable or not. For example, in a linear regression case, if the AIC is positive, you can make it negative (or vice versa) just by multiplying every observation on the dependent variable by the same number. That obviously does not change the characteristics of the model."*

This, to me, says that a negative AIC is fine, as we are still looking for the lowest possible value. Unlike positive AICs where we are looking for the number closest to zero, with negative AICs we simply flip that strategy and search for the AIC furthest from zero. 

## Example Workflow Visualized: 

If judging by lowest RMSE & similar RMSE between train & test groups, the best ARIMA model was the one made from the loudness time series. The following are the steps taken to build & forecast with this model:

### Visualize the Time Series:

![loudness_viz](/readme_images/loudness_viz.png)

### Use auto_arima to find the best order & seasonal order (if it exists) for the ARIMA (or SARIMA) model

![loudness_auto_arima](/readme_images/loudness_auto_arima.jpg)

### Fit the Model, Examine Results, Examine Residuals

![loudness_results](/readme_images/loudness_results.jpg)

![loudness_errors](/readme_images/loudness_errors.png)

### Validate with Train & Test RMSE

![loudness_train_rmse](/readme_images/loudness_train_rmse.png)

![loudness_test_rmse](/readme_images/loudness_test_rmse.png)

### Forecast 

![loudness_forecast](/readme_images/loudness_forecast.jpg)

## Other Measures to Determine Best Model?

If one was to visually examine the residuals and use this to determine a best model, that model would be the one built on the mode time series. Here is the residual analysis for that model--unfortunately the RMSE was considerably higher: train & test were both closer to 0.1 as opposed to the 0.048 from both the train & test RMSE of the loudness model. 

![mode_errors](/readme_images/mode_errors.png)

## Step Seven: When Finished Modelling, Compile Results and Give Conclusive Analysis: 

While the big conclusion will come after we complete the LSTM analysis, the conclusions thus far are as follows: 

### Conclusion for Popular Features: The prevalence of Loudness & Energy will grow the most of the 4 features that most likely predict popularity.

If you are an executive looking to emphasize certain features in the artists sponsored by your label, you may want to pursue these features if following trends is your thing however if you are looking to break the mold, investing in acoustic artists may be ideal as the prevalence of acousticness is quite low right now.




