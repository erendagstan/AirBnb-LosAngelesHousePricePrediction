<h1> AIR BNB: LOS ANGELES HOUSE PRICE PREDICTION PROJECT </h1>

<h2> Project Objective </h2>
<ul>
  <li><h3>Accurate Price Predictions: To guide homeowners in setting a competitive yet fair price for their properties.</h3></li>
  <li><h3>User Satisfaction: Assisting accommodation seekers in finding lodging options that fit their budgets and meet their expectations.</h3></li>
  <li><h3>Market Analysis: Understanding trends in Airbnb house prices and determining how specific factors impact prices.</h3></li>
  <li><h3>Optimization: Helping homeowners optimize pricing for their accommodation units based on various factors.</h3></li>
</ul>

<h2> Steps </h2>
<ul>
  <li>Data Mining</li>
  <li>Exploration Data Analysis</li>
  <li>Outlier Handling</li>
  <li>Log/Sqrt Transformation</li>
  <li>Amenities Detection</li>
  <li>Feature Engineering</li>
  <li>Modeling</li>
  <li>Hyperparameter Tuning</li>
</ul>

<h2> Data Mining </h2>
We obtained our dataset from the Open Data Soft website, specifically focusing on Airbnb homes in Los Angeles. This process involved data mining, where we systematically extracted relevant information from the available datasets to use in our analysis and project. The collected data from Los Angeles Airbnb listings serves as the foundation for our project, allowing us to perform in-depth analyses and derive meaningful insights for various purposes such as accurate price predictions, user satisfaction, market analysis, and optimization.

<h2> EDA </h2>
To understand the exploration of the dataset and methods for feature derivation, various charts were examined. The following graph illustrates some observations and frequencies related to the 'amenities' feature in our dataset.
<img width="100%" alt="amenities_eda" src="https://github.com/erendagstan/AirBnb-LosAngelesHousePricePrediction/assets/86521359/975eef9e-b369-4629-960a-0cc1c19c8c48">

<h2> Feature Engineering </h2>
<div style="display: flex;">

  <!-- Dataset Features -->
  <div style="flex: 1; margin-right: 5%;">
    <p>Dataset Features;</p>
    <img width="45%" alt="dataset_features" src="https://github.com/erendagstan/AirBnb-LosAngelesHousePricePrediction/assets/86521359/fbffe6f1-ab78-4d5e-a3b1-c09ae8935734">
  </div>

  <!-- API Features -->
  <div style="flex: 1;">
    <p>API Features;</p>
    <img width="25%" alt="api_features" src="https://github.com/erendagstan/AirBnb-LosAngelesHousePricePrediction/assets/86521359/eda8c5d2-a88c-4333-8e2f-376b21164cae">
  </div>

</div>

<h2> Modelling </h2>
<p>We applied five different regression models: Linear Regression, Ridge Regression, Random Forest Regressor, Gradient Boosting Regressor, XGBoost Regressor, and LightGBM Regressor. After evaluating the root mean squared error (RMSE) and R² scores for each model, we decided to choose XGBoost Regressor</p>
<img width="25%" alt="modelling_image" src="https://github.com/erendagstan/AirBnb-LosAngelesHousePricePrediction/assets/86521359/9db99493-5a10-4c17-92b8-134d75e22efc">

<h2> Optimization </h2>
<p>Grid and random search methods were employed using 5-fold cross-validation to enhance the success rate of the methods in terms of R². As a result, the XGBoost Regressor demonstrated the best performance with an R² score of 0.813.</p>
<img width="50%" alt="optimization_datalanta" src="https://github.com/erendagstan/AirBnb-LosAngelesHousePricePrediction/assets/86521359/a9e3ea2b-2d91-44df-b801-f9b1c017180b">

<h2> Streamlit </h2>
<p> How to use streamlit? Open terminal and "streamlit run .../Homepage.py" </p>
<h3> Homepage</h3>
<p>This study presents a house price prediction model developed to estimate the prices of Airbnb homes in Los Angeles. In the creation of the model, an approach was adopted that takes into account various factors around Airbnb homes. Interactions around significant locations such as fire stations, police stations, landmarks, schools, metro stations, areas of arrests, coffee shops, hospitals, and other important places were considered as critical factors in determining home prices. This comprehensive model aims to contribute to a better understanding of price fluctuations in the Airbnb housing market in Los Angeles by considering the complexity and diversity of multiple variables in predicting home prices. The dataset used in the analysis includes various factors influencing home prices, providing users with a valuable resource to make more informed decisions.</p>
<h3> Los Angeles Maps </h3>
<p>Through the 'Los Angeles Maps' tab, the visual distribution and locations of significant landmarks evaluated in our study, including fire stations, police stations, monuments, schools, metro stations, areas of arrests, cafes, and hospitals, are presented on the map. This map represents the dataset used in the analysis of various factors around Airbnb homes. Emphasizing the environmental factors crucial to the development of our model for predicting home prices, this map illustrates the specific interactions or features represented by each location. By examining the distributions on the map, you can gain valuable insights into how specific locations may impact home prices, allowing you to make more informed decisions in this regard.</p>
<img width="100%" alt="la-maps3" src="https://github.com/erendagstan/AirBnb-LosAngelesHousePricePrediction/assets/86521359/eb707c40-4792-4839-803a-1dcdc3acfea2">
<img width="100%" alt="la-maps2" src="https://github.com/erendagstan/AirBnb-LosAngelesHousePricePrediction/assets/86521359/6e882853-8d85-4607-b2be-a0bc998ee92c">
<img width="100%" alt="la-maps3" src="https://github.com/erendagstan/AirBnb-LosAngelesHousePricePrediction/assets/86521359/18fffd8b-e933-488a-8c67-79779695f6a8">
<h3> Meet Los Angeles Landmarks </h3>
<p>Explore Los Angeles Landmarks: The 'Closest Airbnb Houses' tab allows users to discover the nearest Airbnb homes around the landmarks they want to visit in Los Angeles. Through this tab, users can view homes around a specific landmark, examine their prices, and access the Airbnb listing page. The ability to see the locations of homes on the map provides users with the opportunity to compare and choose accommodation options by evaluating environmental factors. This allows users to visually explore and assess the closest accommodation options to the landmark they plan to visit and make informed decisions.</p>
<img width="100%" alt="meet_la_landmarks" src="https://github.com/erendagstan/AirBnb-LosAngelesHousePricePrediction/assets/86521359/f32ca6a3-493f-4ea0-8b7b-5bacf4189189">
<h3>House Price Prediction</h3>
<p>The 'House Price Prediction' tab provides users with the opportunity to predict Airbnb house prices by taking various inputs from them. Users make various entries to determine the features of the house they want to predict. These inputs include factors such as property type, neighborhood, distance to Echo Park, number of bedrooms, room type, number of bathrooms, amenities, accommodation capacity, number of beds, security deposit, and the number of included guests. These details are processed by our model, and a predicted price is generated based on the specified house features. The 'House Price Prediction' tab allows users to make price predictions based on specific house features and evaluate accommodation options.</p>
<img width="70%" alt="housepricepred-datalanta" src="https://github.com/erendagstan/AirBnb-LosAngelesHousePricePrediction/assets/86521359/bb348174-b543-42e7-a99d-5ea3af5ae6dd">

