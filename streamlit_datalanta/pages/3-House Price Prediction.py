import math
import pandas as pd
import streamlit as st
import time
import numpy as np
import joblib
import pickle

# Page settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', None)

st.set_page_config(page_title="Model", page_icon="📈", layout="wide")

st.header(":red[Air Bnb: House Price Prediction]")


# loading data
@st.cache_data
def load_model_and_data():
    df_model = pd.read_csv("DatalantaProject/datasets/model_datalanta_dataset.csv")
    df_model = df_model.drop('Unnamed: 0', axis=1)
    model = joblib.load("DatalantaProject/model/xgb_model.joblib")
    return df_model, model


# assigning data
df_model, joblib_model = load_model_and_data()

# Property Type ( except some features(villa,etc.) )
option_property_type = st.selectbox(
    'Which Property type?',
    ('Apartment', 'House', 'Loft', 'Condominium', 'Guesthouse', 'Townhouse', 'Bungalow', 'Other',
     'Bed & Breakfast'))

property_type_dict = {
    'Apartment': 4.64342263833902091363,
    'House': 4.70790097881329394625,
    'Loft': 4.68712885929782441252,
    'Condominium': 4.67041613681394185420,
    'Guesthouse': 4.66991616598477854438,
    'Townhouse': 4.67203991520136874271,
    'Bungalow': 4.66738860427828594624,
    'Other': 4.66649836094235492823,
    'Bed & Breakfast': 4.65718721444923300368
}

selected_propert_type = property_type_dict[option_property_type]
selected_property = "{:.20f}".format(selected_propert_type)
selected_property = float(selected_property)

# Neighbourhood Type
neighourhood_types = ["Echo Park", "Hollywood", "Hollywood Hills", "Koreatown", "Mid-Wilshire", "Other", "Silver Lake",
                      "Venice", "Westlake"]
option_neighbourhood_type = st.selectbox(
    'Which Neighbourhood?',
    ("Echo Park", "Hollywood", "Hollywood Hills", "Koreatown", "Mid-Wilshire", "Silver Lake", "Venice", "Westlake",
     "Other"))
other_neighbourhoods = [neighbourhood for neighbourhood in neighourhood_types if
                        neighbourhood != option_neighbourhood_type]

# Bedrooms
option_bedrooms_num = st.number_input('How many bedrooms?', min_value=0, max_value=10, value=1, step=1)
log_transformed_number = np.log(option_bedrooms_num + 1)

# Room Type
option_room_type = st.selectbox(
    'Which Room Type?',
    ("Entire home/apt", "Private room", "Shared room"))
room_type_dict = {
    'Entire home/apt': 4.937194,
    'Private room': 4.335201,
    'Shared room': 4.265124
}
df_model["room_type_encoded"].value_counts()
selected_room_type = room_type_dict[option_room_type]

# Bathrooms
option_bathrooms_num = st.number_input('How many bathrooms?', min_value=0.0, max_value=10.0, value=0.0, step=0.5)
log_transformed_bathrooms_number = np.log(np.log(option_bathrooms_num + 1) + 1)

# amenities list
amenities_list = ["Has_Air_conditioning", "Has_Pool", "Has_Gym", "Has_Shampoo", "Has_Self_Check-In",
                  "Has_Carbon_monoxide_detector", "Has_Private_entrance", "Has_Wheelchair_accessible",
                  "Has_Family/kid_friendly", "Has_TV", "Has_Free_parking_on_premises", "Has_Cable_TV",
                  "Has_Indoor_fireplace", "Has_Dryer", "Has_Washer", "Has_24-hour_check-in", "Has_Pets_allowed",
                  "Has_NoParties"]

# Checkbox'ları göster
st.write("Select amenities")
selected_amenities = [
    st.checkbox(amenity, key=amenity) for amenity in amenities_list]

# Seçilen özellikleri görüntüle
selected_amenities_dict = dict(zip(amenities_list, selected_amenities))

# Accomodates
option_accomodates_type = st.number_input(
    'How many accomodates?', value=1, min_value=0, max_value=16, step=1)
log_option_accomodates = np.log(option_accomodates_type + 1)

# Beds
option_beds_num = st.number_input(
    'How many beds?', value=1, min_value=0, max_value=16, step=1)
log_option_beds = math.sqrt(np.log(option_beds_num + 1))

# weekend or weekday # _weekend_or_weekday
selected_week_df = df_model[df_model[option_neighbourhood_type] == True]
week_df_mode = selected_week_df["_weekend_or_weekday"].mode().iloc[0]

# security deposit
options_secdop_num = st.number_input(
    'Security Deposit?', value=200, min_value=0, max_value=1000, step=50)
log_option_secdop = np.log1p(options_secdop_num)

# sqrt_log_guests_included
# df_model["sqrt_log_guests_included"].value_counts()
# option_sqrt_log_guests_included = st.number_input(
#   'How many guests included?', value=1, min_value=1, max_value=20, step=1)
# log_option_guests = math.sqrt(np.log(option_sqrt_log_guests_included + 1))
pd.set_option('display.float_format', '{:.20f}'.format)

# creating df_user based on user's selection
df_user = pd.DataFrame({"property_type_encoded": [selected_property],
                        option_neighbourhood_type: True,
                        "log_bedrooms": log_transformed_number,
                        "room_type_encoded": selected_room_type,
                        "log_accommodates": log_option_accomodates,
                        "log_log_bathrooms": log_transformed_bathrooms_number,
                        "sqrt_log_beds": log_option_beds,
                        "_weekend_or_weekday": week_df_mode,
                        "log_security_deposit": log_option_secdop  # ,
                        #  "sqrt_log_guests_included": log_option_guests
                        })

# adding neighbourhood columns except selected neighbourhood
for neighbourhood in other_neighbourhoods:
    df_user[neighbourhood] = False

# concating amenities list to df_user
df_user = pd.concat([df_user, pd.DataFrame(selected_amenities_dict, index=[0])], axis=1)

# modes of seasons
selected_neighbourhood_df = df_model[df_model[option_neighbourhood_type] == True]
_season_Spring_mode = selected_neighbourhood_df["_season_Spring"].mode().iloc[0]
_season_Spring_mode = bool(_season_Spring_mode)
_season_Summer_mode = selected_neighbourhood_df["_season_Summer"].mode().iloc[0]
_season_Summer_mode = bool(_season_Summer_mode)
_season_Winter_mode = selected_neighbourhood_df["_season_Winter"].mode().iloc[0]
_season_Winter_mode = bool(_season_Winter_mode)

# creating new dataframe for prediction
df_new = df_model[df_model[option_neighbourhood_type] == True].groupby(option_neighbourhood_type).agg(
    {"landmark_score": "mean",
     "latitude": "mean",
     "longitude": "mean",
     # "log_accommodates": "mean",
     # "log_security_deposit": "mean",
     "log_cleaning_fee": "mean",
     "log_extra_people": "mean",
     "log_number_of_reviews": "mean",
     "log_reviews_per_month": "mean",
     "_total_reviews": "mean",
     "booking_flexibility": "mean",
     "_weighted_total_review_score": "mean",
     "host_duration": "mean",
     "_host_activity_score": "mean",
     "_long_term_availability_score": "mean",
     "log_log_host_total_listings_count": "mean",
     "log__days_since_last_review": "mean",
     "log__host_about_length": "mean",
     "sqrt_log_guests_included": "mean",
     "sqrt_log_minimum_nights": "mean",
     "sqrt_log_calculated_host_listings_count": "mean",
     "sqrt_month": "mean",
     # "sqrt_log_beds": "mean",
     "sqrt__host_verification_count": "mean",
     "sqrt__listing_duration_days": "mean",
     # "_weekend_or_weekday": "mean",  # int
     "is_multi_host": "mean"
     })

# merging df_user & df_new dataframes
df_user = pd.merge(df_user, df_new, how="inner", on=option_neighbourhood_type)
df_new_months = pd.DataFrame({
    option_neighbourhood_type: True,
    "_season_Spring": [_season_Spring_mode],
    "_season_Summer": [_season_Summer_mode],
    "_season_Winter": [_season_Winter_mode]
}, index=[False])

# merging seasons modes to dataframes
df_user = pd.merge(df_user, df_new_months, how="inner", on=option_neighbourhood_type)


# PREDICTION

# Düğmeye basıldığında yapılacak işlemi belirleyen bir fonksiyon
def on_button_click():
    user_inputs = df_user.values
    prediction = joblib_model.predict(user_inputs)
    pred_val = np.exp(prediction[0] + 1)
    st.header("XGBoost :")
    st.success(f"${round(pred_val, 3)}")
    # st.markdown(f"<p style='font-size:45px;color:green;'>${round(pred_val, 3)}</p>", unsafe_allow_html=True)


# Streamlit uygulamasında düğmeyi oluşturmak
if st.button("Tahmin Et"):
    on_button_click()

# lgbm_model = joblib.load("DatalantaProject/model/lgbm_model.joblib")
# prediction2 = lgbm_model.predict(df_user.values)
# pred2_val = (np.exp(prediction2[0] + 1))
# st.header(f"LGBM : {round(pred2_val, 3)}")
