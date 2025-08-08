import streamlit as st
import pandas as pd
import joblib

# Load model & encoders
model = joblib.load("RandomForestClassifier_model.pkl") 
encoders = joblib.load("label_encoders.pkl") 

st.title("🎬 Movie Success Prediction App")
st.markdown("""
## 🎬 Movie Success Prediction App

This web app predicts whether a movie will be a **Flop**, **Average**, or **Hit**  
based on its production details, cast, budget, and popularity metrics.

### 📌 How it Works
1. Select movie details from the dropdowns (real names for director, actors, etc.).
2. The app uses pre-trained **Label Encoders** to convert names/labels into numeric form.
3. These values are fed into a trained **Random Forest Classifier** model.
4. The model predicts the movie's success category.

### 📂 Dataset & Model
- **Dataset: https://github.com/ProgrammerNikhil/Movie-Prediction-Model  
- **Model:** Trained Random Forest Classifier with categorical label encoding.

**Developer:** Nikhil Patel  
**Libraries Used:** Streamlit, Pandas, Joblib, Scikit-learn  
""")
st.write("Select movie details below to predict whether it will be a Flop, Average, or Hit.")
#GBI INTERFACE
director_name = st.selectbox("Director Name", encoders["le_dir"].classes_)
actor_1_name = st.selectbox("Actor 1 Name", encoders["le_ac1"].classes_)
actor_2_name = st.selectbox("Actor 2 Name", encoders["le_ac2"].classes_)
actor_3_name = st.selectbox("Actor 3 Name", encoders["le_ac3"].classes_)
genres = st.selectbox("Genre", encoders["le_gen"].classes_)
language = st.selectbox("Language", encoders["le_lang"].classes_)
country = st.selectbox("Country", encoders["le_coun"].classes_)
content_rating = st.selectbox("Content Rating", encoders["le_ra"].classes_)
plot_keywords = st.selectbox("Plot Keywords", encoders["le_keyw"].classes_)

num_critic_for_reviews = st.number_input("Number of Critic Reviews", value=0.0)
director_facebook_likes = st.number_input("Director Facebook Likes", value=0.0)
gross = st.number_input("Gross (in billions)", value=0.0)
num_voted_users = st.number_input("Number of Voted Users", value=0)
cast_total_facebook_likes = st.number_input("Cast Total Facebook Likes", value=0)
facenumber_in_poster = st.number_input("Face Number in Poster", value=0)
num_user_for_reviews = st.number_input("Number of User Reviews", value=0.0)
budget = st.number_input("Budget (in $)", value=0.0)
aspect_ratio = st.number_input("Aspect Ratio", value=0.0)
movie_facebook_likes = st.number_input("Movie Facebook Likes", value=0)

# Predict button pr pehle encode karega input ko then input me encoded data lenge var ka 
if st.button("Predict"):
    encoded_director = encoders["le_dir"].transform([director_name])[0]
    encoded_actor1 = encoders["le_ac1"].transform([actor_1_name])[0]
    encoded_actor2 = encoders["le_ac2"].transform([actor_2_name])[0]
    encoded_actor3 = encoders["le_ac3"].transform([actor_3_name])[0]
    encoded_genre = encoders["le_gen"].transform([genres])[0]
    encoded_lang = encoders["le_lang"].transform([language])[0]
    encoded_country = encoders["le_coun"].transform([country])[0]
    encoded_rating = encoders["le_ra"].transform([content_rating])[0]
    encoded_keywords = encoders["le_keyw"].transform([plot_keywords])[0]

    # input data me encoded data 
    input_data = pd.DataFrame({
        'director_name': [encoded_director],
        'num_critic_for_reviews': [num_critic_for_reviews],
        'director_facebook_likes': [director_facebook_likes],
        'actor_2_name': [encoded_actor2],
        'gross': [gross],
        'genres': [encoded_genre],
        'actor_1_name': [encoded_actor1],
        'num_voted_users': [num_voted_users],
        'cast_total_facebook_likes': [cast_total_facebook_likes],
        'actor_3_name': [encoded_actor3],
        'facenumber_in_poster': [facenumber_in_poster],
        'plot_keywords': [encoded_keywords],
        'num_user_for_reviews': [num_user_for_reviews],
        'language': [encoded_lang],
        'country': [encoded_country],
        'content_rating': [encoded_rating],
        'budget': [budget],
        'aspect_ratio': [aspect_ratio],
        'movie_facebook_likes': [movie_facebook_likes]
    })

    predicted_score = model.predict(input_data)[0]

    # 0,1,2 ko devide kr diya 
    if predicted_score == 0:
        st.error("🎯 Prediction: Flop Movie")
    elif predicted_score == 1:
        st.warning("🎯 Prediction: Average Movie")
    elif predicted_score == 2:
        st.success("🎯 Prediction: Hit Movie")
    else:
        st.info("Invalid Score")
