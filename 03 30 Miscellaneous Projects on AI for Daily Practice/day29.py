# Import Libraries
import pandas as pd
import numpy as np
from surprise import SVD
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate, train_test_split
from surprise import accuracy
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load Dataset
# Load MovieLens dataset
df = pd.read_csv('ratings.csv')
df.drop('timestamp', axis=1, inplace=True)

# Display the first few rows
print("Dataset Preview:")
print(df.head())

# Step 2: Preprocessing the Dataset
# Define a Reader object for Surprise
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)

# Step 3: Split Data into Training and Testing Sets
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Step 4: Build Collaborative Filtering Model using SVD
model = SVD()
model.fit(trainset)

# Step 5: Evaluate the Model
predictions = model.test(testset)
rmse = accuracy.rmse(predictions)
print(f"RMSE: {rmse:.4f}")

# Step 6: Make a Prediction for a Specific User and Movie
user_id = 196
movie_id = 242

predicted_rating = model.predict(user_id, movie_id).est
print(f"Predicted Rating for User {user_id} on Movie {movie_id}: {predicted_rating:.2f}")

# Step 7: Visualize Distribution of Ratings
plt.figure(figsize=(10, 6))
sns.histplot(df['rating'], bins=5, kde=True)
plt.title('Distribution of Movie Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()


















