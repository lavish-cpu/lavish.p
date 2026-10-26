import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title='TerraNautics AI', page_icon='🏡', layout='wide')
st.title('TerraNautics AI')
st.caption('Educational property-price modelling demo • synthetic data only')

@st.cache_data
def make_data(n=1200, seed=42):
    rng = np.random.default_rng(seed)
    area = rng.integers(450, 3500, n)
    bedrooms = rng.integers(1, 6, n)
    age = rng.integers(0, 35, n)
    location_score = rng.uniform(1, 10, n)
    price = area * (3500 + location_score * 850) + bedrooms * 350000 - age * 18000 + rng.normal(0, 250000, n)
    return pd.DataFrame({'area_sqft': area, 'bedrooms': bedrooms, 'property_age': age, 'location_score': location_score, 'price_inr': np.maximum(price, 100000)})

df = make_data()
X = df.drop(columns='price_inr')
y = df['price_inr']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=160, random_state=42, min_samples_leaf=2, n_jobs=-1)
model.fit(X_train, y_train)
pred = model.predict(X_test)
a, b, c = st.columns(3)
a.metric('MAE (demo test split)', f'₹{mean_absolute_error(y_test, pred):,.0f}')
b.metric('RMSE', f'₹{mean_squared_error(y_test, pred) ** 0.5:,.0f}')
c.metric('R²', f'{r2_score(y_test, pred):.3f}')
st.subheader('Estimate a sample property')
left, right = st.columns(2)
area = left.slider('Area (sq ft)', 450, 3500, 1200, 50)
rooms = right.slider('Bedrooms', 1, 5, 2)
age = left.slider('Property age (years)', 0, 35, 5)
location = right.slider('Location score (1–10)', 1.0, 10.0, 6.5, 0.5)
if st.button('Estimate price', type='primary'):
    sample = pd.DataFrame([{'area_sqft': area, 'bedrooms': rooms, 'property_age': age, 'location_score': location}])
    st.success(f'Illustrative model estimate: ₹{model.predict(sample)[0]:,.0f}')
st.subheader('Synthetic dataset preview')
st.dataframe(df.head(12), use_container_width=True)
st.bar_chart(pd.Series(model.feature_importances_, index=X.columns).sort_values())
st.info('Synthetic data is designed to demonstrate a workflow. It cannot estimate actual market prices or investment returns.')
