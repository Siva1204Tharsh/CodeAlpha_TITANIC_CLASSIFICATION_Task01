import pandas as pd
from flask import Flask, request, render_template , jsonify
import pickle

app = Flask(__name__)
# Load your model (replace 'model.pkl' with your model file)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
  return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    data = request.json
    print(data)
    # Extract features from the request
    features = [
        data['Pclass'], data['Sex'],data['Age'], data['SibSp'], 
        data['Parch'], data['Fare'],  data['Embarked']
    ]
    # Preprocess as needed (convert strings to numbers, etc.)
    features[1] = 1 if features[1] == 'Male' else 0  # Example for 'Sex'
    features[6] = {'S': 0, 'C': 1, 'Q': 2}[features[6]]  # Example for 'Embarked'
    features=list(map(int,features))
    print(features) #['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    df=pd.DataFrame([features],columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])
    print(df)
    prediction = model.predict(df)  # Adjust as needed for your model
    result = "Survived" if prediction[0] == 1 else "Did not survive"
    
    return jsonify({'prediction': result})
    #return render_template('index.html', prediction_text=f"Result: {result}")

if __name__ == '__main__':
    app.run(debug=True)
