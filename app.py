import uvicorn 
from fastapi import FastAPI
import joblib,os
gender_vectorizer = open("models/gender_vectorizer.pkl","rb") 
#will open gender vectorizer model, that uses NLP to get the root of the words. check point(1) in the documents for content of pkl.
gender_cv = joblib.load(gender_vectorizer)
# joblib is to provide lightweight pipelining.
gender_nv_model = open("models/gender_nv_model.pkl","rb")
#will open gender nv model, that will predict the name's gender.
gender_clf = joblib.load(gender_nv_model)

app = FastAPI()

@app.get('/predict/{name}') #the process will be held in localhost:PORT/predict/"whatever name is inputted"
async def predict(name): # name of the process
    vectorized_name = gender_cv.transform([name]).toarray() 
    #the name will be put into the array and through some process the string will be cleaned from punctations and whitespaces
    #then it will be put into the list of models
    prediction = gender_clf.predict(vectorized_name)
    #will predict the name
    if prediction[0] == 0:
        result = 'female'   
    else:
        result = 'male'
    return {"orig_name" : name,"prediction":result}

if __name__ == '__main__':
    uvicorn.run(app,host="127.0.0.1",port=8000)
