#train.py
import pandas as pd
from sklearn.utils import shuffle
from encoding import encode
from models import get_model , evaluate_model
import joblib
from load_data import load_process_save


df,val,test = load_process_save()
df = shuffle(df, random_state=43)

preprocessor = encode(df)
X_train = preprocessor.fit_transform(df)
X_val = preprocessor.transform(val)
X_test = preprocessor.transform(test)

y_train = df['log_trip_duration']
y_val = val['log_trip_duration']
y_test = test['log_trip_duration']

models = get_model()
for name ,model in models.items():
    r2 = evaluate_model(model,X_train,X_val,X_test,y_train,y_val,y_test)
    print('Name: ',name)
    for k , v in r2.items():
        print(k,v,'\n')

    #joblib.dump(model, f"{name}.pkl")


