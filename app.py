from flask import Flask, request, render_template
from sklearn.preprocessing import StandardScaler
import pandas as pd
import keras
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("training_data.csv")
test_data = pd.read_csv("confirm_data.csv")

train_data = data.iloc[:-50]
eval_data = data.iloc[-50:]

X_train = train_data.drop(columns=["DEATH_EVENT"])
y_train = train_data["DEATH_EVENT"]

X_eval = eval_data.drop(columns=["DEATH_EVENT"])
y_eval = eval_data["DEATH_EVENT"]

# Chuẩn hóa dữ liệu sử dụng StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_eval_scaled = scaler.transform(X_eval)

loaded_model = tf.keras.models.load_model("heart_failure_prediction.h5")
#loaded_model = keras.layers.TFSMLayer(heart_failure_clinical_model.keras, call_endpoint='serving_default')

eval_loss, eval_acc = loaded_model.evaluate(X_eval_scaled, y_eval)
print(
    f"\nacc50: {eval_acc * 100:.2f}%")

X_test = test_data.drop(columns=["DEATH_EVENT"])
y_test = test_data["DEATH_EVENT"]

X_test_scaled = scaler.transform(X_test)

predictions = loaded_model.predict(X_test_scaled)

predicted_classes = (predictions > 0.5).astype(int)

correct_preds = predicted_classes.flatten() == y_test.values
correct_count = sum(correct_preds)
print(f"Số lượng dự đoán chính xác: {correct_count}/{len(y_test)}")

for i in range(len(y_test)):
    if correct_preds[i]:
        data_row = test_data.iloc[i]
        data_str = "|".join(map(str, data_row))
        print(
            f"Sample {i + 1}: Predicted {predicted_classes[i][0]}, Actual {y_test[i]} --(CORRECT)-- of Record data: {data_str}")
    else:
        data_row = test_data.iloc[i]
        data_str = ",".join(map(str, data_row))
        print(
            f"Sample {i + 1}: Predicted {predicted_classes[i][0]}, Actual {y_test[i]} --(INCORRECT)-- of Record data: {data_str}")


# ---------------APP-------------------- #
app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Get user input from the web form
        age = float(request.form["age"])
        anaemia = int(request.form["anaemia"])
        creatinine_phosphokinase = float(
            request.form["creatinine_phosphokinase"])
        diabetes = int(request.form["diabetes"])
        ejection_fraction = float(request.form["ejection_fraction"])
        high_blood_pressure = int(request.form["high_blood_pressure"])
        platelets = float(request.form["platelets"])
        serum_creatinine = float(request.form["serum_creatinine"])
        serum_sodium = float(request.form["serum_sodium"])
        sex = int(request.form["sex"])
        smoking = int(request.form["smoking"])
        time = float(request.form["time"])

        # Create a DataFrame from user input
        input_data = pd.DataFrame({
            "age": [age],
            "anaemia": [anaemia],
            "creatinine_phosphokinase": [creatinine_phosphokinase],
            "diabetes": [diabetes],
            "ejection_fraction": [ejection_fraction],
            "high_blood_pressure": [high_blood_pressure],
            "platelets": [platelets],
            "serum_creatinine": [serum_creatinine],
            "serum_sodium": [serum_sodium],
            "sex": [sex],
            "smoking": [smoking],
            "time": [time]
        })

        # Scale the input data using the pre-trained scaler
        input_data_scaled = scaler.transform(input_data)

        # Make a prediction using the loaded model
        prediction = loaded_model.predict(input_data_scaled)
        predicted_classe = (prediction > 0.5).astype(int)
        print(input_data)
        print(prediction)
        print(predicted_classe)
        # Chuyển đổi giá trị prediction và predicted_classe thành chuỗi
        predicted_classe_str = str(predicted_classe[0][0])
        prediction_str = "{:.2f}%".format(prediction[0][0]*100)

        # Determine the result (DEATH_EVENT)
        result = "Possible" if predicted_classe[0][0] == 1 else "Not Possible"

        return render_template("result.html", result=result, input_data=input_data, prediction=prediction_str, predicted_classe=predicted_classe_str)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)
