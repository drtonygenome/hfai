import pandas as pd
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

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu',
                          input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(
    patience=10, restore_best_weights=True)

history = model.fit(X_train_scaled, y_train, validation_data=(X_eval_scaled, y_eval),
                    epochs=10000, callbacks=[early_stopping], verbose=1)

# Dừng training khi độ chính xác > 95%
for i in range(len(history.history['val_accuracy'])):
    if history.history['val_accuracy'][i] > 0.95:
        model.stop_training = True
        break

model.save("heart_failure_clinical_model")

loaded_model = tf.keras.models.load_model("heart_failure_clinical_model")

eval_loss, eval_acc = loaded_model.evaluate(X_eval_scaled, y_eval)
print(f"\nĐộ chính xác của model đã tải trên dữ liệu đánh giá với 50 record thử nghiệm: {eval_acc * 100:.2f}%")

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
