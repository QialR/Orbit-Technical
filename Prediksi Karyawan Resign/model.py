import pickle

# global variable
global model, scaler

def load():
    global model, scaler
    model = pickle.load(open('model/model_ds.pkl', 'rb'))
    scaler = pickle.load(open('model/scaler_ds.pkl', 'rb'))

def prediksi(data):
    data = scaler.transform(data)
    prediksi = int(model.predict(data))
    nilai_kepercayaan = model.predict_proba(data).flatten()
    nilai_kepercayaan = max(nilai_kepercayaan) * 100
    nilai_kepercayaan = round(nilai_kepercayaan)

    if prediksi == 0:
        hasil_prediksi = "Tidak Resign"
    else:
        hasil_prediksi = "Akan Resign"
    return hasil_prediksi, nilai_kepercayaan