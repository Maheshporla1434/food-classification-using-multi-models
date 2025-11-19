# ============================================================
# FINAL MERGED APP.PY (TFLITE + JSON + SAFE GROUP REPORT + TEMP FOLDER + STRONG UPLOAD FIX)
# ============================================================

from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model  # fallback only
import numpy as np
import json
import os
import cv2
import tensorflow as tf
import requests

app = Flask(__name__)

MODELS_BASE = "models"
STATIC_TEMP = "static/temp.jpg"
FOOD_JSON_PATH = "food_json_files"
MODEL_JSON_PATH = "model_json_files"
IMG_SIZE = (256, 256)

# >>> UPLOAD FOLDER (added)
UPLOAD_FOLDER = "uploads"

# ------------------------------------------------------------
# CLASS – GROUP MAPPING
# ------------------------------------------------------------
ALL_CLASSES_GROUPS = {
    "Group_1": ["Baked Potato", "Crispy Chicken", "Donut"],
    "Group_2": ["Fries", "Hot Dog", "Sandwich"],
    "Group_3": ["apple_pie", "Taco", "Taquito"],
    "Group_4": ["burger", "butter_naan", "chai"],
    "Group_5": ["chapati", "cheesecake", "chicken_curry"],
    "Group_6": ["chole_bhature", "dal_makhani", "dhokla"],
    "Group_7": ["fried_rice", "ice_cream", "idli"],
    "Group_8": ["jalebi", "kaathi_rolls", "kadai_paneer"],
    "Group_9": ["kulfi", "masala_dosa", "momos"],
    "Group_10": ["omelette", "paani_puri", "pakode"],
    "Group_11": ["pav_bhaji", "pizza", "samosa", "sushi"]
}

# ------------------------------------------------------------
# GOOGLE DRIVE TFLITE LINKS
# ------------------------------------------------------------
MODEL_DOWNLOAD_LINKS = {
    "custom_models": {
        1:"https://drive.google.com/uc?export=download&id=111Co3RjKT6XFI41BicidFGRXQEh270z-",
        2:"https://drive.google.com/uc?export=download&id=1EtjxoV9aAOdp0cJkiOpGhtNlP9xwNhZE",
        3:"https://drive.google.com/uc?export=download&id=15LJYEhHDGETE2OkhROUFrEdgqCTCOyeu",
        4:"https://drive.google.com/uc?export=download&id=1yaJewoW6h9tuWQmnt1HajlsCSwtvPXvl",
        5:"https://drive.google.com/uc?export=download&id=1QLwCX8iX2kRnCiz6Z2ij24beAadFbeC9",
        6:"https://drive.google.com/uc?export=download&id=1-OH_WHHOfiMt_cDyv1n7AhCdqwfGnC96",
        7:"https://drive.google.com/uc?export=download&id=1GslnUM8UoY08Y-OnOq7I4IUGlyBfq2V2",
        8:"https://drive.google.com/uc?export=download&id=1YKlZBKR_z3tFjW7YzJImLEBlQruA2wCv",
        9:"https://drive.google.com/uc?export=download&id=1px_qKPsenR_PvxW-bJP7iCWy0LGFcUBs",
        10:"https://drive.google.com/uc?export=download&id=1EHo8FfNL7sEIjWD5GsKEUt9QzPF3l_qg",
        11:"https://drive.google.com/uc?export=download&id=1IxRSxFbcEGEEekCFX_vAG0RIH_7ffRSR"
    },
    "vgg16_models": {
        1:"https://drive.google.com/uc?export=download&id=11xBjM6deXQ7vAS_5wRztAK84l3LylHP9",
        2:"https://drive.google.com/uc?export=download&id=1T_a0ueZ8hidUNjJuG3d1Y8wL63GN5Hgw",
        3:"https://drive.google.com/uc?export=download&id=18JKOo65KWe95duCWYH4FxcSv-hK2BMXl",
        4:"https://drive.google.com/uc?export=download&id=16xzV2MV2u9pT_EAcd55PIjMZYe_Kdw4o",
        5:"https://drive.google.com/uc?export=download&id=1S8vqi2oe4dvJ7hYbX41a-qiF_D2Fgg8Q",
        6:"https://drive.google.com/uc?export=download&id=1q-f1cXL_nSXsxMnUr3g_z-NjWRcadoUg7",
        7:"https://drive.google.com/uc?export=download&id=1uPhEbVHGR84edN0YiHciVhQhtPdzPLUd",
        8:"https://drive.google.com/uc?export=download&id=1FWDxwhmF0HXsxBGKk306fwwE5j-9ZrAL",
        9:"https://drive.google.com/uc?export=download&id=1-IkV55ZCJszcSP88rSwNbTWvMTh3ZzaJ",
        10:"https://drive.google.com/uc?export=download&id=14qYDsaLG_kfFZdASJl6S6pxMu8mQVXil",
        11:"https://drive.google.com/uc?export=download&id=1NNedaopRaIwMPggPyHGZ1mVN4zJTFvfj"
    },
    "resnet50_models": {
        1:"https://drive.google.com/uc?export=download&id=1yFb68TyZqAOX8bcQncUFOgfk-tVzbr7k",
        2:"https://drive.google.com/uc?export=download&id=1bTNiCx3HfSCITjiZZBpQXRB6JzV-pN7U",
        3:"https://drive.google.com/uc?export=download&id=1IH6xidqv2jzZnWDx32RSL5B73p6yjLrc",
        4:"https://drive.google.com/uc?export=download&id=1XhzIAnzpDQHyRH9Z0ybRQmnPEHdiRmzc",
        5:"https://drive.google.com/uc?export=download&id=1vGYWY-auj-xisKt_RpKGbUHk8ju0_mk5",
        6:"https://drive.google.com/uc?export=download&id=15Z0ZVlFXybSaYvHfm-h52juFHlrOcEPF",
        7:"https://drive.google.com/uc?export=download&id=1FVUJEnig1naJ79A7B_jWReJWMg5ys3bB",
        8:"https://drive.google.com/uc?export=download&id=1sb7_IZlaEfB4C1mf0zWR1p8mPtp8tMzY",
        9:"https://drive.google.com/uc?export=download&id=1HdQwYdQOvRloveEknhzhZG4HEUZQQT05",
        10:"https://drive.google.com/uc?export=download&id=1qfZZ2rnJvkokJjtQmYd_I4LiLNIrg9Ci",
        11:"https://drive.google.com/uc?export=download&id=16Tk7TXRbME-KgSCYaGc6cdTo5CUNQMWi"
    }
}

# ------------------------------------------------------------
# TEMP MODEL DIRECTORY
# ------------------------------------------------------------
TEMP_MODEL_DIR = "temp_models"

def download_tflite_model(model_group, group_number):
    os.makedirs(TEMP_MODEL_DIR, exist_ok=True)

    for f in os.listdir(TEMP_MODEL_DIR):
        try:
            os.remove(os.path.join(TEMP_MODEL_DIR, f))
        except:
            pass

    url = MODEL_DOWNLOAD_LINKS[model_group][group_number]
    file_path = os.path.join(TEMP_MODEL_DIR, f"{model_group}_group{group_number}.tflite")

    r = requests.get(url)
    with open(file_path, "wb") as f:
        f.write(r.content)

    return file_path


def load_tflite_interpreter(model_group, model_name):
    group_number = int(model_name.split("group")[1].split(".")[0])
    model_path = download_tflite_model(model_group, group_number)
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# ------------------------------------------------------------
# LOAD BIG JSON METRICS
# ------------------------------------------------------------
BIG_JSONS = {}
if os.path.exists(MODEL_JSON_PATH):
    for f in os.listdir(MODEL_JSON_PATH):
        if f.endswith(".json"):
            try:
                BIG_JSONS[f] = json.load(open(os.path.join(MODEL_JSON_PATH, f)))
            except:
                BIG_JSONS[f] = {}

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def list_class_files():
    if not os.path.exists(FOOD_JSON_PATH):
        return []
    return [f for f in os.listdir(FOOD_JSON_PATH) if f.endswith(".json")]


def load_class_json(file_name):
    path = os.path.join(FOOD_JSON_PATH, file_name)
    if not os.path.exists(path):
        return {}
    return json.load(open(path))


def find_big_json(selected_class):
    for name, js in BIG_JSONS.items():
        if selected_class in js:
            return name, js
    return None, None


def get_model_file_for_class(selected_class, model_group):
    for g, cls_list in ALL_CLASSES_GROUPS.items():
        if selected_class in cls_list:
            group_number = int(g.split("_")[1])
            break

    return (
        f"custom_model_group{group_number}.tflite"
        if model_group == "custom_models" else
        f"vgg16_group{group_number}.tflite"
        if model_group == "vgg16_models" else
        f"resnet50_group{group_number}.tflite"
    )

# ------------------------------------------------------------
# SAFE GROUP REPORT
# ------------------------------------------------------------
def build_group_classification_report(group_json):
    report = {}
    p, r, f1, sup = [], [], [], []

    for cls, m in group_json.items():
        prec = m.get("precision")
        rec = m.get("recall")
        f1s = m.get("f1-score")
        support = m.get("support")

        report[cls] = {
            "precision": prec,
            "recall": rec,
            "f1-score": f1s,
            "support": support
        }

        if isinstance(prec, (int, float)): p.append(prec)
        if isinstance(rec, (int, float)): r.append(rec)
        if isinstance(f1s, (int, float)): f1.append(f1s)
        if isinstance(support, int): sup.append(support)

    report["macro avg"] = {
        "precision": sum(p)/len(p) if p else None,
        "recall": sum(r)/len(r) if r else None,
        "f1-score": sum(f1)/len(f1) if f1 else None,
        "support": sum(sup) if sup else None
    }

    return report

# ------------------------------------------------------------
# ROUTES
# ------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/list_json_files")
def route_list_json_files():
    return jsonify(list_class_files())

@app.route("/get_per_class")
def route_get_per_class():
    file = request.args.get("file")
    if not file:
        return jsonify({}), 400
    return jsonify(load_class_json(file))

@app.route("/find_big_json")
def route_find_big_json():
    cls = request.args.get("class")
    if not cls:
        return jsonify({}), 400

    name, js = find_big_json(cls)
    if not js:
        return jsonify({"found": False})

    return jsonify({
        "found": True,
        "big_json_file": name,
        "class_list": list(js.keys()),
        "metrics": js.get(cls, {})
    })

# ------------------------------------------------------------
# >>> STRONG UPLOAD FIX (ONLY CHANGE)
# ------------------------------------------------------------
@app.route("/upload", methods=["POST"])
def route_upload():
    f = request.files.get("image")
    if not f:
        return jsonify({"error":"No file"}), 400

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    # FULL HARD DELETE of old images
    for old in os.listdir(UPLOAD_FOLDER):
        old_path = os.path.join(UPLOAD_FOLDER, old)
        try:
            os.chmod(old_path, 0o777)  # unlock file on Windows
            os.remove(old_path)
        except Exception as e:
            print("Delete failed:", e)

    # save ONLY the new image
    ext = f.filename.split(".")[-1].lower()
    save_path = os.path.join(UPLOAD_FOLDER, f"uploaded_image.{ext}")
    f.save(save_path)

    # update static/temp.jpg for prediction
    os.makedirs(os.path.dirname(STATIC_TEMP), exist_ok=True)
    img = cv2.imread(save_path)
    if img is not None:
        cv2.imwrite(STATIC_TEMP, img)

    return jsonify({"status":"ok"})

# ------------------------------------------------------------
# PREDICTION HANDLER
# ------------------------------------------------------------
def predict_image_from_file(model):
    if not os.path.exists(STATIC_TEMP):
        raise FileNotFoundError("Uploaded image missing")

    img = cv2.imread(STATIC_TEMP)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32)/255.0
    img = np.expand_dims(img, 0)

    try:
        inp = model.get_input_details()
        out = model.get_output_details()
        model.set_tensor(inp[0]['index'], img)
        model.invoke()
        preds = model.get_tensor(out[0]['index'])[0]
    except:
        preds = model.predict(img)[0]

    idx = int(np.argmax(preds))
    return idx, float(preds[idx]), preds.tolist()

# ------------------------------------------------------------
# FINAL PREDICT API
# ------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def route_predict():
    data = request.get_json() or {}
    selected_class = data.get("selected_class")
    model_group = data.get("model_group")

    if not selected_class:
        return jsonify({"error":"selected_class required"}), 400
    if not model_group:
        return jsonify({"error":"model_group required"}), 400

    _, big_json = find_big_json(selected_class)
    metrics = big_json.get(selected_class, {})

    model_file = get_model_file_for_class(selected_class, model_group)
    model = load_tflite_interpreter(model_group, model_file)

    pred_idx, conf, preds = predict_image_from_file(model)

    class_list = list(big_json.keys())
    predicted_class = class_list[pred_idx] if pred_idx < len(class_list) else "IndexOutOfRange"
    group_report = build_group_classification_report(big_json)

    return jsonify({
        "selected_class":selected_class,
        "predicted_class":predicted_class,
        "model_used":model_group,
        "model_name":model_file,
        "confidence":round(conf,4),
        "accuracy":metrics.get("accuracy"),
        "precision":metrics.get("precision"),
        "recall":metrics.get("recall"),
        "confusion_matrix":metrics.get("confusion_matrix"),
        "classification_report":big_json,
        "group_classification_report":group_report,
        "full_predictions":preds
    })

# ------------------------------------------------------------
# RUN APP
# ------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
