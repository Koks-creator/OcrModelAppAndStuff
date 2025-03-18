import sys
from pathlib import Path
import os
sys.path.append(str(Path(__file__).resolve().parent.parent))

from flask import render_template, session, redirect, url_for, flash, request
import base64
import tensorflow as tf

from config import Config
from ocr_webapp import app, forms, ocr_predictor


@app.route("/", methods=["GET", "POST"])
def home():
    try:
        form = forms.MainForm()
        image_datas = []
        image_filenames = []
        predictions = []
        tf_images = []
        if not app.config["TESTING"]:
            val_mode = 0
            form_validation = form.validate_on_submit()
        else:
            val_mode = 1
            form_validation = request.method == 'POST'

        if form_validation:
            # this if is made to prevent errors when using not allowed file extension in testing mode
            if val_mode == 1:
                for img in form.images.data:
                    if os.path.splitext(img.filename)[1] not in (".png", ".jpg", ".jpeg"):
                        return {"Status": "File should be png or jpg"}, 400
            
            app.logger.info(f"Uploaded {len(form.images.data)} files")
            for image_file in form.images.data:
                content = image_file.read()
                
                encoded_img = base64.b64encode(content).decode("utf-8")
                image_data = f"data:image/png;base64,{encoded_img}"
                image_datas.append(image_data)
                image_filenames.append(image_file.filename)
                
                try:
                    image_tf = tf.image.decode_png(content, channels=1)
                except Exception as e:
                    image_tf = tf.image.decode_jpeg(content, channels=1)
                tf_images.append(image_tf)
            predictions = ocr_predictor.get_predictions(
                images=tf_images,
                img_size=(Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH)
            )
            app.logger.info(f"{predictions=}")
        if image_datas:
            res = zip(image_datas, predictions)
        else:
            res = []
        res_to_save = {
            "Filename": image_filenames,
            "Prediction": predictions
        }
        return render_template("home.html", form=form, image_datas=res, res_to_save=res_to_save)
    except Exception as e:
        app.logger.error(f"Unknown error: {e}")
        return redirect(url_for("error_page", error=e, status_code=500))
    

@app.route("/error", methods=["GET"])
def error_page():
    error_msg = request.args.get("error", "Unknown error occured")
    error_status_code = request.args.get("status_code", 500)
    try:
        error_status_code = int(error_status_code)
    except ValueError:
        error_status_code = 500
    
    return render_template("error_page.html", status_code=500, error_text=error_msg)


@app.route('/static/sw.js')
def service_worker():
    return app.send_static_file('sw.js')

@app.route('/manifest.json')
def manifest():
    return app.send_static_file('manifest.json')