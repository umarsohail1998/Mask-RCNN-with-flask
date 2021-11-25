from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import threading
import ml

app = Flask(__name__)
app.config['UPLOAD_PATH'] = './uploads'

@app.route('/')
def index():
    files = os.listdir('./predict')
    return render_template('index.html', files=files)

@app.route('/', methods=['POST'])
def upload_file():
    uploaded_file = request.files['img']
    if uploaded_file.filename != '':
      uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], uploaded_file.filename))

    # ml.func(uploaded_file.filename)
    threading.Thread(target=ml.func, args=(uploaded_file.filename,)).start()
    return redirect(url_for('index'))

@app.route('/uploads/<filename>')
def upload(filename):
    return send_from_directory('./predict', filename)
 
if __name__ == '__main__':
   app.run(debug = True)

