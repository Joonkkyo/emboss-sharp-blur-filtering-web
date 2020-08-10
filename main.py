from flask import Flask, request, render_template, redirect
import cv2
import numpy as np

app = Flask(__name__)


@app.route('/')
def index():
    return render_template("imageprocessing.html", ctx={"title": "영상처리"})


@app.route('/upload', methods=["POST"])
def upload():
    global image
    f = request.files['file1']
    filename = "./static/" + f.filename
    f.save(filename)
    image = cv2.imread(filename)
    cv2.imwrite("./static/result.jpg", image)
    print(image.shape)

    return redirect('/')


@app.route('/imageprocess')
def imageprocess():
    global image
    method = request.args.get("method")
    if method == "emboss":
        print("emboss")
        print(image.shape)
        emboss = np.array([
            [-1, -1, 0],
            [-1, 0, 1],
            [0, 1, 1]], np.float32)
        dst = cv2.filter2D(image, -1, emboss, delta=128)
        cv2.imwrite("./static/result.jpg", np.hstack((image, dst)))

    if method == "sharp":
        print("sharp")
        print(image.shape)
        sharp = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]], np.float32)
        dst = cv2.filter2D(image, -1, sharp)
        cv2.imwrite("./static/result.jpg", np.hstack((image, dst)))

    if method == "blur":
        print("sharp")
        print(image.shape)
        size = int(request.args.get("size", 3))
        dst = cv2.blur(image, (size, size))
        cv2.imwrite("./static/result.jpg", np.hstack((image, dst)))

    return "hello"


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8000)
