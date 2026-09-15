from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import numpy as np
import os
import uuid

app = Flask(__name__)

# ============================================================
# SETTINGS
# ============================================================

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Your USN / Roll Number
ROLL_NO = "CS24103"


# ============================================================
# PRACTICAL LIST
# Website numbering starts from 01
# ============================================================

PRACTICALS = {
    1: "RGB / Grayscale, Arithmetic & Bitwise Operations",
    2: "2-D Geometric Transformations",
    3: "Histogram Equalization, Spatial Enhancement & Thresholding",
    4: "Spatial Domain Filters",
    5: "Image Inpainting",
    6: "Lossless Image Compression",
    7: "Morphological Operations",
    8: "Object Detection using Correlation",
    9: "Top-Hat Transformation",
    10: "Colour Space Conversion",
    11: "Edge Detection"
}


# ============================================================
# ADD USN TO OUTPUT IMAGE
# ============================================================

def add_roll_no(image):
    """
    Adds CS24103 to the bottom-right corner
    of every generated output image.
    """

    if image is None:
        return image

    # If output is grayscale, convert it to BGR
    # so that the text can be drawn properly.
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image = image.copy()

    height, width = image.shape[:2]

    # Text size changes according to image width
    font_scale = max(0.6, min(1.2, width / 900))

    thickness = max(
        1,
        int(round(font_scale * 2))
    )

    font = cv2.FONT_HERSHEY_SIMPLEX

    text_size, baseline = cv2.getTextSize(
        ROLL_NO,
        font,
        font_scale,
        thickness
    )

    text_width, text_height = text_size

    margin = max(
        15,
        int(width * 0.02)
    )

    # Bottom-right position
    x = width - text_width - margin
    y = height - margin

    # White background rectangle
    cv2.rectangle(
        image,
        (
            x - 10,
            y - text_height - 10
        ),
        (
            x + text_width + 10,
            y + 10
        ),
        (255, 255, 255),
        -1
    )

    # Write USN
    cv2.putText(
        image,
        ROLL_NO,
        (x, y),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA
    )

    return image


# ============================================================
# READ IMAGE
# ============================================================

def read_image(file_storage, grayscale=False):

    if file_storage is None:
        return None

    if file_storage.filename == "":
        return None

    data = np.frombuffer(
        file_storage.read(),
        np.uint8
    )

    if data.size == 0:
        return None

    if grayscale:
        return cv2.imdecode(
            data,
            cv2.IMREAD_GRAYSCALE
        )

    return cv2.imdecode(
        data,
        cv2.IMREAD_COLOR
    )


# ============================================================
# RESIZE SECOND IMAGE
# ============================================================

def resize_to_match(image1, image2):

    if image1.shape[:2] != image2.shape[:2]:

        image2 = cv2.resize(
            image2,
            (
                image1.shape[1],
                image1.shape[0]
            )
        )

    return image2


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html",
        practicals=PRACTICALS
    )


# ============================================================
# PRACTICAL PAGE
# ============================================================

@app.route("/practical/<int:number>")
def practical(number):

    if number not in PRACTICALS:
        return "Practical not found", 404

    return render_template(
        "practical.html",
        number=number,
        title=PRACTICALS[number]
    )


# ============================================================
# PROCESS IMAGE
# ============================================================

@app.route(
    "/process/<int:number>",
    methods=["POST"]
)
def process(number):

    if number not in PRACTICALS:

        return jsonify({
            "error": "Invalid practical number."
        }), 400

    # --------------------------------------------------------
    # Read main input image
    # --------------------------------------------------------

    image = read_image(
        request.files.get("image")
    )

    if image is None:

        return jsonify({
            "error": "Please select a valid input image."
        }), 400

    operation = request.form.get(
        "operation",
        ""
    ).strip()

    if not operation:

        return jsonify({
            "error": "Please select an operation."
        }), 400

    result = None
    extra = {}

    try:

        # ====================================================
        # PRACTICAL 01
        # RGB / GRAYSCALE / ARITHMETIC / BITWISE
        # ====================================================

        if number == 1:

            # -------------------------------
            # Grayscale
            # -------------------------------

            if operation == "grayscale":

                result = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2GRAY
                )

            # -------------------------------
            # RGB
            # -------------------------------

            elif operation == "rgb":

                rgb_image = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB
                )

                # Convert back for correct image saving
                result = cv2.cvtColor(
                    rgb_image,
                    cv2.COLOR_RGB2BGR
                )

            # -------------------------------
            # Addition
            # -------------------------------

            elif operation == "add":

                image2 = read_image(
                    request.files.get("image2")
                )

                if image2 is None:

                    return jsonify({
                        "error":
                        "Please select the second image."
                    }), 400

                image2 = resize_to_match(
                    image,
                    image2
                )

                result = cv2.add(
                    image,
                    image2
                )

            # -------------------------------
            # Subtraction
            # -------------------------------

            elif operation == "subtract":

                image2 = read_image(
                    request.files.get("image2")
                )

                if image2 is None:

                    return jsonify({
                        "error":
                        "Please select the second image."
                    }), 400

                image2 = resize_to_match(
                    image,
                    image2
                )

                result = cv2.subtract(
                    image,
                    image2
                )

            # -------------------------------
            # Bitwise NOT
            # -------------------------------

            elif operation == "bitwise_not":

                result = cv2.bitwise_not(
                    image
                )

            # -------------------------------
            # Bitwise AND
            # -------------------------------

            elif operation == "and":

                image2 = read_image(
                    request.files.get("image2")
                )

                if image2 is None:

                    return jsonify({
                        "error":
                        "Please select the second image."
                    }), 400

                image2 = resize_to_match(
                    image,
                    image2
                )

                result = cv2.bitwise_and(
                    image,
                    image2
                )

            # -------------------------------
            # Bitwise OR
            # -------------------------------

            elif operation == "or":

                image2 = read_image(
                    request.files.get("image2")
                )

                if image2 is None:

                    return jsonify({
                        "error":
                        "Please select the second image."
                    }), 400

                image2 = resize_to_match(
                    image,
                    image2
                )

                result = cv2.bitwise_or(
                    image,
                    image2
                )

            # -------------------------------
            # Bitwise XOR
            # -------------------------------

            elif operation == "xor":

                image2 = read_image(
                    request.files.get("image2")
                )

                if image2 is None:

                    return jsonify({
                        "error":
                        "Please select the second image."
                    }), 400

                image2 = resize_to_match(
                    image,
                    image2
                )

                result = cv2.bitwise_xor(
                    image,
                    image2
                )

            else:

                return jsonify({
                    "error":
                    "Invalid Practical 01 operation."
                }), 400


        # ====================================================
        # PRACTICAL 02
        # GEOMETRIC TRANSFORMATIONS
        # ====================================================

        elif number == 2:

            height, width = image.shape[:2]

            # -------------------------------
            # Translation
            # -------------------------------

            if operation == "translation":

                tx = float(
                    request.form.get(
                        "tx",
                        100
                    )
                )

                ty = float(
                    request.form.get(
                        "ty",
                        50
                    )
                )

                matrix = np.float32([
                    [1, 0, tx],
                    [0, 1, ty]
                ])

                result = cv2.warpAffine(
                    image,
                    matrix,
                    (width, height)
                )

            # -------------------------------
            # Rotation
            # -------------------------------

            elif operation == "rotation":

                angle = float(
                    request.form.get(
                        "value",
                        30
                    )
                )

                center = (
                    width // 2,
                    height // 2
                )

                matrix = cv2.getRotationMatrix2D(
                    center,
                    angle,
                    1.0
                )

                result = cv2.warpAffine(
                    image,
                    matrix,
                    (width, height)
                )

            # -------------------------------
            # Scaling
            # -------------------------------

            elif operation == "scaling":

                scale = float(
                    request.form.get(
                        "value",
                        1.5
                    )
                )

                if scale <= 0:

                    return jsonify({
                        "error":
                        "Scaling value must be greater than 0."
                    }), 400

                result = cv2.resize(
                    image,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_LINEAR
                )

            # -------------------------------
            # X Shearing
            # -------------------------------

            elif operation == "shearing_x":

                shear = float(
                    request.form.get(
                        "value",
                        0.3
                    )
                )

                matrix = np.float32([
                    [1, shear, 0],
                    [0, 1, 0]
                ])

                new_width = int(
                    width +
                    abs(shear) * height
                )

                result = cv2.warpAffine(
                    image,
                    matrix,
                    (new_width, height)
                )

            # -------------------------------
            # Y Shearing
            # -------------------------------

            elif operation == "shearing_y":

                shear = float(
                    request.form.get(
                        "value",
                        0.3
                    )
                )

                matrix = np.float32([
                    [1, 0, 0],
                    [shear, 1, 0]
                ])

                new_height = int(
                    height +
                    abs(shear) * width
                )

                result = cv2.warpAffine(
                    image,
                    matrix,
                    (width, new_height)
                )

            # -------------------------------
            # Reflection
            # -------------------------------

            elif operation == "reflection":

                result = cv2.flip(
                    image,
                    1
                )

            # -------------------------------
            # Cropping
            # -------------------------------

            elif operation == "cropping":

                x1 = int(width * 0.20)
                x2 = int(width * 0.80)

                y1 = int(height * 0.20)
                y2 = int(height * 0.80)

                result = image[
                    y1:y2,
                    x1:x2
                ].copy()

            else:

                return jsonify({
                    "error":
                    "Invalid Practical 02 operation."
                }), 400


        # ====================================================
        # PRACTICAL 03
        # IMAGE ENHANCEMENT
        # ====================================================

        elif number == 3:

            # -------------------------------
            # Histogram Equalization
            # -------------------------------

            if operation == "histogram_equalization":

                gray = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2GRAY
                )

                result = cv2.equalizeHist(
                    gray
                )

            # -------------------------------
            # Smoothing
            # -------------------------------

            elif operation == "smoothing":

                result = cv2.GaussianBlur(
                    image,
                    (5, 5),
                    0
                )

            # -------------------------------
            # Sharpening
            # -------------------------------

            elif operation == "sharpening":

                kernel = np.array([
                    [0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]
                ])

                result = cv2.filter2D(
                    image,
                    -1,
                    kernel
                )

            # -------------------------------
            # Thresholding
            # -------------------------------

            elif operation == "threshold":

                threshold_value = int(
                    float(
                        request.form.get(
                            "value",
                            127
                        )
                    )
                )

                threshold_value = max(
                    0,
                    min(
                        255,
                        threshold_value
                    )
                )

                gray = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2GRAY
                )

                _, result = cv2.threshold(
                    gray,
                    threshold_value,
                    255,
                    cv2.THRESH_BINARY
                )

            else:

                return jsonify({
                    "error":
                    "Invalid Practical 03 operation."
                }), 400


        # ====================================================
        # PRACTICAL 04
        # SPATIAL FILTERS
        # ====================================================

        elif number == 4:

            # -------------------------------
            # Averaging
            # -------------------------------

            if operation == "averaging":

                result = cv2.blur(
                    image,
                    (5, 5)
                )

            # -------------------------------
            # Gaussian
            # -------------------------------

            elif operation == "gaussian":

                result = cv2.GaussianBlur(
                    image,
                    (5, 5),
                    0
                )

            # -------------------------------
            # Median
            # -------------------------------

            elif operation == "median":

                result = cv2.medianBlur(
                    image,
                    5
                )

            # -------------------------------
            # Bilateral
            # -------------------------------

            elif operation == "bilateral":

                result = cv2.bilateralFilter(
                    image,
                    9,
                    75,
                    75
                )

            else:

                return jsonify({
                    "error":
                    "Invalid Practical 04 operation."
                }), 400


        # ====================================================
        # PRACTICAL 05
        # IMAGE INPAINTING
        # ====================================================

        elif number == 5:

            mask = read_image(
                request.files.get("mask"),
                grayscale=True
            )

            if mask is None:

                return jsonify({
                    "error":
                    "Please select an inpainting mask."
                }), 400

            mask = cv2.resize(
                mask,
                (
                    image.shape[1],
                    image.shape[0]
                )
            )

            _, mask = cv2.threshold(
                mask,
                127,
                255,
                cv2.THRESH_BINARY
            )

            # -------------------------------
            # Telea
            # -------------------------------

            if operation == "telea":

                result = cv2.inpaint(
                    image,
                    mask,
                    3,
                    cv2.INPAINT_TELEA
                )

            # -------------------------------
            # Navier-Stokes
            # --------------
