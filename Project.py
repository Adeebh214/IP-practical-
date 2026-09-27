 app.py
from flask import Flask, render_template, request
import cv2
import numpy as np
import os
import time
import base64
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ==========================================
# GAMMA CORRECTION
# ==========================================

def gamma_correction(image, gamma):

    normalized = image / 255.0

    corrected = np.power(normalized, gamma)

    output = np.uint8(corrected * 255)

    return output


# ==========================================
# IMAGE TO BASE64
# ==========================================

def image_to_base64(image):

    success, buffer = cv2.imencode(".jpg", image)

    if not success:
        return ""

    return base64.b64encode(buffer).decode("utf-8")


# ==========================================
# HISTOGRAM GENERATION
# ==========================================

def create_histogram(image, title):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    plt.figure(figsize=(7, 4))

    plt.hist(
        gray.ravel(),
        bins=256,
        range=[0, 256]
    )

    plt.title(title)

    plt.xlabel("Pixel Intensity")

    plt.ylabel("Frequency")

    plt.xlim([0, 256])

    plt.tight_layout()

    histogram_path = os.path.join(
        UPLOAD_FOLDER,
        title.replace(" ", "_") + ".png"
    )

    plt.savefig(
        histogram_path,
        dpi=120
    )

    plt.close()

    return histogram_path


# ==========================================
# PERFORMANCE METRICS
# ==========================================

def calculate_metrics(original, enhanced):

    original_gray = cv2.cvtColor(
        original,
        cv2.COLOR_BGR2GRAY
    )

    enhanced_gray = cv2.cvtColor(
        enhanced,
        cv2.COLOR_BGR2GRAY
    )


    # Brightness

    brightness_original = np.mean(
        original_gray
    )

    brightness_enhanced = np.mean(
        enhanced_gray
    )


    # Contrast

    contrast_original = np.std(
        original_gray
    )

    contrast_enhanced = np.std(
        enhanced_gray
    )


    # MSE

    mse = np.mean(
        (
            original_gray.astype(float)
            -
            enhanced_gray.astype(float)
        ) ** 2
    )


    # PSNR

    if mse == 0:

        psnr = float("inf")

    else:

        psnr = 10 * np.log10(
            (255 ** 2) / mse
        )


    # Entropy

    histogram = cv2.calcHist(
        [enhanced_gray],
        [0],
        None,
        [256],
        [0, 256]
    )

    histogram = histogram / histogram.sum()

    entropy = -np.sum(
        histogram[histogram > 0]
        *
        np.log2(
            histogram[histogram > 0]
        )
    )


    return {

        "brightness_original":
            round(float(brightness_original), 2),

        "brightness_enhanced":
            round(float(brightness_enhanced), 2),

        "contrast_original":
            round(float(contrast_original), 2),

        "contrast_enhanced":
            round(float(contrast_enhanced), 2),

        "mse":
            round(float(mse), 2),

        "psnr":
            round(float(psnr), 2),

        "entropy":
            round(float(entropy), 2)
    }


# ==========================================
# MAIN PAGE
# ==========================================

@app.route("/", methods=["GET", "POST"])
def index():

    result = None

    if request.method == "POST":

        start_time = time.time()


        # Get uploaded file

        file = request.files.get("image")


        # Get gamma value

        gamma = float(
            request.form.get(
                "gamma",
                0.7
            )
        )


        if file and file.filename:

            filename = file.filename

            filepath = os.path.join(
                UPLOAD_FOLDER,
                filename
            )

            file.save(filepath)


            # Read original image

            original = cv2.imread(
                filepath
            )


            if original is None:

                return "Invalid image file."


            # Gamma correction

            enhanced = gamma_correction(
                original,
                gamma
            )


            # Save enhanced image

            enhanced_filename = (
                "enhanced_" + filename
            )

            enhanced_path = os.path.join(
                UPLOAD_FOLDER,
                enhanced_filename
            )

            cv2.imwrite(
                enhanced_path,
                enhanced
            )


            # Create histograms

            original_histogram = create_histogram(
                original,
                "Original Histogram"
            )

            enhanced_histogram = create_histogram(
                enhanced,
                "Enhanced Histogram"
            )


            # Metrics

            metrics = calculate_metrics(
                original,
                enhanced
            )


            # Processing time

            processing_time = (
                time.time() - start_time
            )


            # Convert images

            original_base64 = image_to_base64(
                original
            )

            enhanced_base64 = image_to_base64(
                enhanced
            )


            # Convert histogram files to base64

            with open(
                original_histogram,
                "rb"
            ) as f:

                original_hist_base64 = (
                    base64.b64encode(
                        f.read()
                    ).decode("utf-8")
                )


            with open(
                enhanced_histogram,
                "rb"
            ) as f:

                enhanced_hist_base64 = (
                    base64.b64encode(
                        f.read()
                    ).decode("utf-8")
                )


            result = {

                "original":
                    original_base64,

                "enhanced":
                    enhanced_base64,

                "original_histogram":
                    original_hist_base64,

                "enhanced_histogram":
                    enhanced_hist_base64,

                "gamma":
                    gamma,

                "processing_time":
                    round(
                        processing_time,
                        4
                    ),

                "metrics":
                    metrics
            }


    return render_template(
        "index.html",
        result=result
    )


# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    app.run(
        debug=True
    )
[27/09, 7:58 pm] Adeebh: index.html
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <title>Gamma Correction for Underwater Image Enhancement</title>

    <link rel="stylesheet"
          href="{{ url_for('static', filename='style.css') }}">
</head>

<body>

    <!-- ================= HEADER ================= -->

    <header>

        <h1>
            Gamma Correction for Underwater
            Image Enhancement
        </h1>

        <p>
            Underwater Image Enhancement using Python and OpenCV
        </p>

    </header>


    <!-- ================= NAVIGATION ================= -->

    <nav>

        <a href="#home">Home</a>
        <a href="#objective">Objective</a>
        <a href="#upload">Upload</a>
        <a href="#results">Results</a>
        <a href="#histogram">Histogram</a>
        <a href="#performance">Performance</a>
        <a href="#gamma">Gamma</a>
        <a href="#references">References</a>

    </nav>


    <main>


        <!-- ================= HOME ================= -->

        <section id="home" class="hero">

            <h2>
                Gamma Correction for Underwater
                Image Enhancement
            </h2>

            <p>
                An image-processing system for improving the
                brightness, visibility and contrast of underwater
                images using gamma correction.
            </p>


            <!-- PROCESS FLOW -->

            <div class="process-flow">

                <div class="flow-box">
                    <h3>Input</h3>
                    <p>Underwater Image</p>
                </div>

                <div class="arrow">→</div>

                <div class="flow-box">
                    <h3>Processing</h3>
                    <p>Gamma Correction</p>
                </div>

                <div class="arrow">→</div>

                <div class="flow-box">
                    <h3>Output</h3>
                    <p>Enhanced Image</p>
                </div>

            </div>

        </section>



        <!-- ================= OBJECTIVE ================= -->

        <section id="objective" class="card">

            <h2>Project Objective</h2>

            <p>
                The objective of this project is to enhance underwater
                images by improving brightness and visibility using
                gamma correction while preserving important image
                details and colors.
            </p>

            <div class="three-columns">

                <div class="info-box">
                    <h3>Input</h3>
                    <p>
                        Underwater image with low brightness,
                        poor visibility or low contrast.
                    </p>
                </div>

                <div class="info-box">
                    <h3>Processing</h3>
                    <p>
                        Image normalization followed by
                        gamma correction.
                    </p>
                </div>

                <div class="info-box">
                    <h3>Output</h3>
                    <p>
                        Enhanced underwater image with
                        improved brightness and visibility.
                    </p>
                </div>

            </div>

        </section>



        <!-- ================= UPLOAD ================= -->

        <section id="upload" class="card">

            <h2>Upload Underwater Image</h2>

            <form method="POST"
                  enctype="multipart/form-data">

                <div class="upload-row">

                    <div>

                        <label>
                            Select Image
                        </label>

                        <input
                            type="file"
                            name="image"
                            accept=".jpg,.jpeg,.png,.webp"
                            required
                        >

                    </div>


                    <div>

                        <label>
                            Gamma Value
                        </label>

                        <input
                            type="number"
                            name="gamma"
                            value="0.7"
                            min="0.1"
                            max="3"
                            step="0.1"
                            required
                        >

                    </div>


                    <div>

                        <button type="submit">
                            Enhance Image
                        </button>

                    </div>

                </div>

            </form>

        </section>



        <!-- ================= RESULTS ================= -->

        {% if result %}

        <section id="results" class="card">

            <h2>Image Processing Results</h2>

            <div class="image-results">


                <!-- ORIGINAL -->

                <div class="image-box">

                    <h3>Original Image</h3>

                    <img
                        src="data:image/jpeg;base64,{{ result.original }}"
                        alt="Original Underwater Image"
                    >

                </div>


                <!-- ENHANCED -->

                <div class="image-box">

                    <h3>Gamma Corrected Image</h3>

                    <img
                        src="data:image/jpeg;base64,{{ result.enhanced }}"
                        alt="Gamma Corrected Underwater Image"
                    >

                </div>

            </div>


            <div class="gamma-display">

                <strong>
                    Gamma Value Used:
                </strong>

                {{ result.gamma }}

            </div>

        </section>



        <!-- ================= HISTOGRAM ================= -->

        <section id="histogram" class="card">

            <h2>Histogram Analysis</h2>

            <p>
                The histogram represents the distribution of
                pixel intensity values from 0 to 255 before and
                after gamma correction.
            </p>

            <div class="histogram-area">

                <div class="histogram-box">

                    <h3>Original Image Histogram</h3>

                    <img
                 src="data:image/png;base64,{{ result.original_histogram }}"
                 alt="Original Image Histogram"
                 class="histogram-image"
>

                </div>


                <div class="histogram-box">

                    <h3>Enhanced Image Histogram</h3>

                    <img
                 src="data:image/png;base64,{{ result.enhanced_histogram }}"
                 alt="Enhanced Image Histogram"
                 class="histogram-image"
>

                </div>

            </div>

        </section>



        <!-- ================= PERFORMANCE ================= -->

        <section id="performance" class="card">

            <h2>Performance Metrics</h2>


            <div class="metrics">


                <div class="metric">

                    <h3>Gamma Value</h3>

                    <p>
                        {{ result.gamma }}
                    </p>

                </div>


                <div class="metric">

                    <h3>Original Brightness</h3>

                    <p>
                        {{ result.metrics.brightness_original }}
                    </p>

                </div>


                <div class="metric">

                    <h3>Enhanced Brightness</h3>

                    <p>
                        {{ result.metrics.brightness_enhanced }}
                    </p>

                </div>


                <div class="metric">

                    <h3>Original Contrast</h3>

                    <p>
                        {{ result.metrics.contrast_original }}
                    </p>

                </div>


                <div class="metric">

                    <h3>Enhanced Contrast</h3>

                    <p>
                        {{ result.metrics.contrast_enhanced }}
                    </p>

                </div>


                <div class="metric">

                    <h3>Entropy</h3>

                    <p>
                        {{ result.metrics.entropy }}
                    </p>

                </div>


                <div class="metric">

                    <h3>MSE</h3>

                    <p>
                        {{ result.metrics.mse }}
                    </p>

                </div>


                <div class="metric">

                    <h3>PSNR</h3>

                    <p>
                        {{ result.metrics.psnr }} dB
                    </p>

                </div>


                <div class="metric">

                    <h3>Processing Time</h3>

                    <p>
                        {{ result.processing_time }} sec
                    </p>

                </div>


            </div>

        </section>



        <!-- ================= GAMMA ANALYSIS ================= -->

        <section id="gamma" class="card">

            <h2>Gamma Value Analysis</h2>

            <table>

                <thead>

                    <tr>

                        <th>Gamma Value</th>

                        <th>Expected Effect</th>

                    </tr>

                </thead>

                <tbody>

                    <tr>

                        <td>0.5</td>

                        <td>
                            Strong brightness enhancement
                        </td>

                    </tr>

                    <tr>

                        <td>0.7</td>

                        <td>
                            Moderate brightness enhancement
                        </td>

                    </tr>

                    <tr>

                        <td>0.9</td>

                        <td>
                            Slight brightness enhancement
                        </td>

                    </tr>

                    <tr>

                        <td>1.0</td>

                        <td>
                            Original intensity
                        </td>

                    </tr>

                    <tr>

                        <td>1.3</td>

                        <td>
                            Image becomes darker
                        </td>

                    </tr>

                </tbody>

            </table>

        </section>



        <!-- ================= FORMULAS ================= -->

        <section class="card">

            <h2>Mathematical Formulas</h2>

            <div class="formula">

                I<sub>out</sub>
                =
                c(I<sub>in</sub>)<sup>γ</sup>

            </div>


            <div class="formula">

                I<sub>norm</sub>
                =
                I / 255

            </div>


            <div class="formula">

                MSE
                =
                (1/N) Σ(I - I')²

            </div>


            <div class="formula">

                PSNR
                =
                10 log<sub>10</sub>(255² / MSE)

            </div>


            <div class="formula">

                H
                =
                -Σ p(i) log<sub>2</sub> p(i)

            </div>

        </section>



        <!-- ================= TECHNOLOGIES ================= -->

        <section class="card">

            <h2>Technologies Used</h2>

            <div class="technology-grid">

                <div class="tech">
                    <h3>Python</h3>
                    <p>Main programming language</p>
                </div>

                <div class="tech">
                    <h3>OpenCV</h3>
                    <p>Image processing</p>
                </div>

                <div class="tech">
                    <h3>NumPy</h3>
                    <p>Numerical operations</p>
                </div>

                <div class="tech">
                    <h3>Matplotlib</h3>
                    <p>Histogram analysis</p>
                </div>

                <div class="tech">
                    <h3>Flask</h3>
                    <p>Web application framework</p>
                </div>

                <div class="tech">
                    <h3>HTML & CSS</h3>
                    <p>Website design</p>
                </div>

            </div>

        </section>



        <!-- ================= REFERENCES ================= -->

        <section id="references" class="card">

            <h2>References</h2>

            <ol>

                <li>
                    OpenCV Documentation –
                    Image Processing
                </li>

                <li>
                    Gonzalez and Woods –
                    Digital Image Processing
                </li>

                <li>
                    Research literature on underwater
                    image enhancement and gamma correction
                <
