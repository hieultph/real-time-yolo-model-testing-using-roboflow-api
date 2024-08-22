# Real-time YOLO Model Testing using Roboflow API (Rock Paper Scissors)

## Overview

This project demonstrates real-time testing on a local machine using a YOLO model trained with [Roboflow](https://roboflow.com/). The project is part of my AI training course. For an in-depth explanation of the training process, please watch my detailed video:

<iframe width="560" height="315" src="https://www.youtube.com/embed/bAFI27Tpm3E?si=XqwGcEY5iIq_e4Kv" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

## Getting Started

Follow these steps to get the project up and running:

1. **Clone the repository:**
    ```sh
    git clone <repository-url>
    ```
2. **Install the required libraries:**
   Make sure you have all the necessary dependencies installed. You can find the list of required libraries in the `requirements.txt` file. Install them using pip:
   ```sh
   pip install -r requirements.txt
   ```
3. **Update Roboflow API Key and Model**
   Edit the `roboflow_config.json` file to replace `"ROBOFLOW_MODEL"` with your model version and `"ROBOFLOW_API_KEY"` with your actual API key obtained from Roboflow.
   
3. **Run the Testing Script:**
   Start testing by running either the `infer-simple.py` or `infer-async.py` script:
   ```sh
   python infer-simple.py
   # or
   python infer-async.py
   ```

**Note:** If you prefer running the YOLO model locally, download the appropriate YOLO version from GitHub, install the necessary dependencies, and refer to the `for-local-use.py` script for local testing.

## Dataset

The dataset used in this project is sourced from [Roboflow Universe](https://universe.roboflow.com/roboflow-58fyf/rock-paper-scissors-sxsw). It contains a collection of labeled images specifically designed for training models to recognize hand gestures for rock, paper, and scissors. This dataset serves as the foundation for developing and testing the YOLO model in this project.

## Contributing

Contributions are welcome! If you have suggestions for improvements or find any issues, please open an issue or submit a pull request.

## Like This Project?

If you found this project helpful, please consider giving it a star! Follow me for more interesting projects.
