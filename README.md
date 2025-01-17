# ResSleepNet

The code supporting the findings of this study is available for non-commercial academic purposes, subject to a formal code usage agreement. For access, please contact the corresponding author (hongnju@njust.edu.cn).

## System Requirements

- **Operating System**: Windows 10/11, Ubuntu 20.04, MacOS
- **Python Version**: 3.8
- **Software Dependencies**:
  - `keras==2.9.0`
  - `matplotlib==3.7.5`
  - `numpy==1.24.4`
  - `scipy==1.15.1`
  - `tensorflow_gpu==2.9.0`
  - `importlib-metadata==7.1.0`


Make sure all required dependencies are installed using the following steps.

## Installation Guide

### Step 1: Clone the Repository

Clone the repository to your local machine using:

```bash
git clone https://github.com/zhuangzx1127/ResSleepNet.git
cd ResSleepNet
```

### Step 2: Install Dependencies

It is highly recommended to create a virtual environment for this project. You can do this by running:

```bash
python -m venv venv
source venv/bin/activate  # For Linux/macOS
venv\Scripts\activate  # For Windows
```

Then, install the required dependencies:

```bash
pip install -r requirements.txt
```

This will install the necessary libraries including Keras, Matplotlib, Numpy, SciPy, and TensorFlow.

## Demo

### Step 1: Run the Demo Script

After installing all dependencies, you can run the demo with the following command:

```bash
python test.py
```

### Expected Output

After running the demo, you should see the following output:
- **Predicted AHI Value**:  
  ```bash
  predicted AHI: X.XX
  ```

- **Sleep Stage Plot**: A plot will be generated showing the predicted sleep stages and their probability distribution. The top part of the plot represents the predicted sleep stages (REM, Deep, Light, Wake), while the bottom part shows the probability distribution for each stage across epochs.

### Expected Run Time

The demo script typically runs in **5 seconds** on a "normal" desktop computer.

## Instructions for Use

1. Clone the repository as described above.
2. Install the dependencies using the command: `pip install -r requirements.txt`.
3. Run the script `test.py` to start the demonstration or to interact with the main functions of the code.

## Optional Reproduction Instructions

If you want to reproduce all quantitative results in the manuscript, follow these additional steps:

1. Make sure that you have a compatible GPU setup for TensorFlow.
2. Modify any parameters as needed in `test.py` for your data or to fit specific experimental conditions.
