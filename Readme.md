# Digit Classification with Web Interface

This project implements a Convolutional Neural Network (CNN) for digit classification with an interactive web interface. Users can draw digits and get real-time predictions from the trained model.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [License](#license)

## Introduction
This project combines deep learning with web development to create an interactive digit classification system. It uses a simple CNN model trained on the MNIST dataset and provides a web interface where users can draw digits and see real-time predictions.

## Features
- Real-time digit classification using a trained CNN model
- Interactive web interface with drawing canvas
- Real-time prediction updates as you draw
- Confidence score display for predictions
- Clean and responsive user interface

## Installation
1. Clone the repository:
```bash
git clone https://github.com/your-username/CNN.Improvements.git
cd CNN.Improvements
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. First, train the model by running:
```bash
python test.py
```
This will train the CNN model and save it as 'mnist_model.h5'.

2. Start the web application:
```bash
python app.py
```

3. Open your web browser and navigate to:
```
http://localhost:5000
```

4. Use the interface:
   - Draw a digit (0-9) in the canvas
   - See real-time predictions and confidence scores
   - Use the "Clear" button to start over

## Project Structure
- `test.py`: Contains the CNN model training code
- `app.py`: Flask web application for the interactive interface
- `templates/index.html`: Web interface with drawing canvas
- `requirements.txt`: Project dependencies
- `mnist_model.h5`: Trained model file (generated after running test.py)

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.