# Orion's Vision
Unlock the potential of real-time tracking with Orion's Vision. Our innovative algorithm is redefining the efficiency and adaptability of monitoring systems.

## Table of Contents
- [Introduction](#introduction)
- [Algorithm Overview](#algorithm-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Demo](#demo)
- [Technology Stack](#technology-stack)
- [License](#license)
- [Contributing](#contributing)
- [Credits and Acknowledgements](#credits-and-acknowledgements)
- [Contact Information](#contact-information)

## Introduction
Orion's Vision presents a pioneering approach to real-time processing for tracking applications. Our algorithm significantly enhances processing speeds on CPUs by using individuals' coordinates and image depth. It minimizes the reliance on neural networks by utilizing them periodically, not constantly, leading to a more efficient use of computational resources.

## Algorithm Overview
Orion's Vision integrates a cutting-edge algorithm that leverages individual coordinates and image depth to dramatically enhance real-time processing speed on CPUs, reducing the dependency on neural networks by utilizing them only periodically. This method stands out for its robust ability to compensate for any network's limitations, independent of its base accuracy.

It is designed with versatility in mind, able to efficiently track a wide array of objects or individuals across diverse monitoring scenarios with minimal reliance on neural networks. A standout feature of Orion's Vision is its proficiency in not only tracking known objects but also in accurately counting 'Unknown' entities, enhancing its utility in environments where not all objects may be identifiable or registered.

Moreover, we believe in transparency and the advancement of knowledge. Therefore, while we ensure the core of our sophisticated algorithm remains proprietary, a portion of Orion's Vision's algorithm is available for public review, reflecting our commitment to contributing to the community while also safeguarding our innovative solutions.

![Orion's Vision Algorithm Scheme](./assets/orions_vision_algorithm.jpg)

## Features
- **Intelligent Frame Skipping**: Dynamically adjusts the frame rate, offering a balanced trade-off between speed and accuracy, thereby optimizing computational resources.
- **Adaptive Thresholding**: Introduces a flexible threshold parameter that adapts to detection requirements, ensuring reliable tracking even in varying conditions.
- **Optimized Frame Capture**: Tailors the frame-per-second rate for seamless integration with the processing capabilities of diverse systems.
Streamlined Model Management: Simplifies the use of advanced detection models by enforcing a standardized placement within the ./models directory, facilitating ease of access and consistency.
- **Robust Database Handling**: Employs an intelligent database management system that maintains a cap on the number of trackable entities, ensuring database integrity without compromising on the system’s performance.
- **Cross-Model Compatibility**: Designed to function with any detector or tracking model that uses distance as a similarity metric, ensuring broad application potential and ease of integration.
- **Minimal Neural Dependence**: Achieves significant efficiency by reducing the frequency of neural network computations, favoring a periodic over a constant approach.

## Installation
To install Orion's Vision, follow the steps below:

1. Clone the repository:
```bash
git clone https://github.com/Amir-Nassimi/Orions-Vision.git
cd Orions-Vision
```

2. Install the required dependencies:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

3. Create ./models directory. All the weights should be placed in this directory; yet only the names of them will be presented to the algorithm!

## Usage
To use Orion's Vision for tracking, run the following command:
```bash
python run.py --skip_frame_rate 0 --threshold 45 --fps 30 --filepath "/path/to/your/video/or/rtsp" --detector_model_name "yolov8x.pt" --thresh_on_estimate 500 --tracking_model_name "model.pth.tar-80" --no_db_limit 15
```
---
- ***skip_frame_rate***: (Optional) Number of frames to skip for speedup.
- ***threshold***: (Optional) Threshold for detection accuracy.
- ***fps***: (Optional) The frame capture rate per second.
- ***filepath***: (Required) The file path to the video or the RTSP address of the camera.
- ***detector_model_name***: (Optional) Name of the YOLO model weights file. Must be located in the ./models directory.
- ***thresh_on_estimate***: (Optional) Threshold on estimate for updating the database based on the tracking algorithm.
- ***tracking_model_name***: (Optional) Name of the tracking model file. Must be located in the ./models directory.
- ***no_db_limit***: (Optional) The limit after which older database entries are discarded in favor of new ones.


## Demo
Get a glimpse of Orion's Vision in action with our demo. This visual demonstration highlights the application's real-time gesture recognition capabilities. You can find it in ./assets directory.

![Orion's Vision Demo](./assets/orions_vision_demo.mp4)

## Technology Stack
Orion's Vision is built using:
- **OpenCV (cv2)**: For real-time image processing and computer vision tasks.
- **PyTorch**: Serving as the backbone for deep learning operations.
- **torchreid**: A library specialized in deep learning-based person re-identification.
- **Numpy**: For efficient numerical computations on large data arrays.
- **Python Imaging Library (PIL)**: Used for opening, manipulating, and saving many different image file formats.
- **Threading**: To improve the performance of I/O-bound operations by using threads.
- **Transforms from torchvision**: For preprocessing and normalizing image data for neural network inputs.
- **Glob**: To retrieve files/pathnames matching a specified pattern.
- **Math**: Providing access to the mathematical functions defined by the C standard.
- **Datetime**: For manipulating dates and times in both simple and complex ways.
- **Dist (hypot from math)**: To calculate the Euclidean distance between points in a fast and reliable way.

## License
Orion's Vision is open-sourced under the MIT License. See [LICENSE](LICENSE) for more details.

## Contributing
While we deeply value community input and interest in Orion's Vision, the project is currently in a phase where we're mapping out our next steps and are not accepting contributions just yet. We are incredibly grateful for your support and understanding. Please stay tuned for future updates when we'll be ready to welcome contributions with open arms.

## Credits and Acknowledgements
We would like to extend our heartfelt thanks to Mr.Mohammad Fadaeian for his guidance and wisdom throughout the development of Orion's Vision. His insights have been a beacon of inspiration for this project.

## Contact Information
Although we're not open to contributions at the moment, your feedback and support are always welcome. Please feel free to star the project or share your thoughts through the Issues tab on GitHub, and we promise to consider them carefully.please [open an issue](https://github.com/Amir-Nassimi/Orions-Vision/issues) in the Orion's Vision repository, and we will assist you.