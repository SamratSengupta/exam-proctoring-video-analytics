# exam-proctoring-video-analytics
This project implements a computer vision based virtual online exam proctoring software by capturing facial recognition, head pose and eye gaze through webcam using CNN based deep learning models

## Details

This project aims to design, develop and implement a computer vision based virtual proctoring software. 
This is achieved through monitoring of head and eyeball movements, capturing head positions and gaze angles using a webcam.
It utilizes facial recognition, head pose estimation and eye gaze detection.
We have used Open Source Computer Vision Library (OpenCV with Python) for doing the project. 
OpenCV is a library of programming functions mainly aimed at real time computer vision.

Rules From the face recognition model identify the person and generate a message if he/she is an outsider 
Determine a default range beyond which signals that the person is involved in malpractices. 
Collect yaw/roll/pitch from the generated data and check the range to determine if they are looking into the camera
 Determine the gaze angle of that person and check if its beyond the acceptable default angle. 
 
 If the person is not facing the screen for 5 seconds it is a sign that they are indulging in cheating. 
 This is detected by comparing yaw/pitch/roll information between frames. 
 If it is outside the threshold and has small differences for 5 seconds then an alert is generated. 
 If the person is glancing at something other than the screen for a periods of 5 seconds.
 This is achieved through eye gaze detection.
generating Face recognition data The purpose of this code was to generate images of people in different positions in order to create a train/dev/test set.
 The program captures images in 9 directions - up, down, left, right, up-left, up-right, down-left, down-right, centre. 
 The packages/modules used were cv2, os and time.

Facial landmark detection This code detects all the faces in a given frame. We have used the “dlib” library to detect faces. 
In addition to detecting the faces, we have used the shape predictor to get the landmarks of various parts of the face. 
Facial landmarks are used to localise and represent salient regions of the face, such as eyes, eyebrows, nose, mouth and jawline.

Designing a face recognition system We used the generated images to train the facial recognition model It ensures 
that only the people who are supposed to write the online exams are the ones actually writing it.

Head Pose Estimation The purpose of this code is to determine whether the student is looking in the right direction during the online exam.
 We are trying to determine how the head is oriented with respect to the camera in order to detect if any student is involved in cheating. 
 We used the Deepgaze library for head pose estimation.

Gaze Angle Estimation We decompose the gaze angle into a subject-dependent bias term and a subject-independent difference term between the gaze angle and the bias.
 The difference term is estimated by a deep convolutional network. For calibration-free tracking, we set the subject-dependent bias term to zero.

Eye Gaze Estimation The GazeML architecture estimates eye region landmarks with a stacked-hourglass network trained on synthetic data (UnityEyes),
evaluating directly on eye images taken in unconstrained real-world settings. 
 The landmark coordinates can directly be used for model or feature-based gaze estimation.



##Installation

Installing Deepgaze

```bash
git clone https://github.com/mpatacchiola/deepgaze.git
cd deepgaze
python setup.py install
```

### Setting up virtual environment (Optional)

Run the following commands in the project directory:

**For Linux**

```bash
python3.7 -m pip install --user virtualenv
python3.7 -m venv env
source env/bin/activate
```

**For Windows**

```bash
py -m pip install --user virtualenv
py -m venv env
.\env\Scripts\activate
```

Setting up kernel for Jupyter Notebook
```bash
pip install ipykernel
ipython kernel install --user --name=env
```

### Installing required modules

```bash
pip install -r requirements.txt
```

## Usage

### Jupyter Notebook
Run the following notebook to capture images and get frame statistics
```bash
jupyter notebook main.ipynb
```
Select the kernel 'env' from Kenrel > Change kernel  
Cell > Run All to run all the cells  
Press 'q' to stop capturing  
Captured images are stored in img_cap/  
Frame statistics are stored as outputs/img_stats.json  
  
Run this notebook to check for malpractice and reconstruct the video
```bash
jupyter notebook check.ipynb
```
Cell > Run All to run all the cells
Reconstructed video is stored in output/

### Python File

Run the following commands for the same using .py files
```bash
python main.py
python check.py
```
