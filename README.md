# Smile-Detector

 i have build an end-to-end computer vision and deep learning application to perform smile detection.
 Once trained, I evaluated LeNet on our testing set and found the network obtained a re- spectable 93% classification accuracy. Higher classification accuracy can be obtained by gathering more training data or applying data augmentation to existing training data.
I then created a Python script to read frames from a webcam/video file, detect faces, and then apply our pre-trained network. In order to detect faces, we used OpenCV’s Haar cascades. Once a face was detected it was extracted from the frame and then passed through LeNet to determine if the person was smiling or not smiling. As a whole, our smile detection system can easily run in real-time on the CPU using modern hardware.

To activate our detector ---"python detect_smile.py --cascade haarcascade_frontalface_default.xml --model lenet.hdf5"
