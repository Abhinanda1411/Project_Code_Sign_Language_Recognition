# Project_Code_Sign_Language_Recognition
This project presents a real-time system that recognises
American Sign Language (ASL) fingerspelling from a laptop
webcam and converts it into text and speech. Hand landmarks
are extracted with MediaPipe and classified by machine
learning models trained on a custom dataset of 8,580
samples, recorded across six sessions with both hands. Static
letters and the moving letters J and Z are handled by separate
models. On an unseen test session, the best model (SVM)
reached 93.5% accuracy on static letters and 98.3% on J, Z
and non-gesture movements. NLP-assisted word prediction
reduces typing effort, and text-to-speech outputs the
sentence. The system runs as a web application on a standard
laptop.
