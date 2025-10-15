**AI-Powered Real-Time Weapon Detection & Alert System**

This project applies AI and computer vision to build a real-time weapon detection and alert system.
It uses YOLOv8 for object detection, Streamlit for live visualization, and Firebase for secure user authentication — showing how machine learning models can be deployed to solve safety-critical problems.

**Overview**

The system continuously monitors live webcam footage, detects weapons such as guns or knives, and automatically sends alert emails to the logged-in user with visual evidence.
It also logs all detections and displays a clean dashboard for analytics and review.

**Applied AI Skills**

Object Detection (YOLOv8): Integrated and fine-tuned YOLOv8 for high-confidence real-time detection.

Computer Vision Pipeline: Built a real-time OpenCV–Streamlit inference loop with noise filtering and dynamic thresholding.

Automation & Alerting: Designed an automated email alert workflow attaching frame and cropped detection images.

Secure Cloud Integration: Used Firebase Authentication for user-level access and logging.

Data Handling & Visualization: Implemented detection logs, data storage, and a Streamlit-based analytics dashboard.

**Tech Stack**
| Component                   | Technology                | Purpose             |
| --------------------------- | ------------------------- | ------------------- |
| **Detection Model**         | YOLOv8 (Ultralytics)      | Object detection    |
| **Frame Pipeline**          | OpenCV + Streamlit WebRTC | Real-time video     |
| **Auth & Users**            | Firebase (Pyrebase)       | Secure login/signup |
| **Alert Automation**        | SMTP (App Password)       | Email notifications |
| **Logging & Visualization** | Pandas + Streamlit        | Data analytics      |
| **Environment Config**      | dotenv                    | Config management   |

**Outputs**
1. Real-Time Detection
<img width="284" height="144" alt="image" src="https://github.com/user-attachments/assets/596e1f5a-a016-4a76-a33e-6c3bcf622ed8" />

2. Automated Alert Email
<img width="126" height="881" alt="image" src="https://github.com/user-attachments/assets/f757f4a1-bb37-40be-ad43-e8722abf523c" />

Streamlit Dashboard
<img width="284" height="144" alt="image" src="https://github.com/user-attachments/assets/a5e798e3-c09d-4e3e-90fb-647f374e954a" />

**AI in Practice**

This project demonstrates how machine learning can move beyond research into real-world automation.
It shows how computer vision can enhance situational awareness, safety, and decision-making through:

Multi-camera or edge deployment for smart surveillance

Cloud-based detection logging and monitoring

Real-time alerts for proactive safety response


Author
Sanjana S.
AI Engineer | Computer Vision | Applied Machine Learning
Building real-world AI systems that bridge perception, automation, and safety.
