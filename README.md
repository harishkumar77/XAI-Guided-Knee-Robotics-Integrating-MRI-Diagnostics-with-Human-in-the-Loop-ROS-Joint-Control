# XAI Guided Knee Robotics Integrating MRI Diagnostics with Human in the Loop ROS Joint Control

Multi-Modal Surgical Robotics: XAI-Guided Knee Diagnostics & ROS Joint Control
Overview

This project, developed for ScarletHacks 2026, implements a closed-loop system for automated knee surgery assistance. By integrating Explainable AI (XAI) with Human-in-the-Loop (HITL) robotics, the system analyzes MRI scans, localizes ACL tears, and orchestrates robotic camera positioning via voice commands within NVIDIA Isaac Sim.

Core Features

1. Computer Vision: ResNet18-based classification of ACL healthy vs. partial/complete tears.
2. Explainable AI (XAI): Grad-CAM heatmaps localize the "Region of Interest" (ROI) to justify diagnostic decisions.
3. Robotics: ROS2 node commanding a Franka Panda manipulator via JointState messages for specific anatomical viewing angles.
4. HRI (Human-Robot Interaction): Voice-activated command interpreter using Vosk for supervisor-led robot navigation (Wait, Resume, Stop, and Directional overrides).
5. Tech StackML Framework: PyTorch (Torchvision ResNet18).
6. Robotics: ROS2 (Humble/Foxy), Joint Trajectory Control.
7. Simulation: NVIDIA Isaac Sim.
8. Voice Processing: Vosk (Speech-to-Text),
9. Pyttsx3 (Text-to-Speech).Logic:
10.  Python 3.x, NumPy, OpenCV, Matplotlib.

System Architecture

1. Diagnostic Phase: The system loads a random abnormal MRI volume and classifies the ACL status.
2. Localization Phase: Grad-CAM generates a heatmap. The system calculates the horizontal center of mass to suggest a viewing zone: Medial (Left), Lateral (Right), or Central (Knee).
3. Command Phase: The system speaks the diagnosis and suggested robotic position. It then enters a listening state for surgeon confirmation.Execution Phase: Upon a "Positive" voice command (e.g., "Go for it"), a ROS2 node publishes target joint configurations to Isaac Sim.📋

System Architecture & Operational Workflow
The following flowchart illustrates the End-to-End Multimodal Pipeline of the project. The system is designed to bridge the gap between Deep Learning-based Diagnostics and Robotic Actuation through a high-fidelity simulation in NVIDIA Isaac Sim.

Key Workflow Stages:
1. Autonomous Perception: The pipeline begins with a ResNet18 classifier analyzing the MRI scan to detect healthy tissue, partial tears, or complete ruptures.
2. Explainable AI (XAI) Localization: If a pathology is detected, Grad-CAM generates a spatial heatmap to localize the injury site. This provides the "visual justification" required for clinical decision support.
3. Human-in-the-Loop (HITL) Supervision: To ensure surgical safety, the robot does not move autonomously. It suggests a viewing angle (Left, Right, or Center) and waits for a Voice Command via the Vosk interpreter.
4. Joint-Space Actuation: Upon receiving a "Positive" voice trigger (e.g., "Go for it"), the ROS 2 node translates the diagnostic intent into specific JointState configurations for the Franka Emika Panda manipulator.
5. Dynamic Control States: The system supports real-time interrupts, including Wait, Resume, and Manual Override commands, allowing the surgeon to maintain full authority over the robotic hardware.

<img width="1680" height="638" alt="Gemini_Generated_Image_nl9rbwnl9rbwnl9r" src="https://github.com/user-attachments/assets/1b98911e-d07b-4792-a77d-eca4adf4e6f5" />
