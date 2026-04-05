#!/usr/bin/env python3

import os, json, queue, time, pickle
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision.models import resnet18
import matplotlib.pyplot as plt
import pandas as pd
import pyttsx3

# ROS2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

# VOICE INPUT
import sounddevice as sd
from vosk import Model, KaldiRecognizer


MODEL_PATH = "/home/harish/Downloads/resnet_knee_model.pth"
DATA_PATH = "/home/harish/Downloads/dataset for hackathon"
CSV_PATH = "/home/harish/Downloads/dataset for hackathon/metadata.csv"
VOSK_MODEL_PATH = "vosk-model-small-en-us-0.15"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


REGION_DESCRIPTIONS = {
    "left":  "the medial side of the knee, focusing on the inner ligament bundle",
    "right": "the lateral side of the knee, focusing on the outer ligament bundle",
    "knee":  "the central knee region, focusing on the mid-substance of the ACL"
}


engine = pyttsx3.init()
engine.setProperty('rate', 135)

def speak(text):
    print("SYSTEM:", text)
    engine.say(text)
    engine.runAndWait()

q = queue.Queue()

def callback(indata, frames, time_, status):
    q.put(bytes(indata))

vosk_model = Model(VOSK_MODEL_PATH)
recognizer = KaldiRecognizer(vosk_model, 16000)

def listen_voice():
    print("🎤 Listening...")
    with sd.RawInputStream(samplerate=16000, blocksize=8000,
                           dtype='int16', channels=1, callback=callback):
        while True:
            data = q.get()
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "")
                print("You said:", text)
                return text.lower()


# =========================
def interpret_command(text):
    t = text.lower().strip()

    POSITIVE_PHRASES = [
        "go for it", "go ahead", "go for", "sounds good", "make it so",
        "let's go", "lets go", "move there", "that's right", "thats right",
        "do it", "take it", "i'm ready", "im ready"
    ]
    WAIT_PHRASES = [
        "hang on", "hold on", "not yet", "one moment", "just a moment"
    ]
    RESUME_PHRASES = [
        "go on", "carry on", "let's continue", "lets continue", "i'm ready", "im ready"
    ]
    STOP_PHRASES = [
        "that's enough", "thats enough", "shut down"
    ]

    for p in POSITIVE_PHRASES:
        if p in t:
            return "positive"
    for p in WAIT_PHRASES:
        if p in t:
            return "wait"
    for p in RESUME_PHRASES:
        if p in t:
            return "resume"
    for p in STOP_PHRASES:
        if p in t:
            return "negative"

    if any(w in t for w in [
        "next", "necks", "text", "skip", "after", "forward", "onwards"
    ]):
        return "next"

    if any(w in t for w in [
        "left", "lef", "lift", "loft", "medial", "inner", "inside", "inward"
    ]):
        return "left"

    if any(w in t for w in [
        "right", "rite", "write", "reit", "lateral", "outer", "outside", "outward"
    ]):
        return "right"

    if any(w in t for w in [
        "knee", "nee", "need", "ni", "center", "central", "middle", "mid", "core", "acl"
    ]):
        return "knee"

    if any(w in t for w in [
        "go", "goe", "doe", "dough",
        "confirm", "confirmed", "yes", "yeah", "yep", "yup",
        "okay", "ok", "sure", "correct", "proceed", "execute",
        "approved", "affirmative", "accept", "perfect", "positive"
    ]):
        return "positive"

    if any(w in t for w in [
        "stop", "stopped", "no", "nope", "nah", "negative",
        "cancel", "abort", "quit", "exit", "end",
        "done", "finish", "finished", "halt", "terminate"
    ]):
        return "negative"

    if any(w in t for w in [
        "wait", "weight", "wade", "late",
        "pause", "hold", "standby", "freeze", "stay"
    ]):
        return "wait"

    if any(w in t for w in [
        "resume", "rezoom", "result", "continue", "start",
        "unpause", "back", "ready"
    ]):
        return "resume"

    return "unknown"


class FrankaStaticPositionNode(Node):
    def __init__(self, mode):
        super().__init__('franka_node')
        self.publisher_ = self.create_publisher(JointState, '/joint_command', 10)

        self.joint_names = [
            "panda_joint1","panda_joint2","panda_joint3",
            "panda_joint4","panda_joint5","panda_joint6",
            "panda_joint7","panda_finger_joint1","panda_finger_joint2"
        ]

        self.poses = {
            "left":  [0.5, -1.16, 0, -2.3, 0, 1.6, 1.1, 0.04, 0.04],
            "right": [-0.5, -1.16, 0, -2.3, 0, 1.6, 1.1, 0.04, 0.04],
            "knee":  [0, -0.5, 0, -2.0, 0, 2.5, 0.8, 0.04, 0.04]
        }

        self.mode = mode

    def send(self):
        time.sleep(1)
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = self.poses[self.mode]
        self.publisher_.publish(msg)

def move_robot(mode):
    rclpy.init()
    node = FrankaStaticPositionNode(mode)
    node.send()
    rclpy.spin_once(node, timeout_sec=1)
    node.destroy_node()
    rclpy.shutdown()



# =========================
# MODEL
# =========================
model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 3)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE).eval()

df = pd.read_csv(CSV_PATH)
labels = {0: "Healthy", 1: "Partial Tear", 2: "Complete Rupture"}

# =========================
# PREPROCESS
# =========================
def preprocess(vol):
    img = vol[len(vol) // 2]
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = cv2.resize(img, (224, 224))
    tensor = torch.tensor(np.stack([img] * 3), dtype=torch.float32).unsqueeze(0).to(DEVICE)
    return img, tensor

# =========================
# GRADCAM
# =========================
features, grads = [], []

def f_hook(m, i, o): features.append(o)
def b_hook(m, gi, go): grads.append(go[0])

model.layer4[1].conv2.register_forward_hook(f_hook)
model.layer4[1].conv2.register_backward_hook(b_hook)

def gradcam(tensor):
    features.clear(); grads.clear()
    out = model(tensor)
    pred = out.argmax()
    out[0, pred].backward()

    g = grads[0][0].cpu().detach().numpy()
    f = features[0][0].cpu().detach().numpy()

    w = np.mean(g, axis=(1, 2))
    cam = np.zeros(f.shape[1:])

    for i, val in enumerate(w):
        cam += val * f[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    return cam / (cam.max() + 1e-8)

# =========================
# LOCATION
# =========================
def get_loc(cam):
    x = np.argmax(np.sum(cam, axis=0))
    w = cam.shape[1]
    if x < w // 3:
        return "left"
    elif x > 2 * w // 3:
        return "right"
    return "knee"

_sample_pool = []
_used_samples = []

def get_sample():
    global _sample_pool, _used_samples

    # Build pool once
    if not _sample_pool and not _used_samples:
        print("Scanning dataset for abnormal cases...")
        file_index = {}
        for root, _, files in os.walk(DATA_PATH):
            for f in files:
                file_index[f] = os.path.join(root, f)

        for _, r in df.iterrows():
            if r["aclDiagnosis"] != 0:
                fname = r["volumeFilename"]
                if fname in file_index:
                    _sample_pool.append(file_index[fname])

        import random
        random.shuffle(_sample_pool)
        print(f"Found {len(_sample_pool)} abnormal cases.")

    # Refill if all used
    if not _sample_pool:
        import random
        print("All cases reviewed — reshuffling pool.")
        _sample_pool = _used_samples[:]
        _used_samples.clear()
        random.shuffle(_sample_pool)

    chosen = _sample_pool.pop()
    _used_samples.append(chosen)
    print(f"Selected: {os.path.basename(chosen)}")
    return chosen

# =========================
# PRINT COMMAND REFERENCE (shown at startup)
# =========================
def print_command_reference():
    print("\n" + "="*65)
    print("  ACL ROBOT VOICE COMMAND REFERENCE")
    print("="*65)
    print(f"  {'COMMAND':<20} {'ACTION / REGION'}")
    print("-"*65)
    print(f"  {'\"left\"':<20} Move to medial (inner) side of the knee")
    print(f"  {'\"right\"':<20} Move to lateral (outer) side of the knee")
    print(f"  {'\"knee\" / \"center\"':<20} Move to central ACL mid-substance region")
    print(f"  {'\"go\" / \"move\"':<20} Move robot to the AI-suggested region")
    print(f"  {'\"next\"':<20} Skip to next unvisited region")
    print(f"  {'\"wait\" / \"pause\"':<20} Enter standby mode (hold position)")
    print(f"  {'\"resume\"':<20} Exit standby and continue")
    print(f"  {'\"stop\" / \"no\"':<20} Shut down system safely")
    print("="*65 + "\n")

# =========================
# MAIN LOOP
# =========================
def run():
    print_command_reference()

    while True:
        file = get_sample()

        with open(file, "rb") as f:
            vol = pickle.load(f)

        img, tensor = preprocess(vol)
        pred = torch.argmax(model(tensor), dim=1).item()

        if pred == 0:
            speak("This scan appears healthy. Moving to next case.")
            continue

        cam = gradcam(tensor)
        loc = get_loc(cam)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle(os.path.basename(file), fontsize=11)

        axes[0].imshow(img, cmap='gray')
        axes[0].set_title("MRI Scan")
        axes[0].axis('off')

        axes[1].imshow(img, cmap='gray')
        axes[1].imshow(cam, cmap='jet', alpha=0.45)
        axes[1].set_title("GradCAM Overlay")
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()

        diagnosis = labels[pred]
        speak(f"Diagnosis: {diagnosis} detected in the ACL.")
        speak(f"The AI suggests examining {REGION_DESCRIPTIONS[loc]}.")

        visited = set()
        waiting_mode = False
        done = False

        while not done:

            # WAIT / STANDBY MODE — silent loop, no repeated speech
            if waiting_mode:
                print("💤 Standby...")
                text = listen_voice()
                if not text.strip():
                    continue
                cmd = interpret_command(text)
                if cmd == "resume":
                    speak("Resuming.")
                    waiting_mode = False
                continue

            # NORMAL COMMAND LOOP
            print("🎤 Ready.")
            text = listen_voice()
            if not text.strip():
                continue

            command = interpret_command(text)
            print("COMMAND:", command)

            if command == "unknown":
                continue

            elif command == "wait":
                speak("Standby.")
                waiting_mode = True

            elif command == "positive":
                speak(f"Moving to {REGION_DESCRIPTIONS[loc]}.")
                move_robot(loc)
                visited.add(loc)

            elif command in ["left", "right", "knee"]:
                speak(f"Moving to {command} — {REGION_DESCRIPTIONS[command]}.")
                move_robot(command)
                visited.add(command)
                loc = command

            elif command == "negative":
                speak("Shutting down. Goodbye.")
                return  # exit entirely

            elif command == "next":
                options = ["left", "right", "knee"]
                remaining = [o for o in options if o not in visited]

                if not remaining:
                    speak("All regions covered. Loading next case.")
                    done = True
                else:
                    loc = remaining[0]
                    speak(f"Next region: {loc} — {REGION_DESCRIPTIONS[loc]}.")

# =========================
run()
