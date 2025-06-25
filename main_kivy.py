import cv2
import numpy as np
import yaml
from kivy.app import App
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import onnxruntime as ort

"""
This is the Kivy prototype used to test the real-conditions performance of model's inference
"""

class YOLOCameraApp(App):
    def build(self):
        self.capture = cv2.VideoCapture(0)
        self.image_widget = Image()
        Clock.schedule_interval(self.update, 1.0 / 30.0)

        # Load ONNX model
        model_path = "/runs/detect/train38/weights/best.onnx"
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

        # Load class names from data.yaml
        yaml_path = "dataset_old_archive/old_dataset_attempts/wikipedia_dataset/data.yaml"
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            raw_names = data['names']
            if isinstance(raw_names, dict):
                self.class_names = [raw_names[i] for i in sorted(raw_names)]
            else:
                self.class_names = raw_names

        return self.image_widget

    def update(self, dt):
        ret, frame = self.capture.read()
        if not ret:
            return

        # Flip both vertically and horizontally (mirror + upside-down)
        frame = cv2.flip(frame, -1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_image = cv2.resize(rgb_frame, (736, 736))

        # Normalize and prepare input
        img = input_image.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        img = np.expand_dims(img, axis=0)  # Add batch dim

        # Run inference
        outputs = self.session.run(None, {self.input_name: img})
        predictions = outputs[0]

        if predictions.ndim == 3:
            predictions = predictions[0]

        for pred in predictions:
            try:
                x1, y1, x2, y2, conf, class_id = [float(p.item()) if hasattr(p, 'item') else float(p) for p in pred[:6]]
            except Exception as e:
                print("Skipping prediction due to conversion error:", e)
                continue

            if conf < 0.5:
                continue

            # Rescale coordinates to original frame
            x1 = int(x1 / 736 * rgb_frame.shape[1])
            y1 = int(y1 / 736 * rgb_frame.shape[0])
            x2 = int(x2 / 736 * rgb_frame.shape[1])
            y2 = int(y2 / 736 * rgb_frame.shape[0])

            class_id = int(class_id)
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"id_{class_id}"
            label = f"{class_name} {conf:.2f}"

            # Log detection to console
            print(f"[DETECTED] {label} at ({x1}, {y1}), ({x2}, {y2})")

            # Draw detection on frame
            cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(rgb_frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Show frame in Kivy UI
        buf = rgb_frame.tobytes()
        texture = Texture.create(size=(rgb_frame.shape[1], rgb_frame.shape[0]), colorfmt='rgb')
        texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
        self.image_widget.texture = texture

    def on_stop(self):
        self.capture.release()


if __name__ == '__main__':
    YOLOCameraApp().run()
