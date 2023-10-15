# Face detection and encoding.
from PIL import Image
import numpy as np
import cv2

from annoy import AnnoyIndex
from typing import Tuple, Dict, List
from keras_vggface import utils
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace


from functools import wraps
import sys
import io


class FaceDetection:
    def __init__(self, face_detector, encoder_model, id_to_name, id_to_images):
        self.face_detector = face_detector
        self.encoder_model = encoder_model
        self.id_to_name = id_to_name
        self.id_to_images = id_to_images

    def read_image_array(self, filepath):
        img = cv2.imread(filepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def capture_output(self, func):
        """
        Wrapper to capture print output.
        https://stackoverflow.com/questions/75231091/deepface-dont-print-logs-from-mtcnn-backend
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            old_stdout = sys.stdout
            new_stdout = io.StringIO()
            sys.stdout = new_stdout
            try:
                return func(*args, **kwargs)
            finally:
                sys.stdout = old_stdout

        return wrapper

    def get_face_descriptors(self, img):
        """
        img: array of pixels
        Returns tuple of (encoding, bounding_box) for each face in image.
        """

        results = self.capture_output(self.face_detector.detect_faces)(img)
        if not results:
            print("No detected faces")
            return None, None

        face_descriptors = []
        for result in results:
            x1, y1, width, height = result["box"]
            x1 = max(x1, 0.0)
            y1 = max(y1, 0.0)
            x2, y2 = x1 + width, y1 + height
            image = Image.fromarray(img[y1:y2, x1:x2]).resize((224, 224))
            face_array = np.asarray(image)
            samples = preprocess_input(np.asarray(face_array, "float32"), version=2)
            samples = np.expand_dims(samples, axis=0)
            encoding = self.encoder_model.predict(samples)
            face_descriptors.append((encoding[0], (x1, y1, width, height)))
        return face_descriptors

    def compute_embeddings(self, ann_index, id, filename):
        counter = 0
        video_length = np.load(filename)["colorImages"].shape[-1] - 1
        for i in [0, video_length // 2, video_length]:
            image = np.load(filename)["colorImages"][:, :, :, i]
            face_descriptors = self.get_face_descriptors(image)
            if not face_descriptors:
                print(f"Warning: no descriptor for {filename}")
                continue
            if len(face_descriptors) > 1:
                print(f"Warning: more than two faces found in {filename}")
                continue
            embedding = np.array(face_descriptors[0][0])
            embedding += embedding
            counter += 1
        if counter != 0:
            embedding /= counter
            ann_index.add_item(id, embedding)
        return ann_index

    def recognize(self, img, ann_index):
        face_descriptors = self.get_face_descriptors(img)
        for index, face_descriptor in enumerate(face_descriptors):
            enc, bbox = face_descriptor
            img = cv2.rectangle(img.astype(np.uint8), bbox, (255,0,0), 2)
            temp_data = {}
            temp_data["bbox"] = bbox
            results = ann_index.get_nns_by_vector(enc, 10, search_k=-1, include_distances=True)
            return results
