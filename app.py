import streamlit as st
import cv2
import numpy as np
from PIL import Image

from celebrity_lookalike.face_detection import FaceDetection
from mtcnn.mtcnn import MTCNN
from celebrity_lookalike.utils import build_image_index
from keras_vggface.vggface import VGGFace
from annoy import AnnoyIndex
import wikipedia as wp

with st.sidebar:
    img_file = st.file_uploader("Choose an image of someone")


st.title("Celebrity Lookalike Finder")

# Load FaceDetection class and Ann Index
id_to_name, name_to_id, id_to_images, image_to_id = build_image_index()
face_detection = FaceDetection(
    face_detector=MTCNN(),
    encoder_model=VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg'),
    id_to_name=id_to_name,
    id_to_images=id_to_images
    )

ann_index = AnnoyIndex(2048, 'angular')
ann_index.load('ann_index.ann')


def ask_wikipedia(celebrity_name):
    print(f"Searching for {celebrity_name}'s Wikipedia page")
    title = wp.search(celebrity_name, results=1)
    print("Title found: ", title)
    return wp.summary(title, sentences=3, auto_suggest=False)

# Face detection function
def detect(img_file):
    col1, col2 = st.columns(2)
    with col1:
        st.title("Input image")
        img_array = cv2.imdecode(np.frombuffer(img_file.getvalue(), np.uint8), -1)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        enc, bbox = face_detection.get_face_descriptors(img_array)[0]
        img = cv2.rectangle(img_array.astype(np.uint8), bbox, (255, 0, 0), 2)
        st.image(img, use_column_width=True, caption='Input image')

    with col2:
        results = face_detection.recognize(img_array, ann_index)
        print("Prediction: ", {
            f"{id_to_name[id]} ({id})": distance
            for id, distance in zip(results[0], results[1])
        })

        top_result_id = results[0][0]
        top_result_name = id_to_name[top_result_id]
        top_result_img = Image.fromarray(np.load(id_to_images[top_result_id])['colorImages'][:, :, :, 0])
        st.title(f'Result')
        st.image(top_result_img, width=200, caption=f"{top_result_name}")

    st.title(f'Who is {top_result_name}?')
    try:
        st.markdown(ask_wikipedia(top_result_name))
    except:
        st.markdown(f"No Wikipedia page found for {top_result_name}")



if img_file:
    detect(img_file)
