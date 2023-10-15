from mtcnn.mtcnn import MTCNN
from PIL import Image
import numpy as np

from celebrity_lookalike.utils import build_image_index
from celebrity_lookalike.face_detection import FaceDetection

from annoy import AnnoyIndex
from keras_vggface.vggface import VGGFace
from joblib import Parallel, delayed
from tqdm import tqdm

id_to_name, name_to_id, id_to_images, image_to_id = build_image_index()

face_detection = FaceDetection(
    face_detector=MTCNN(), 
    encoder_model=VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg'),
    id_to_name=id_to_name,
    id_to_images=id_to_images
    )

ann_index = AnnoyIndex(2048, "angular")
Parallel(n_jobs=20, prefer="threads")(
    delayed(face_detection.compute_embeddings)(ann_index, id, filename)
    for id, filename in tqdm(id_to_images.items())
)
ann_index.build(20, n_jobs=-1)
ann_index.save("ann_index.ann")
