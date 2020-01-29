import os
from collections import namedtuple
from datetime import datetime

import face_recognition

FACE_PATH = './data/Faces/'
COMMON_FACE_PATH = './data/Common/'

Face = namedtuple('Face', 'name encodings counter')
faces = []

print("START date and time =", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))


class FaceFile(object):
    """Context manager to grab encodings safely."""

    def __init__(self, file_name):
        self.file_name = file_name

    def __enter__(self):
        try:
            encodings = face_recognition.face_encodings(
                face_recognition.load_image_file(
                    COMMON_FACE_PATH + self.file_name))
            return encodings[0]
        except Exception:
            print('Not recognized encoding {}'.format(self.file_name))

    def __exit__(self, *args):
        pass


# collect all file names
known_face_files, unknown_face_files = [
    os.listdir(path) for path in [FACE_PATH, COMMON_FACE_PATH]]

# collect all known faces
for face_file in known_face_files:
    faces.append(
        Face(face_file,
             face_recognition.face_encodings(
                 face_recognition.load_image_file(
                     FACE_PATH + face_file))[0], [0]))

# recognize faces
for face_file in unknown_face_files:
    with FaceFile(face_file) as unknown_encoding:
        try:
            results = face_recognition.compare_faces(
                [face.encodings for face in faces], unknown_encoding)
            for elem in enumerate(results):
                if elem[1]:
                    faces[elem[0]].counter[0] += 1
        except Exception:
            print('can not recognize any known face in {}'.format(face_file))

print("FINISH date and time =", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
# print results
for face in faces:
    print('{} {}'.format(face.name, face.counter))
