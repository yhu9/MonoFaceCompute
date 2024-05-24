from glob import glob
import re
from pathlib import Path

import argparse
from tqdm import tqdm
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np 

import mediapipe as mp
import face_alignment
from fdlite import (
    FaceDetection,
    FaceLandmark,
    face_detection_to_roi,
    IrisLandmark,
    iris_roi_from_face_landmarks,
)

class ImageDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path)
        img = np.array(img) # (H,W,3)
        img = torch.from_numpy(img)
        return img, img_path

class MediaPipeLandmarksDetector:
    def __init__(self, threshold=0.1, max_faces=1, video_based=False):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_face_detection = mp.solutions.face_detection
        # self.mp_face_mesh_options = mp.FaceMeshCalculatorOptions()

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=not video_based,
            refine_landmarks=True,
            max_num_faces=max_faces,
            min_detection_confidence=threshold)

    def __call__(self, image):
        '''
        image: 0-255, uint8, rgb, [h, w, 3]
        return: detected box list (in image coordinates), landmarks list (in image coordinates)
        '''

        results = self.face_mesh.process(image)

        if not results.multi_face_landmarks: 
            # this is a really weird thing, but somehow (especially when switching from one video to another) nothing will get picked up on the 
            # first run but it will be after the second run.
            results = self.face_mesh.process(image) 

        if not results.multi_face_landmarks:
            return None

        face_landmarks = results.multi_face_landmarks[0]
        landmarks = MediaPipeLandmarksDetector._mediapipe2torch(face_landmarks.landmark)
        
        # scale landmarks to image size
        landmarks[:,0] *= image.shape[1]
        landmarks[:,1] *= image.shape[0]

        # left = np.min(landmarks[:, 0])
        # right = np.max(landmarks[:, 0])
        # top = np.min(landmarks[:, 1])
        # bottom = np.max(landmarks[:, 1])
        # bbox = [left, top, right, bottom]
        # return landmarks, boxes

        return landmarks
    
    def _mediapipe2torch(landmarks): 
        t = torch.zeros((len(landmarks), 3), dtype=torch.float)
        for i, lmk in enumerate(landmarks):
            t[i, 0] = float(lmk.x)
            t[i, 1] = float(lmk.y)
            t[i, 2] = float(lmk.z)
        return t


def interpolate_missing_detections(detections, txt=""):
    missing = [i for i,v in enumerate(detections) if v is None]
    if len(missing) == 0:
        return
    else:
        print(f"Interpolating values for {len(missing)} frame{'s' if len(missing) > 1 else ''} without {txt} detections")
        print(missing)

    last_ok_i = -1
    for i, detection in enumerate(detections):
        if detection is not None:
            if last_ok_i != i-1:
                if last_ok_i == -1:
                    # Copy from frame 0 to frame {i-1}
                    print(f"Missing detections at the beginning of the sequence - copying first detection")
                    for j in range(i):
                        detections[j] = detections[i].clone()
                else:
                    # Interpolate from frame {last_ok_i+1} to frame {i-1}
                    # ex: last_ok_i=10, i=12 => t = [1/2]
                    # ex: last_ok_i=10, i=13 => t = [1/3, 2/3]    
                    for j in range(last_ok_i+1, i):
                        t = (j-last_ok_i) / (i-last_ok_i)
                        detections[j] = (1-t) * detections[last_ok_i] + t * detections[i]
            last_ok_i = i
        elif i == len(detections) - 1:
            if last_ok_i == -1:
                raise RuntimeError("No detections, unable to recover.")
            # Copy from frame {last_ok_i+1} to frame {n-1}
            print(f"Missing detections at the end of the sequence - copying last detection")
            for j in range(last_ok_i+1, len(detections)):
                detections[j] = detections[last_ok_i].clone()


if __name__ == "__main__":
    """
        Detects FAN and MediaPipe landmarks on the input images.
        This is not very efficient and runs on the CPU because of the underlying implementations.
    """
    parser = argparse.ArgumentParser(description="Detect keypoints on face images.")
    parser.add_argument("--images", type=str, help="Pattern to images (glob)")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    args = parser.parse_args()

    image_paths = glob(args.images)
    image_paths.sort(key=lambda f: int(re.sub("\D", "", f)))

    output_dir = Path(args.output_dir)

    detector_mp = MediaPipeLandmarksDetector(video_based=True)
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D)
    detect_faces = FaceDetection()
    detect_face_landmarks = FaceLandmark()
    detect_iris_landmarks = IrisLandmark()

    landmarks_mp_all = []
    landmarks_fan_all = []
    landmarks_iris_all = []

    dataset = ImageDataset(image_paths)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
    for img, path in tqdm(data_loader):
        img = img[0].numpy()
        path = path[0]
        height, width, _ = img.shape

        ###################################################
        # MediaPipe detection
        landmarks_mp = detector_mp(img)
        # if landmarks_mp is None:
        #     print(f"No MediaPipe landmarks detection for frame {path}")
        landmarks_mp_all.append(landmarks_mp)
        ###################################################

        ###################################################
        # FAN detection
        # We could reuse the box from MediaPipe, but probably better to use this one
        bbox = fa.face_detector.detect_from_image(img)
        if len(bbox) == 0:
            # raise ValueError(f"No face detected for frame {path}")
            landmarks_fan = None
        else:
            if len(bbox) > 1:
                print(f"Detected multiple faces for frame {path} - using first one")
            landmarks_fan = fa.get_landmarks_from_image(img, bbox)
            # if landmarks_fan is None:
            #     print(f"No FAN detection for frame {path}")
            landmarks_fan = torch.from_numpy(landmarks_fan[0])
            
        landmarks_fan_all.append(landmarks_fan)
        ###################################################

        ###################################################
        # fdlite iris detection
        # This is all a bit wasteful since we're re-detecting the MP landmarks (we can't just reuse the mediapipe ones since they're 378 and fdlite expects 568)
        face_detections = detect_faces(img)
        if len(face_detections) == 0:
            # raise ValueError(f"No fdlite face detection for frame {path}")
            iris_lmks = None
        else:
            face_detection = max(face_detections, key=lambda d: d.score)
            face_roi = face_detection_to_roi(face_detection, (width, height))
            face_landmarks = detect_face_landmarks(img, face_roi)
            if len(face_landmarks) == 0:
                # print(f"No fdlite landmarks detection for frame {path}")
                iris_lmks = None
            else:
                iris_rois = iris_roi_from_face_landmarks(face_landmarks, (width, height))
                if len(iris_rois) != 2:
                    # raise ValueError(f"No fdlite iris ROIs for frame {path}")
                    iris_lmks = None
                else:
                    left_eye_detection = detect_iris_landmarks(img, iris_rois[0]).iris[0]
                    right_eye_detection = detect_iris_landmarks(img, iris_rois[1], is_right_eye=True).iris[0]
                    iris_lmks = torch.tensor([[right_eye_detection.x * width, right_eye_detection.y * height],
                                            [left_eye_detection.x * width, left_eye_detection.y * height]], dtype=torch.float)
        landmarks_iris_all.append(iris_lmks)
        ###################################################    output_dir = Path(args.output_dir)

    interpolate_missing_detections(landmarks_mp_all, "MediaPipe landmarks")
    interpolate_missing_detections(landmarks_fan_all, "FAN landmarks")
    interpolate_missing_detections(landmarks_iris_all, "iris landmarks")

    landmarks_mp_all = torch.stack(landmarks_mp_all).round().int() # (n_frames, 478, 3)
    landmarks_fan_all = torch.stack(landmarks_fan_all).round().int() # (n_frames, 68, 3)
    landmarks_iris_all = torch.stack(landmarks_iris_all).round().int() # (n_frames, 4)
    torch.save(landmarks_mp_all, output_dir / "landmarks_mp.pt")
    torch.save(landmarks_fan_all, output_dir / "landmarks_fan.pt")
    torch.save(landmarks_iris_all, output_dir / "landmarks_iris.pt")

