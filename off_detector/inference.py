import errno
import sys
from math import ceil
import json
import pandas as pd
import numpy as np
import cv2
import os
import random
import imutils
import torch
from torchvision import transforms
from off_detector_model import OffDetector

def _candidate_person_for_new_box(people, new_box):
    """
    Returns the existing person from `people` who's last frame is closest to `new_box`
    """
    new_box = np.array(new_box)
    min_distance = float("inf")
    candidate_person_idx = None

    for i, person in enumerate(people):
        last_box = np.array(person[-1]["box"])
        new_box = np.array(new_box)

        dist = np.linalg.norm(last_box[:2] - new_box[:2])

        if dist < min_distance:
            min_distance = dist
            candidate_person_idx = i

    return candidate_person_idx, min_distance

def _new_face_indices(people, detection_results):
    result_dists = []

    for i, detection_res in enumerate(detection_results):
        box = detection_res["box"]
        _, min_dist = _candidate_person_for_new_box(people, box)
        result_dists.append((i, min_dist))

    sorted_result_dists = sorted(result_dists, key=lambda x: x[1])
    surplus = len(detection_results) - len(people)
    return np.array(sorted_result_dists[-surplus:])[:, 0]

def process_and_group_sample(sample, min_confidence=0,
                             min_matching_distance=30,
                             min_frames_to_keep_face=0,
                             padding=0):
    people = []

    for detection_results in sample:
        # New faces have been detected. Check which ones are probably the new ones
        if len(detection_results) > len(people):
            new_face_idxs = _new_face_indices(people, detection_results)
        else:
            new_face_idxs = []

        # Match each face with the person
        for face_idx, detection_res in enumerate(detection_results):
            box = np.array(detection_res["box"])

            if float(detection_res["confidence"]) < min_confidence:
                continue

            if face_idx in new_face_idxs:
                person = []
                people.append(person)
                person_idx = len(people) - 1
                dist = 0
            else:
                person_idx, dist = _candidate_person_for_new_box(people, box)
                person = people[person_idx]

                if dist > min_matching_distance:
                    continue

            box[:2] -= padding
            box[2:] += padding

            detection_res["matching_dist"] = dist
            detection_res["person_idx"] = person_idx
            detection_res["box"] = box

            person.append(detection_res)

    return [person for person in people if len(person) > min_frames_to_keep_face]


def gen_annotations(video_frames, detector_fn, resolution,
                    num_samples_per_video, num_frames_per_sample):
    samples = []

    for start_idx in range(0, len(video_frames), len(video_frames) // num_samples_per_video):
        sample = []

        for frame_idx in range(start_idx, min(len(video_frames), start_idx + num_frames_per_sample)):
            frame = video_frames[frame_idx]
            print(f"\r {frame_idx}", end="")

            ho, wo, co = frame.shape

            if ho < wo:
                frame = imutils.resize(frame, height=resolution)
            else:
                frame = imutils.resize(frame, width=resolution)

            h, w, c = frame.shape
            detection_results = detector_fn(frame)

            for face_idx, detection_res in enumerate(detection_results):
                box = detection_res["box"]
                box = (box * wo / w).astype(int).tolist()
                detection_res["box"] = box
                detection_res["frame_idx"] = frame_idx

            sample.append(detection_results)

        samples.append(sample)

    print("\n")
    return samples

def frames_from_video(cv2_video):
    result = []
    success, image = cv2_video.read()

    while success:
        result.append(image)
        success, image = cv2_video.read()

    cv2_video.release()
    return result

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def save_samples(dir_name, file_name, samples):
    mkdir_p(dir_name)

    for i, sample in enumerate(samples):
        with open(f"{dir_name}/{file_name}_{i}.txt", 'w') as outfile:
            json.dump(sample, outfile)

def process_a_video(path_to_vid,
                    detector_fn,
                    resolution,
                    num_samples_per_video,
                    num_frames_per_sample):

    video_name = path_to_vid.split('/')[-1].split('.')[0]

    cv2_video = get_video(path_to_vid)
    vid_frames = frames_from_video(cv2_video)
    samples = gen_annotations(video_frames=vid_frames,
                              detector_fn=detector_fn,
                              resolution=resolution,
                              num_samples_per_video=num_samples_per_video,
                              num_frames_per_sample=num_frames_per_sample)

    return samples

def get_burst(path_to_vid):
    annot = process_a_video(path_to_vid, detector_fn=light_fast, resolution=200,
                num_samples_per_video=1, num_frames_per_sample=300)
    people = process_and_group_sample(annot[0])
    
    frames_per_person = ", ".join([str(len(person)) for i, person in enumerate(people)])
    print(f"{path_to_vid}. {len(people)} person(s). {frames_per_person} frames respectively")
    return people

def pad_image(img, height, width):
    h, w = img.shape[:2]
    t = 0
    b = height - h
    l = 0
    r = width - w
    return cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT, value=0)

def resize_and_pad(img, height, width, resample=cv2.INTER_AREA):
    target_aspect_ratio = height/width
    im_h, im_w, _ = img.shape
    im_aspect_aspect_ratio = im_h/im_w
    if im_aspect_aspect_ratio>target_aspect_ratio:
        target_height = height
        target_width = int(im_w * target_height/im_h)
    else:
        target_width = width
        target_height = int(im_h * target_width/im_w)
    resized = cv2.resize(img, (target_width, target_height), interpolation=resample)
    return pad_image(resized,height,width)

def get_arrays(path_to_vid, n_subbursts, padding=50, overlap=5, 
               crops_per_subburst=20, height=224, width=224):
    """
    Input:
    path_to_vid
    n_subbursts - number of subbursts to sample from video
    padding - number of pixels to add to detected face before resizing
    overlap - number of frames to overlap the subbursts with
    crops_per_subburst - number of consequtive frames to include in a subburst
    height - integer to resize all crops
    width - integer to resize all crops

    Output:
    5D ndarray of size n_subbursts X CROPS_PER_SUBBURST x 3 X height X width, data from video as unsigned ints
    """

    bursts_metadata = get_burst(path_to_vid)
    bursts = {}
    for burst_id,burst in enumerate(bursts_metadata):
        bursts[burst_id] = []
        for frame_id,frame in enumerate(burst):
            bursts[burst_id].append(frame_id)

    video = get_video(path_to_vid)
    whole_frames = frames_from_video(video)

    burst_frame_inds = []
    for burst_id in bursts:
        frames = bursts[burst_id]
        if crops_per_subburst > len(frames):
            frames += [frames[-1]] * (crops_per_subburst-len(frames)) ## pad if we dont have enough crops
        allsubburst_frames = []
        ind = 0
        for i in range(n_subbursts):
            if ind<len(frames):
                subburst_frames = frames[ind:ind+crops_per_subburst]
                if len(subburst_frames)<crops_per_subburst:
                    subburst_frames += [subburst_frames[-1]] * (crops_per_subburst-len(subburst_frames))
                ind += crops_per_subburst - overlap
            else:
                subburst_frames = allsubburst_frames[-1]
            allsubburst_frames.append(subburst_frames)
        burst_frame_inds.append((burst_id,allsubburst_frames))

    to_return = []
    for burst_id,allsubburst_frames in burst_frame_inds:
        subbursts = []
        for subburst_frames in allsubburst_frames:
            crops = []
            for frame in subburst_frames:
                frame_metadata = bursts_metadata[burst_id][frame]
                frame_mat = whole_frames[frame_metadata['frame_idx']]
                frame_mat = cv2.cvtColor(frame_mat, cv2.COLOR_BGR2RGB)
                left, top, right, bottom = frame_metadata['box']
                left -= padding
                top -= padding
                right += padding
                bottom += padding
                if left<0:
                    left = 0
                if top<0:
                    top = 0
                frame_mat = frame_mat[top:bottom, left:right]
                if frame_mat.size==0:
                    print('empty mat found',vid_id)
                    continue
                frame_mat = resize_and_pad(frame_mat, height, width)
                crops.append(frame_mat)
            subbursts.append(crops)
        to_return.append(np.stack(subbursts))
        
    if not to_return:
        print('no crops found')
        crops = [np.random.random_integers(0,255,(height,width,3)).astype(np.uint8)]
        crops += [crops[-1] for _ in range(crops_per_subburst - 1)]
        subbursts = [crops] * n_subbursts
        to_return = [np.stack(subbursts)]

    return to_return

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
off_detector = OffDetector()
off_detector.load_state_dict(torch.load('weights.pt', map_location='cpu'))
off_detector = off_detector.to(device)
off_detector.eval()

def run_prediction(path_to_vid):

    arrs = get_arrays(path_to_vid, 20)
    preds = []
    for arr in arrs:
        tens = torch.stack([torch.stack([transform(crop) for crop in arr_]) for arr_ in arr]).to(device)
        tens = tens.unsqueeze(0)
        with torch.no_grad():
            pred = torch.sigmoid(off_detector(tens))[0].item()
        preds.append(pred)
    return preds

def run_predictions(dir_):
    num_fake = 0
    num_real = 0
    ave_prob = 0
    all_preds = []
    dir_name = dir_.split('/')[-1]
    for fn in os.listdir(dir_):
        if fn.endswith('mp4'):
            preds = run_prediction(f"{dir_}/{fn}")
            all_preds.append(preds)
            pred = np.max(preds)
            pred_label = 'REAL' if pred<.5 else 'FAKE'
            print(num_real+num_fake, pred_label, pred, preds)
            if pred_label=='FAKE':
                num_fake += 1
            else:
                num_real += 1
            ave_prob += pred
            print(num_fake/(num_fake+num_real),ave_prob/(num_fake+num_real))
    ave_prob /= (num_fake + num_real)
    print(num_fake,num_real,ave_prob)
    with open(f'{dir_name}_preds.json','w') as f:
        json.dump(all_preds,f)



## ================= Light and fast detector ========================

import argparse
import cv2
import time
import mxnet as mx
import numpy as np
import imutils

import face_detect.face_detect_predict as predict

ctx = mx.cpu() #mx.gpu(0)
import face_detect.configuration_10_320_20L_5scales_v2 as cfg

symbol_file_path = 'face_detect/symbol_10_320_20L_5scales_v2_deploy.json'
model_file_path = 'face_detect/train_10_320_20L_5scales_v2_iter_1000000.params'

face_predictor = predict.Predict(mxnet=mx,
                                 symbol_file_path=symbol_file_path,
                                 model_file_path=model_file_path,
                                 ctx=ctx,
                                 receptive_field_list=cfg.param_receptive_field_list,
                                 receptive_field_stride=cfg.param_receptive_field_stride,
                                 bbox_small_list=cfg.param_bbox_small_list,
                                 bbox_large_list=cfg.param_bbox_large_list,
                                 receptive_field_center_start=cfg.param_receptive_field_center_start,
                                 num_output_scales=cfg.param_num_output_scales)


def light_fast(frame):
    bboxes, infer_time = face_predictor.predict(frame)
    frame_res = []

    for box in bboxes:
        frame_res.append({
            "box": np.array(box)[:4].astype(int),
            "confidence": float(box[4])
        })

    return frame_res

## ==================================================================

#run_predictions('/home/ubuntu/reals') ## pass a path to a directory
print(run_prediction('obama.mp4'))
