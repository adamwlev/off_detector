import errno
from math import ceil
import json
import time
import pandas as pd
import numpy as np
import cv2
import os
import random
import imutils
import boto3
import torch
import torch.nn as nn
from torchvision import transforms
from off_detector_model import OffDetector


CROPS_PER_SUBBURST = 20
TRAIN_PARTS = list(range(35))
VAL_PARTS = list(range(40,50))
TRAIN_PCT_SUBSET = .3
VAL_PCT_SUBSET = .2
BALANCE = True
NUM_WORKERS_DL = 7
PADDING_CROP = 50
BATCH_SIZE_TRAIN = 4
BATCH_SIZE_VAL = 1
STEP_EVERY = 10
NUM_SUBBURSTS_TRAIN = 2
NUM_SUBBURSTS_VAL = 13



def _candidate_person_for_new_res(people, new_res, max_frame_diff):
    """
    Returns the existing person from `people` who's last frame is closest to `new_res.box`
    """
    new_box = np.array(new_res["box"])
    min_distance = float("inf")
    candidate_person_idx = None

    for i, person in enumerate(people):
        last_box = np.array(person[-1]["box"])
        new_box = np.array(new_box)

        dist = np.linalg.norm(last_box[:2] - new_box[:2])
        frame_diff = new_res["frame_idx"] - person[-1]["frame_idx"]

        if dist < min_distance and frame_diff <= max_frame_diff:
            min_distance = dist
            candidate_person_idx = i

    return candidate_person_idx, min_distance


def _new_face_indices(people, detection_results, max_frame_diff):
    result_dists = []

    for i, detection_res in enumerate(detection_results):
        _, min_dist = _candidate_person_for_new_res(people, detection_res, max_frame_diff)
        result_dists.append((i, min_dist))

    sorted_result_dists = sorted(result_dists, key=lambda x: x[1])
    surplus = len(detection_results) - len(people)
    return np.array(sorted_result_dists[-surplus:])[:, 0]


def process_and_group_sample(sample, min_confidence=0,
                              max_matching_distance=50,
                              min_frames_to_keep_face=15,
                              padding=0,
                              max_frame_diff=2):
  
    people = []

    for frame_idx, detection_results in enumerate(sample):
        # New faces have been detected. Check which ones are probably the new ones
        if len(detection_results) > len(people):
            new_face_idxs = _new_face_indices(people, detection_results, max_frame_diff)
        else:
            new_face_idxs = []

        # Match each face with the person
        for face_idx, detection_res in enumerate(detection_results):
            if float(detection_res["confidence"]) < min_confidence:
                continue

            is_new_person = False

            if face_idx in new_face_idxs:
                is_new_person = True
            else:
                person_idx, dist = _candidate_person_for_new_res(people, detection_res, max_frame_diff)

                if person_idx is None:
                  is_new_person = True
                else:
                  person = people[person_idx]

                if dist > max_matching_distance:
                   is_new_person = True

            if is_new_person:
              person = []
              people.append(person)
              person_idx = len(people)-1
              dist = 0

            box = np.array(detection_res["box"])
            box[:2] -= padding
            box[2:] += padding

            detection_res["matching_dist"] = dist
            detection_res["person_idx"] = person_idx
            detection_res["box"] = box.tolist()

            person.append(detection_res)

    return [person for person in people if len(person) > min_frames_to_keep_face]

class DataPipeline:
    def __init__(self, aws_access_key_id, aws_secret_access_key):
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        session = boto3.Session(
        aws_access_key_id = aws_access_key_id,
        aws_secret_access_key = aws_secret_access_key,
        )
        s3 = session.resource('s3')
        self.s3client = session.client('s3')
        self.bucket = 'deep-fake'
  
    def get_metadata(self):
        key = 'all_metadata.csv'
        self.s3client.download_file(Bucket=self.bucket, Key=key, Filename=key)
        self.metadata = pd.read_csv(key)
        return self.metadata
  
    def get_video(self, part, filename):
        path = filename
        if not os.path.exists(path):
            key = f'dfdc_train_all/dfdc_train_part_{part}/{filename}'
            self.s3client.download_file(Bucket=self.bucket, Key=key, Filename=path)
        vid = cv2.VideoCapture(path)
        os.remove(path)
        return vid

    def frames_from_video(self, cv2_video):
        result = []
        success, image = cv2_video.read()

        while success:
            result.append(image)
            success, image = cv2_video.read()

        cv2_video.release()
        return result
  
    def annotations_for_vid(self, vid_id):
        path = f"{vid_id}_0.txt"
        if not os.path.exists(path):
            key = f'face_annnotations/{vid_id}_0.txt'
            self.s3client.download_file(Bucket=self.bucket, Key=key, Filename=path)
        with open(path, 'r') as f:
            d = json.load(f)
        os.remove(path)
        return d

    def get_burst(self, vid_id):
        annot = self.annotations_for_vid(vid_id)
        people = process_and_group_sample(annot)
        
        frames_per_person = ", ".join([str(len(person)) for i, person in enumerate(people)])
        return people

    def pad_image(self, img, height, width):
        h, w = img.shape[:2]
        t = 0
        b = height - h
        l = 0
        r = width - w
        return cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT, value=0)

    def resize_and_pad(self, img, height, width, resample=cv2.INTER_AREA):
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
        return self.pad_image(resized,height,width)
    
    def cache_bursts(self, vid_id, part, cache_path, padding, height, width, fake_vid_id=None):
        os.mkdir(cache_path)
        bursts = self.get_burst(fake_vid_id if fake_vid_id else vid_id)
        crop_inds = [(i,j) for i,b in enumerate(bursts) for j,_ in enumerate(b)]
        video = self.get_video(part, vid_id + '.mp4')
        frames = self.frames_from_video(video)
        for burst_ind, burst in enumerate(bursts):
            for frame_ind, frame in enumerate(burst):
                left, top, right, bottom = frame['box']
                left -= padding
                top -= padding
                right += padding
                bottom += padding
                if left<0:
                    left = 0
                if top<0:
                    top = 0

                frame_mat = frames[frame['frame_idx']][top:bottom, left:right]
                if frame_mat.size==0:
                    print('empty mat found',vid_id)
                    continue
                frame_mat = self.resize_and_pad(frame_mat, height, width)
                save_path = f"{cache_path}/{burst_ind}_{frame_ind}.png"
                cv2.imwrite(save_path, frame_mat)

    def get_arrays(self, vid_id, n_subbursts, padding, random_start_points=True, 
                   stride=3, height=224, width=224):
        """
        Input:
        video id - string with no "mp4"
        n_subbursts - number of subbursts to sample from video
        padding - number of pixels to add to detected face before resizing
        random_start_points - whether to randomly sample the starting points of subbursts
        stride - integer for number of frames to stride between subburts (only 
                 relevant for non random start points)
        height - integer to resize all crops
        width - integer to resize all crops

        Output: dictionary with follow key-value pairs
        "subbursts": 5D ndarray of size n_subbursts X CROPS_PER_SUBBURST x 3 X height X width, data from video as unsigned ints
        "labels": 0D ndarray, 0-1 for whether video is fake (1) or real (0), as a float
        """

        fn = vid_id + ".mp4"
        row = self.metadata[self.metadata['filename'] == fn].iloc[0]

        subbursts = []

        cache_path = f"cache/{vid_id}_{height}_{width}"
        if not os.path.exists(cache_path):
            self.cache_bursts(vid_id, row.part, cache_path, padding, height, width)

        fns = os.listdir(cache_path)
        bursts = {}
        if not fns:
            time.sleep(5.4)
            fns = os.listdir(cache_path)
            if not fns:
                pass
                #print('no crop files found', vid_id)
            else:
                for fn in fns:
                    burst_id, frame_id = int(fn.split('_')[0]), int(fn.split('_')[1].split('.')[0])
                    if burst_id not in bursts:
                        bursts[burst_id] = []
                    bursts[burst_id].append(frame_id)
        else:
            for fn in fns:
                burst_id, frame_id = int(fn.split('_')[0]), int(fn.split('_')[1].split('.')[0])
                if burst_id not in bursts:
                    bursts[burst_id] = []
                bursts[burst_id].append(frame_id)
        for key in bursts:
            bursts[key] = sorted(bursts[key])

        if bursts:
            burst_ids = list(bursts.keys())
            burst_id = random.choice(burst_ids)
            frames = bursts[burst_id]
            #print(frames)

            if CROPS_PER_SUBBURST > len(frames):
                frames += [frames[-1]] * (CROPS_PER_SUBBURST-len(frames)) ## pad if we dont have enough crops
            
            allsubburst_frames = []
            ind = 0
            for i in range(n_subbursts):
                if random_start_points:
                    possible_starting_inds = list(range(0,len(frames)-CROPS_PER_SUBBURST+1))
                    #print(possible_starting_inds)
                    starting_ind = random.choice(possible_starting_inds)
                    subburst_frames = frames[starting_ind:starting_ind+CROPS_PER_SUBBURST]
                else:
                    if ind<len(frames):
                        subburst_frames = frames[ind:ind+CROPS_PER_SUBBURST]
                        if len(subburst_frames)<CROPS_PER_SUBBURST:
                            subburst_frames += [subburst_frames[-1]] * (CROPS_PER_SUBBURST-len(subburst_frames))
                        ind += CROPS_PER_SUBBURST + stride
                    else:
                        subburst_frames = allsubburst_frames[-1]
                allsubburst_frames.append(subburst_frames)
        else:
            allsubburst_frames = []

        #print(allsubburst_frames)
        for subburst_frames in allsubburst_frames:
            crops = []
            for frame in subburst_frames:
                im_path = f"{cache_path}/{burst_id}_{frame}.png"
                frame_mat = cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2RGB)
                crops.append(frame_mat)
            subbursts.append(crops)

        label = np.array(0 if row.label=='REAL' else 1,dtype=np.float)        
            
        if not subbursts:
            #print('no crops found')
            crops = [np.random.random_integers(0,255,(height,width,3)).astype(np.uint8)]
            crops += [crops[-1] for _ in range(CROPS_PER_SUBBURST - 1)]
            subbursts = [crops] * n_subbursts

        subbursts = np.stack(subbursts)

        return dict(zip(("subbursts", "labels"),(subbursts, label)))

class Dataset(torch.utils.data.Dataset):
    def __init__(self, transform, num_subbursts=1, parts_subset=None, 
                 pct_subset=None, balance=True, random_start_points=False, seed1=4, seed2=33):
        """
        num_subbursts = number of subbursts to sample from a burst
        parts_subset = an iterable containing integers corresponding to the parts of the train set to use
        pct_subset = a float fraction of the data to use (after parts subset but before balance)
        balance = a boolean indicating whether to balance the training set by undersampling FAKE examples
        """
        self.dp = DataPipeline("","")
        self.metadata = self.dp.get_metadata()
        if parts_subset:
            self.metadata = self.metadata[self.metadata.part.isin(parts_subset)]
        if pct_subset:
            self.metadata = self.metadata.sample(frac=pct_subset, random_state=seed1)
        if balance:
            reals = self.metadata[self.metadata.label=='REAL']
            self.num_reals = len(reals)
            fakes = self.metadata[self.metadata.label=='FAKE']
            fakes = fakes.sample(n=self.num_reals, random_state=seed2)
            self.metadata = pd.concat([reals,fakes])

        self.num_subbursts = num_subbursts
        self.transform = transform
        self.padding = PADDING_CROP
        self.random_start_points = random_start_points

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, index):
        """
        index is an int. it corresponds to the video at iloc index in the df
        """
        row = self.metadata.iloc[index]
        vid_id = row.filename.split('.')[0]
        array_dict = self.dp.get_arrays(vid_id, self.num_subbursts, self.padding, self.random_start_points)
        array_dict['subbursts'] = self.apply_transform(array_dict['subbursts'])
        return array_dict

    def apply_transform(self, arr):
        return torch.stack([torch.stack([self.transform(crop) for crop in crops]) for crops in arr])


train_transform = transforms.Compose([
                                      #transforms.ToPILImage(),
                                      #transforms.RandomCrop(size=(224,224),pad_if_needed=True),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225])]) ## imagenet defaults       
train_ds = Dataset(transform=train_transform,
                   num_subbursts=NUM_SUBBURSTS_TRAIN, parts_subset=TRAIN_PARTS,
                   pct_subset=TRAIN_PCT_SUBSET, balance=BALANCE, random_start_points=True)
print(f'Number Train Overall Videos: {len(train_ds)}')
print(f'Number Train Real Videos: {train_ds.num_reals}')
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE_TRAIN,
                                       shuffle=True, num_workers=NUM_WORKERS_DL, drop_last=True)

val_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])]) ## imagenet defaults 

val_ds = Dataset(transform=val_transform,
                 num_subbursts=NUM_SUBBURSTS_VAL, parts_subset=VAL_PARTS,
                 pct_subset=VAL_PCT_SUBSET, balance=BALANCE, random_start_points=False)
print(f'Number Val Overall Videos: {len(val_ds)}')
print(f'Number Val Real Videos: {val_ds.num_reals}')
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE_VAL,
                                     shuffle=False, num_workers=NUM_WORKERS_DL, drop_last=True)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = OffDetector()
model = model.to(device)
torch.backends.cudnn.benchmark = True
obj = nn.BCEWithLogitsLoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-3, momentum=.9)
optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, gamma=0.7)

for epoch in range(20):
    model.train()

    train_losses, train_nright, train_n = [], 0, 0
    val_losses, val_nright, val_n = [], 0, 0
    train_start = int(time.time())
    for i,batch in enumerate(train_dl):
        
        subbursts = batch['subbursts'].to(device)
        labels = batch['labels'].to(device)
        #print('train',subbursts.size(),labels.size())

        cls_pred = model(subbursts)
        loss = obj(cls_pred,labels)
        loss.backward()
        train_losses.append(loss.item())
        train_nright += ((cls_pred>0).float()==labels).sum().item()
        train_n += labels.size(0)

        if (i+1)%STEP_EVERY==0:
            optimizer.step()
            optimizer.zero_grad()
    train_end = int(time.time())
    train_elapsed_time = train_end - train_start
    
    val_start = int(time.time())
    model.eval()

    with torch.set_grad_enabled(False):
        for i,batch in enumerate(val_dl):
            
            subbursts = batch['subbursts'].to(device)
            labels = batch['labels'].to(device)
            #print('val',subbursts.size(),labels.size())

            cls_pred = model(subbursts)
            loss = obj(cls_pred,labels)
            val_losses.append(loss.item())
            val_nright += ((cls_pred>0).float()==labels).sum().item()
            val_n += labels.size(0)
    
    val_end = int(time.time())
    val_elapsed_time = val_end - val_start

    train_ex_ps = train_elapsed_time/len(train_ds)
    val_ex_ps = val_elapsed_time/len(val_ds)
    
    print(f'epoch {epoch}: tr loss: {np.mean(train_losses):.4f}, '
          f'vl loss: {np.mean(val_losses):.4f}, '
          f'tr acc: {train_nright/train_n:.4f}, '
          f'vl acc: {val_nright/val_n:.4f}, '
          f'tr time: {train_elapsed_time}, '
          f'vl time: {val_elapsed_time}, '
          f'tr spv : {train_ex_ps:.4f}, '
          f'vl spv : {val_ex_ps:.4f}, ')

    scheduler.step()
    torch.save(model.state_dict(), f'epoch{epoch}.pt')
