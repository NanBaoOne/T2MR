"""
Copyright 2024 LY Corporation
LY Corporation licenses this file to you under the CC BY-NC 4.0
(the "License"); you may not use this file except in compliance
with the License. You may obtain a copy of the License at:
    https://creativecommons.org/licenses/by-nc/4.0/
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations
under the License.
"""

import os
import random
from os.path import join as pjoin

import numpy as np
import torch
from torch.utils import data
from tqdm import tqdm


class TextMotionPatchDataset(data.Dataset):
    """
    HumanML3D / KIT-ML dataset in the NTU-120 25-joint format.

    Motions are loaded from `cfg.dataset.motion_dir` as (T, 25, 3) arrays.
    Captions are loaded from `cfg.dataset.text_dir` as text files whose lines
    are formatted as: caption#tokens#start#end
    """

    def __init__(self, cfg, mean, std, split_file, eval_mode=False, fps=None):
        self.cfg = cfg
        self.eval_mode = eval_mode
        self.max_motion_length = cfg.dataset.max_motion_length
        self.fps = fps
        self.data_dict = {}

        id_list = []
        with open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())

        name_list = []
        length_list = []

        for name in tqdm(id_list):
            try:
                motion_path = pjoin(cfg.dataset.motion_dir, name + ".npy")
                if not os.path.exists(motion_path):
                    continue

                motion = np.load(motion_path)

                # Validate format: (seqlen, joints_num, 3)
                if len(motion.shape) != 3 or motion.shape[1] != cfg.dataset.joints_num:
                    print(f"Skipping {name}: expected (T, {cfg.dataset.joints_num}, 3), got {motion.shape}")
                    continue

                if np.isnan(motion).any():
                    continue

                text_data = []
                flag = False
                text_path = pjoin(cfg.dataset.text_dir, name + ".txt")
                if os.path.exists(text_path):
                    with open(text_path, encoding="utf-8") as f:
                        for index, line in enumerate(f.readlines()):
                            if eval_mode and index >= 1:
                                continue

                            line_split = line.strip().split("#")
                            if len(line_split) < 4:
                                continue

                            caption = line_split[0]
                            tokens = line_split[1].split(" ")
                            f_tag = float(line_split[2])
                            to_tag = float(line_split[3])

                            f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            to_tag = 0.0 if np.isnan(to_tag) else to_tag

                            text_dict = {"caption": caption, "tokens": tokens}

                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True
                                text_data.append(text_dict)
                            else:
                                start_frame = int(f_tag * cfg.dataset.fps)
                                end_frame = int(to_tag * cfg.dataset.fps)
                                n_motion = motion[start_frame:end_frame]

                                new_name = random.choice("ABCDEFGHIJKLMNOPQRSTUVW") + "_" + name
                                while new_name in self.data_dict:
                                    new_name = random.choice("ABCDEFGHIJKLMNOPQRSTUVW") + "_" + name

                                self.data_dict[new_name] = {
                                    "motion": n_motion,
                                    "length": len(n_motion),
                                    "text": [text_dict],
                                }
                                name_list.append(new_name)
                                length_list.append(len(n_motion))

                if flag:
                    self.data_dict[name] = {
                        "motion": motion,
                        "length": len(motion),
                        "text": text_data,
                    }
                    name_list.append(name)
                    length_list.append(len(motion))

            except Exception as e:
                print(f"Error processing {name}: {e}")

        self.name_list, self.length_list = zip(
            *sorted(zip(name_list, length_list), key=lambda x: x[1])
        )

        self.mean = mean
        self.std = std
        self.joint_num = cfg.dataset.joints_num

    def real_len(self):
        return len(self.data_dict)

    def __len__(self):
        return self.real_len() * self.cfg.dataset.times

    def __getitem__(self, item):
        idx = item % self.real_len()
        key = self.name_list[idx]
        data = self.data_dict[key]

        motion, m_length, text_list = data["motion"], data["length"], data["text"]

        # Normalize with the dataset statistics
        motion = (motion - self.mean[np.newaxis, ...]) / self.std[np.newaxis, ...]

        # Select a caption
        if self.eval_mode:
            caption = text_list[-1]["caption"]
        else:
            text_data = random.choice(text_list)
            caption = text_data["caption"]

        max_motion_length = self.max_motion_length

        if m_length >= max_motion_length:
            start_idx = (
                random.randint(0, m_length - max_motion_length)
                if not self.eval_mode
                else 0
            )
            motion = motion[start_idx:start_idx + max_motion_length]
            m_length = max_motion_length
        else:
            if self.cfg.preprocess.padding:
                padding_len = max_motion_length - m_length
                D = motion.shape[1]
                C = motion.shape[2]
                padding_zeros = np.zeros((padding_len, D, C), dtype=np.float32)
                motion = np.concatenate((motion, padding_zeros), axis=0)

        motion = torch.tensor(motion).float()

        return caption, motion, m_length, key
