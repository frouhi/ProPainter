# -*- coding: utf-8 -*-
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import time
import importlib
import os
import argparse
import copy
import datetime
import random
import sys
import json

import torch
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models
import torch.multiprocessing as mp
from torchvision import transforms

# My libs
from core.utils import Stack, ToTorchFormatTensor


w, h = 432, 240
ref_length = 10
neighbor_stride = 5
default_fps = 24

_to_tensors = transforms.Compose([
    Stack(),
    ToTorchFormatTensor()])


# sample reference frames from the whole video 
def get_ref_index(neighbor_ids, length):
    ref_index = []
    for i in range(0, length, ref_length):
        if not i in neighbor_ids:
            ref_index.append(i)
    return ref_index


def load_model(model="sttn", ckpt="STTN/checkpoints/sttn.pth"):
    # load the models 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = importlib.import_module('STTN.model.' + model)
    model = net.InpaintGenerator().to(device)
    data = torch.load(ckpt, map_location=device)
    model.load_state_dict(data['netG'])
    print('loading from: {}'.format(ckpt))
    model.eval()
    return model


def run(model, npframes, npmasks, out_path=None):
    """
        mask: masks TODO: make sure the format is correct (must be the same as read_mask)
        frames: video frames TODO: make sure the format is correct (must be the same as read_frame_from_videos)
        out_path: video is saved here! If its None, no video is saved

        returns comp_ls (frames of the inpainted video)
    """
    # prepare datset, encode all frames into deep space 
    # frames = read_frame_from_videos(args.video)
    frames = []
    for i in range(len(npframes)):
        frames.append(Image.fromarray(npframes[i].astype('uint8'), mode="RGB").resize((w,h)))
    video_length = len(frames)
    feats = _to_tensors(frames).unsqueeze(0)*2-1
    frames = [np.array(f).astype(np.uint8) for f in frames]

    # masks = read_mask(args.mask)
    masks = []
    npmasks = npmasks.reshape(*npmasks.shape, 1).repeat(3, axis=3)
    npmasks = npmasks * npframes
    for i in range(len(npmasks)):
        m = Image.fromarray(npmasks[i].astype('uint8'), mode="RGB")
        m = m.resize((w, h), Image.NEAREST)
        m = np.array(m.convert('L'))
        m = np.array(m > 0).astype(np.uint8)
        m = cv2.dilate(m, cv2.getStructuringElement(
            cv2.MORPH_CROSS, (3, 3)), iterations=4)
        masks.append(Image.fromarray(m*255))
    del npmasks, npframes
    binary_masks = [np.expand_dims((np.array(m) != 0).astype(np.uint8), 2) for m in masks]
    masks = _to_tensors(masks).unsqueeze(0)
    feats, masks = feats.to(device), masks.to(device)
    comp_frames = [None]*video_length

    with torch.no_grad():
        feats = model.encoder((feats*(1-masks).float()).view(*feats.shape[1:]))#video_length, 3, h, w))
        _, c, feat_h, feat_w = feats.size()
        feats = feats.view(1, video_length, c, feat_h, feat_w)
    # print('loading videos and masks from: {}'.format(args.video))

    # completing holes by spatial-temporal transformers
    for f in range(0, video_length, neighbor_stride):
        neighbor_ids = [i for i in range(max(0, f-neighbor_stride), min(video_length, f+neighbor_stride+1))]
        ref_ids = get_ref_index(neighbor_ids, video_length)
        with torch.no_grad():
            pred_feat = model.infer(
                feats[0, neighbor_ids+ref_ids, :, :, :], masks[0, neighbor_ids+ref_ids, :, :, :])
            pred_img = torch.tanh(model.decoder(
                pred_feat[:len(neighbor_ids), :, :, :])).detach()
            pred_img = (pred_img + 1) / 2
            pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy()*255
            for i in range(len(neighbor_ids)):
                idx = neighbor_ids[i]
                img = np.array(pred_img[i]).astype(
                    np.uint8)*binary_masks[idx] + frames[idx] * (1-binary_masks[idx])
                if comp_frames[idx] is None:
                    comp_frames[idx] = img
                else:
                    comp_frames[idx] = comp_frames[idx].astype(
                        np.float32)*0.5 + img.astype(np.float32)*0.5
    
    comp_ls = []
    for f in range(video_length):
        comp = np.array(comp_frames[f]).astype(
            np.uint8)*binary_masks[f] + frames[f] * (1-binary_masks[f])
        comp_ls.append(comp)
    if out_path:
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), default_fps, (w, h))
        for comp in comp_ls:
            writer.write(cv2.cvtColor(np.array(comp).astype(np.uint8), cv2.COLOR_BGR2RGB))
        writer.release()
        print(f'Finished! output is saved to {out_path}')
    return comp_ls
