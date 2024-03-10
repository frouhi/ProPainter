import sys
sys.path.append("../../")

import os
import json
import time
import psutil
import argparse
from glob import glob

import cv2
import torch
import torchvision
import numpy as np
import gradio as gr

from tools.painter import mask_painter
from track_anything import TrackingAnything

from model.misc import get_device
from utils.download_util import load_file_from_url

from STTN.inference import run as sttn_inference
from STTN.inference import load_model as load_sttn_model


def parse_augment():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--sam_model_type', type=str, default="vit_h")
    parser.add_argument('--port', type=int, default=8000, help="only useful when running gradio applications")  
    parser.add_argument('--mask_save', default=False)
    parser.add_argument("--data_root_path", required=True)
    args = parser.parse_args()
    
    if not args.device:
        args.device = str(get_device())

    return args

def get_video_path_generator(only_if_has_mask=False):
    video_paths = glob(f"{args.data_root_path}/training_session_*/*.mp4")
    if only_if_has_mask:
        video_paths = [
            vp for vp in video_paths if \
                os.path.exists(vp.replace(".mp4", ".npy")) and \
                not os.path.exists(vp.replace(".mp4", "_inpainted.mp4"))]
    else:
        # in this case we'd like the ones without a mask (so we can generate a mask for them)
        video_paths = [vp for vp in video_paths if not os.path.exists(vp.replace(".mp4", ".npy"))]
    for video_path in video_paths:
        yield video_path

# convert points input to prompt state
def get_prompt(click_state, click_input):
    inputs = json.loads(click_input)
    points = click_state[0]
    labels = click_state[1]
    for input in inputs:
        points.append(input[:2])
        labels.append(input[2])
    click_state[0] = points
    click_state[1] = labels
    prompt = {
        "prompt_type":["click"],
        "input_point":click_state[0],
        "input_label":click_state[1],
        "multimask_output":"True",
    }
    return prompt

# extract frames from upload video
def get_frames_from_video(video_state):
    """
    Args:
        video_path:str
        timestamp:float64
    Return 
        [[0:nearest_frame], [nearest_frame:], nearest_frame]
    """
    video_path = next(video_path_generator)
    frames = []
    user_name = time.time()
    operation_log = [("[Must Do]", "Click image"), (": Video uploaded! Try to click the image shown in step2 to add masks.\n", None)]
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                current_memory_usage = psutil.virtual_memory().percent
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if current_memory_usage > 90:
                    operation_log = [("Memory usage is too high (>90%). Stop the video extraction. Please reduce the video resolution or frame rate.", "Error")]
                    print("Memory usage is too high (>90%). Please reduce the video resolution or frame rate.")
                    break
            else:
                break
    except (OSError, TypeError, ValueError, KeyError, SyntaxError) as e:
        print("read_frame_source:{} error. {}\n".format(video_path, str(e)))
    image_size = (frames[0].shape[0],frames[0].shape[1]) 
    # initialize video_state
    video_state = {
        "user_name": user_name,
        "video_name": os.path.split(video_path)[-1],
        "video_path": video_path,
        "origin_images": frames,
        "painted_images": frames.copy(),
        "masks": [np.zeros((frames[0].shape[0],frames[0].shape[1]), np.uint8)]*len(frames),
        "logits": [None]*len(frames),
        "select_frame_number": 0,
        "fps": fps
        }
    interactive_state = gr.State({
        "inference_times": 0,
        "negative_click_times" : 0,
        "positive_click_times": 0,
        "mask_save": args.mask_save,
        "multi_mask": {
            "mask_names": [],
            "masks": []
        },
        "track_end_number": None,
        }
    )
    video_info = "Video Name: {},\nFPS: {},\nTotal Frames: {},\nImage Size:{}".format(video_state["video_name"], round(video_state["fps"], 0), len(frames), image_size)
    model.samcontroler.sam_controler.reset_image() 
    model.samcontroler.sam_controler.set_image(video_state["origin_images"][0])
    return video_state, video_info, video_state["origin_images"][0], gr.update(visible=True, maximum=len(frames), value=1), gr.update(visible=True, maximum=len(frames), value=len(frames)), \
                        interactive_state, gr.update(visible=True), \
                        gr.update(visible=True), gr.update(visible=True), \
                        gr.update(visible=True), gr.update(visible=True), \
                        gr.update(visible=True), gr.update(visible=True), \
                        gr.update(visible=True), gr.update(visible=True), \
                        gr.update(visible=True), gr.update(visible=True, choices=[], value=[]), \
                        gr.update(visible=True, value=operation_log), gr.update(visible=True, value=operation_log)

# get the select frame from gradio slider
def select_template(image_selection_slider, video_state, interactive_state, mask_dropdown):

    # images = video_state[1]
    image_selection_slider -= 1
    video_state["select_frame_number"] = image_selection_slider

    # once select a new template frame, set the image in sam

    model.samcontroler.sam_controler.reset_image()
    model.samcontroler.sam_controler.set_image(video_state["origin_images"][image_selection_slider])

    operation_log = [("",""), ("Select tracking start frame {}. Try to click the image to add masks for tracking.".format(image_selection_slider),"Normal")]

    return video_state["painted_images"][image_selection_slider], video_state, interactive_state, operation_log, operation_log

# set the tracking end frame
def get_end_number(track_pause_number_slider, video_state, interactive_state):
    interactive_state["track_end_number"] = track_pause_number_slider
    operation_log = [("",""),("Select tracking finish frame {}.Try to click the image to add masks for tracking.".format(track_pause_number_slider),"Normal")]

    return video_state["painted_images"][track_pause_number_slider],interactive_state, operation_log, operation_log

# use sam to get the mask
def sam_refine(video_state, point_prompt, click_state, interactive_state, evt:gr.SelectData):
    """
    Args:
        template_frame: PIL.Image
        point_prompt: flag for positive or negative button click
        click_state: [[points], [labels]]
    """
    if point_prompt == "Positive":
        coordinate = "[[{},{},1]]".format(evt.index[0], evt.index[1])
        interactive_state["positive_click_times"] += 1
    else:
        coordinate = "[[{},{},0]]".format(evt.index[0], evt.index[1])
        interactive_state["negative_click_times"] += 1
    
    # prompt for sam model
    model.samcontroler.sam_controler.reset_image()
    model.samcontroler.sam_controler.set_image(video_state["origin_images"][video_state["select_frame_number"]])
    prompt = get_prompt(click_state=click_state, click_input=coordinate)

    mask, logit, painted_image = model.first_frame_click( 
                                                      image=video_state["origin_images"][video_state["select_frame_number"]], 
                                                      points=np.array(prompt["input_point"]),
                                                      labels=np.array(prompt["input_label"]),
                                                      multimask=prompt["multimask_output"],
                                                      )
    video_state["masks"][video_state["select_frame_number"]] = mask
    video_state["logits"][video_state["select_frame_number"]] = logit
    video_state["painted_images"][video_state["select_frame_number"]] = painted_image

    operation_log = [("[Must Do]", "Add mask"), (": add the current displayed mask for video segmentation.\n", None),
                     ("[Optional]", "Remove mask"), (": remove all added masks.\n", None),
                     ("[Optional]", "Clear clicks"), (": clear current displayed mask.\n", None),
                     ("[Optional]", "Click image"), (": Try to click the image shown in step2 if you want to generate more masks.\n", None)]
    return painted_image, video_state, interactive_state, operation_log, operation_log

def add_multi_mask(video_state, interactive_state, mask_dropdown):
    try:
        mask = video_state["masks"][video_state["select_frame_number"]]
        interactive_state["multi_mask"]["masks"].append(mask)
        interactive_state["multi_mask"]["mask_names"].append("mask_{:03d}".format(len(interactive_state["multi_mask"]["masks"])))
        mask_dropdown.append("mask_{:03d}".format(len(interactive_state["multi_mask"]["masks"])))
        select_frame, _, _ = save_and_show_mask(video_state, interactive_state, mask_dropdown)
        operation_log = [("",""),("Added a mask, use the mask select for target tracking or inpainting.","Normal")]
    except:
        operation_log = [("Please click the image in step2 to generate masks.", "Error"), ("","")]
    return interactive_state, gr.update(choices=interactive_state["multi_mask"]["mask_names"], value=mask_dropdown), select_frame, [[],[]], operation_log, operation_log

def clear_click(video_state, click_state):
    click_state = [[],[]]
    template_frame = video_state["origin_images"][video_state["select_frame_number"]]
    operation_log = [("",""), ("Cleared points history and refresh the image.","Normal")]
    return template_frame, click_state, operation_log, operation_log

def remove_multi_mask(interactive_state, mask_dropdown):
    interactive_state["multi_mask"]["mask_names"]= []
    interactive_state["multi_mask"]["masks"] = []

    operation_log = [("",""), ("Remove all masks. Try to add new masks","Normal")]
    return interactive_state, gr.update(choices=[],value=[]), operation_log, operation_log

def save_and_show_mask(video_state, interactive_state, mask_dropdown):
    save_mask(video_state, interactive_state, mask_dropdown)
    mask_dropdown.sort()
    select_frame = video_state["origin_images"][video_state["select_frame_number"]]
    for i in range(len(mask_dropdown)):
        mask_number = int(mask_dropdown[i].split("_")[1]) - 1
        mask = interactive_state["multi_mask"]["masks"][mask_number]
        select_frame = mask_painter(select_frame, mask.astype('uint8'), mask_color=mask_number+2)
    
    operation_log = [("",""), ("Added masks {}. If you want to do the inpainting with current masks, please go to step3, and click the Tracking button first and then Inpainting button.".format(mask_dropdown),"Normal")]
    return select_frame, operation_log, operation_log


def save_mask(video_state, interactive_state, mask_dropdown):
    if interactive_state["multi_mask"]["masks"]:
        if len(mask_dropdown) == 0:
            mask_dropdown = ["mask_001"]
        mask_dropdown.sort()
        template_mask = interactive_state["multi_mask"]["masks"][int(mask_dropdown[0].split("_")[1]) - 1] * (int(mask_dropdown[0].split("_")[1]))
        for i in range(1,len(mask_dropdown)):
            mask_number = int(mask_dropdown[i].split("_")[1]) - 1 
            template_mask = np.clip(template_mask+interactive_state["multi_mask"]["masks"][mask_number]*(mask_number+1), 0, mask_number+1)
        video_state["masks"][video_state["select_frame_number"]]= template_mask
    else:      
        template_mask = video_state["masks"][video_state["select_frame_number"]]
    with open(video_state["video_path"].replace("mp4", "npy"), 'wb') as f:
        np.save(f, template_mask)
    print("mask is saved")


def load_mask(video_state):
    with open(video_state["video_path"].replace("mp4", "npy"), 'rb') as f:
        mask = np.load(f)
    video_state["masks"][0] = mask
    return mask


def get_template_mask(video_state, interactive_state, mask_dropdown, chunk_id, chunk_size=80):
    """
        for the first chunk, we have to manually select the mask!
        TODO: explore using masks from other videos! 
        NOTE: the template is the "first frame annotation" used for tracking!
        NOTE: we output chunk_size + 1, but the last one is just used for the next chunk basically
    """
    if chunk_id > 0:
        return video_state["masks"][chunk_id*chunk_size]
    template_mask = load_mask(video_state)
    return template_mask


# tracking vos
def vos_tracking_video(video_state, interactive_state, mask_dropdown, chunk_id, chunk_size=80):
    # operation_log = [("",""), ("Tracking finished! Try to click the Inpainting button to get the inpainting result.","Normal")]
    model.cutie.clear_memory()
    start_idx = chunk_id*chunk_size
    end_idx = (chunk_id+1)*chunk_size+1
    following_frames = video_state["origin_images"][start_idx:end_idx] # we add one because this is the output

    template_mask = get_template_mask(video_state, interactive_state, mask_dropdown, chunk_id, chunk_size)
    fps = video_state["fps"]

    # operation error
    if len(np.unique(template_mask))==1:
        template_mask[0][0]=1
        operation_log = [("Please add at least one mask to track by clicking the image in step2.","Error"), ("","")]
        # return video_output, video_state, interactive_state, operation_error
    masks, logits, painted_images = model.generator(images=following_frames, template_mask=template_mask)
    # clear GPU memory
    model.cutie.clear_memory()

    video_state["masks"][start_idx:end_idx] = masks
    video_state["logits"][start_idx:end_idx] = logits
    video_state["painted_images"][start_idx:end_idx] = painted_images

    print(f"tracking for chunk {chunk_id} is done")
    return video_state, interactive_state

# inpaint 
def inpaint_video(
        video_state, 
        resize_ratio_number, 
        dilate_radius_number, 
        raft_iter_number, 
        subvideo_length_number, 
        neighbor_length_number, 
        ref_stride_number, 
        chunk_id, 
        chunk_size=80):
    # operation_log = [("",""), ("Inpainting finished!","Normal")]
    start_idx = chunk_id*chunk_size
    # NOTE: we don't add one because there's no need to create an extra frame
    # having that said, TODO: explore if generating the extra frame helps robustness
    end_idx = (chunk_id+1)*chunk_size
    frames = np.asarray(video_state["origin_images"][start_idx:end_idx])
    # fps = video_state["fps"]
    inpaint_masks = np.asarray(video_state["masks"][start_idx:end_idx])
    
    # inpaint for videos
    inpainted_frames = model.baseinpainter.inpaint(frames, 
                                                   inpaint_masks, 
                                                   ratio=resize_ratio_number, 
                                                   dilate_radius=dilate_radius_number,
                                                   raft_iter=raft_iter_number,
                                                   subvideo_length=subvideo_length_number, 
                                                   neighbor_length=neighbor_length_number, 
                                                   ref_stride=ref_stride_number)   # numpy array, T, H, W, 3
    print(f"inpainting for chunk {chunk_id} is done")
    return inpainted_frames


def track_and_inpaint(
        video_state, 
        interactive_state,
        resize_ratio_number, 
        dilate_radius_number, 
        raft_iter_number, 
        subvideo_length_number, 
        neighbor_length_number, 
        ref_stride_number,
        mask_dropdown,
        chunk_size=80,
        inpainting_model="sttn"):
    """
        the tracking and inpainting are done together
        we chunk the frames, and perform tracking and inpainting on each chunk
        at the end, we put them together and generate the final video
        TODO: There might be a benefit for recoverability and memory management to cache chunk results/store to memory
        TODO: *** IMPORTANT *** we'd like to generate multiple masks together in one session, 
            and do the inpainting for all of them together in the background
            instead of having to generate mask for each video, inpaint, and repeat (which is very inefficient)
    """
    operation_log = [("",""), ("Chunked Tracking and Inpainting finished!","Normal")]
    frame_count = len(video_state["origin_images"])
    fps = video_state["fps"]
    inpainted_frames = []
    model = load_sttn_model() if inpainting_model == "sttn" else None
    for chunk_id in np.array(range(0, frame_count, chunk_size))//chunk_size:
        # step 1: track
        video_state, interactive_state = vos_tracking_video(video_state, interactive_state, mask_dropdown, chunk_id, chunk_size)
        torch.cuda.empty_cache()
        # step 2: inpaint
        if inpainting_model == "sttn":
            start_idx = chunk_id*chunk_size
            end_idx = (chunk_id+1)*chunk_size
            frames = np.asarray(video_state["origin_images"][start_idx:end_idx])
            inpaint_masks = np.asarray(video_state["masks"][start_idx:end_idx])
            inpainted_frames_chunk = sttn_inference(model, frames, inpaint_masks)
        else:
            inpainted_frames_chunk = inpaint_video(
                video_state, 
                resize_ratio_number, 
                dilate_radius_number, 
                raft_iter_number, 
                subvideo_length_number, 
                neighbor_length_number, 
                ref_stride_number, 
                chunk_id, 
                chunk_size
            )
        inpainted_frames.extend(inpainted_frames_chunk)
    print("all chunks are done")
    # step 3: when all chunks are done, generate video from frames
    video_output = generate_video_from_frames(inpainted_frames, output_path=video_state["video_path"].replace(".mp4", "_inpainted.mp4"), fps=fps) # import video_input to name the output video
    return video_output, operation_log, operation_log


def track_and_inpaint_all(
        video_state, 
        interactive_state,
        resize_ratio_number, 
        dilate_radius_number, 
        raft_iter_number, 
        subvideo_length_number, 
        neighbor_length_number, 
        ref_stride_number,
        mask_dropdown,
        chunk_size=80,
        inpainting_model="ProPainter"):
    global video_path_generator
    vid_count = len(list(get_video_path_generator(only_if_has_mask=True)))
    progress = gr.Progress()
    video_path_generator = get_video_path_generator(only_if_has_mask=True)
    for i in progress.tqdm(range(vid_count)):
        video_state, video_info, template_frame, image_selection_slider, track_pause_number_slider, \
            _, _, point_prompt, clear_button_click, Add_mask_button, template_frame, track_and_inpaint_button, \
            inpaiting_video_output, remove_mask_button, step2_title, step3_title,mask_dropdown, run_status, \
            run_status2 = get_frames_from_video(video_state)
        inpaiting_video_output, run_status, run_status2 = track_and_inpaint(video_state, interactive_state, resize_ratio_number, dilate_radius_number, raft_iter_number, subvideo_length_number, neighbor_length_number, ref_stride_number, mask_dropdown, chunk_size, inpainting_model)
    return inpaiting_video_output, run_status, run_status2


# generate video after vos inference
def generate_video_from_frames(frames, output_path, fps=30):
    """
    Generates a video from a list of frames.
    
    Args:
        frames (list of numpy arrays): The frames to include in the video.
        output_path (str): The path to save the generated video.
        fps (int, optional): The frame rate of the output video. Defaults to 30. TODO: make sure this is the same as the input
    """
    frames = torch.from_numpy(np.asarray(frames))
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    torchvision.io.write_video(output_path, frames, fps=fps, video_codec="libx264")
    return output_path


# args, defined in track_anything.py
args = parse_augment()
video_path_generator = get_video_path_generator()
pretrain_model_url = 'https://github.com/sczhou/ProPainter/releases/download/v0.1.0/'
sam_checkpoint_url_dict = {
    'vit_h': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    'vit_l': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    'vit_b': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
}
checkpoint_fodler = os.path.join('..', '..', 'weights')

sam_checkpoint = load_file_from_url(sam_checkpoint_url_dict[args.sam_model_type], checkpoint_fodler)
cutie_checkpoint = load_file_from_url(os.path.join(pretrain_model_url, 'cutie-base-mega.pth'), checkpoint_fodler)
propainter_checkpoint = load_file_from_url(os.path.join(pretrain_model_url, 'ProPainter.pth'), checkpoint_fodler)
raft_checkpoint = load_file_from_url(os.path.join(pretrain_model_url, 'raft-things.pth'), checkpoint_fodler)
flow_completion_checkpoint = load_file_from_url(os.path.join(pretrain_model_url, 'recurrent_flow_completion.pth'), checkpoint_fodler)

# initialize sam, cutie, propainter models
model = TrackingAnything(sam_checkpoint, cutie_checkpoint, propainter_checkpoint, raft_checkpoint, flow_completion_checkpoint, args)


title = r"""<h1 align="center">ProPainter: Improving Propagation and Transformer for Video Inpainting</h1>"""

css = """
.gradio-container {width: 85% !important}
.gr-monochrome-group {border-radius: 5px !important; border: revert-layer !important; border-width: 2px !important; color: black !important}
button {border-radius: 8px !important;}
.add_button {background-color: #4CAF50 !important;}
.remove_button {background-color: #f44336 !important;}
.mask_button_group {gap: 10px !important;}
.video {height: 300px !important;}
.image {height: 300px !important;}
.video .wrap.svelte-lcpz3o {display: flex !important; align-items: center !important; justify-content: center !important;}
.video .wrap.svelte-lcpz3o > :first-child {height: 100% !important;}
.margin_center {width: 50% !important; margin: auto !important;}
.jc_center {justify-content: center !important;}
"""

with gr.Blocks(theme=gr.themes.Monochrome(), css=css) as iface:
    click_state = gr.State([[],[]])

    interactive_state = gr.State({
        "inference_times": 0,
        "negative_click_times" : 0,
        "positive_click_times": 0,
        "mask_save": args.mask_save,
        "multi_mask": {
            "mask_names": [],
            "masks": []
        },
        "track_end_number": None,
        }
    )

    video_state = gr.State(
        {
        "user_name": "",
        "video_name": "",
        "video_path": "",
        "origin_images": None,
        "painted_images": None,
        "masks": None,
        "inpaint_masks": None,
        "logits": None,
        "select_frame_number": 0,
        "fps": 30
        }
    )

    gr.Markdown(title)

    with gr.Group(elem_classes="gr-monochrome-group"):
        with gr.Row():
            with gr.Accordion('ProPainter Parameters', open=False):
                with gr.Row():
                    resize_ratio_number = gr.Slider(label='Resize ratio',
                                            minimum=0.01,
                                            maximum=1.0,
                                            step=0.01,
                                            value=0.4)
                    raft_iter_number = gr.Slider(label='Iterations for RAFT inference.',
                                            minimum=5,
                                            maximum=20,
                                            step=1,
                                            value=20,)
                with gr.Row():
                    dilate_radius_number = gr.Slider(label='Mask dilation for video and flow masking.',
                                            minimum=0,
                                            maximum=10,
                                            step=1,
                                            value=8,)

                    subvideo_length_number = gr.Slider(label='Length of sub-video for long video inference.',
                                            minimum=40,
                                            maximum=200,
                                            step=1,
                                            value=80,)
                with gr.Row():
                    neighbor_length_number = gr.Slider(label='Length of local neighboring frames.',
                                            minimum=5,
                                            maximum=20,
                                            step=1,
                                            value=10,)
                    
                    ref_stride_number = gr.Slider(label='Stride of global reference frames.',
                                            minimum=5,
                                            maximum=20,
                                            step=1,
                                            value=10,)
  
    with gr.Column():
        # input video
        gr.Markdown("## Step1: Upload video")
        with gr.Row(equal_height=True):
            with gr.Column(scale=2):      
                extract_frames_button = gr.Button(value="Load Video", interactive=True, variant="primary") 
            with gr.Column(scale=2):
                run_status = gr.HighlightedText(value=[("",""), ("Try to upload your video and click the Get video info button to get started!", "Normal")],
                                                color_map={"Normal": "green", "Error": "red", "Clear clicks": "gray", "Add mask": "green", "Remove mask": "red"})
                video_info = gr.Textbox(label="Video Info")
                
        
        # add masks
        step2_title = gr.Markdown("---\n## Step2: Add masks", visible=False)
        with gr.Row(equal_height=True):
            with gr.Column(scale=2):
                template_frame = gr.Image(type="pil",interactive=True, elem_id="template_frame", visible=False, elem_classes="image")
                image_selection_slider = gr.Slider(minimum=1, maximum=100, step=1, value=1, label="Track start frame", visible=False)
                track_pause_number_slider = gr.Slider(minimum=1, maximum=100, step=1, value=1, label="Track end frame", visible=False)
            with gr.Column(scale=2, elem_classes="jc_center"):
                run_status2 = gr.HighlightedText(value=[("",""), ("Try to upload your video and click the Get video info button to get started!", "Normal")],
                                                color_map={"Normal": "green", "Error": "red", "Clear clicks": "gray", "Add mask": "green", "Remove mask": "red"})
                with gr.Row():
                    with gr.Column(scale=2, elem_classes="mask_button_group"):
                        clear_button_click = gr.Button(value="Clear clicks", interactive=True, visible=False)
                        remove_mask_button = gr.Button(value="Remove mask", interactive=True, visible=False, elem_classes="remove_button")
                        Add_mask_button = gr.Button(value="Add mask", interactive=True, visible=False, elem_classes="add_button")
                    point_prompt = gr.Radio(
                        choices=["Positive", "Negative"],
                        value="Positive",
                        label="Point prompt",
                        interactive=True,
                        visible=False,
                        min_width=100,
                        scale=1)
                mask_dropdown = gr.Dropdown(multiselect=True, value=[], label="Mask selection", info=".", visible=False)
            
        # output video
        step3_title = gr.Markdown("---\n## Step3: Track masks and get the inpainting result", visible=False)
        with gr.Row(equal_height=True):
            inpaiting_video_output = gr.Video(visible=True, elem_classes="video")
            track_and_inpaint_button = gr.Button(value="Track & Inpaint All", visible=True, elem_classes="margin_center")

    # first step: get the video information 
    extract_frames_button.click( 
        fn=get_frames_from_video,
        inputs=[
            video_state
        ],
        outputs=[video_state, video_info, template_frame,
                 image_selection_slider, track_pause_number_slider, interactive_state, mask_dropdown, point_prompt, clear_button_click, Add_mask_button, template_frame,
                 track_and_inpaint_button, inpaiting_video_output, remove_mask_button, step2_title, step3_title,mask_dropdown, run_status, run_status2]
    )   

    # second step: select images from slider
    image_selection_slider.release(fn=select_template, 
                                   inputs=[image_selection_slider, video_state, interactive_state], 
                                   outputs=[template_frame, video_state, interactive_state, run_status, run_status2], api_name="select_image")
    track_pause_number_slider.release(fn=get_end_number, 
                                   inputs=[track_pause_number_slider, video_state, interactive_state], 
                                   outputs=[template_frame, interactive_state, run_status, run_status2], api_name="end_image")
    
    # click select image to get mask using sam
    template_frame.select(
        fn=sam_refine,
        inputs=[video_state, point_prompt, click_state, interactive_state],
        outputs=[template_frame, video_state, interactive_state, run_status, run_status2]
    )

    # add different mask
    Add_mask_button.click(
        fn=add_multi_mask,
        inputs=[video_state, interactive_state, mask_dropdown],
        outputs=[interactive_state, mask_dropdown, template_frame, click_state, run_status, run_status2]
    )

    remove_mask_button.click(
        fn=remove_multi_mask,
        inputs=[interactive_state, mask_dropdown],
        outputs=[interactive_state, mask_dropdown, run_status, run_status2]
    )

    # chunked tracking and inpainting
    track_and_inpaint_button.click(
        fn=track_and_inpaint_all,
        inputs=[video_state, interactive_state, resize_ratio_number, dilate_radius_number, raft_iter_number, subvideo_length_number, neighbor_length_number, ref_stride_number, mask_dropdown],
        outputs=[inpaiting_video_output, run_status, run_status2]
    )

    # click to get mask
    mask_dropdown.change(
        fn=save_and_show_mask,
        inputs=[video_state, interactive_state, mask_dropdown],
        outputs=[template_frame, run_status, run_status2]
    )
    
    # points clear
    clear_button_click.click(
        fn = clear_click,
        inputs = [video_state, click_state,],
        outputs = [template_frame,click_state, run_status, run_status2],
    )

iface.queue()
iface.launch(debug=True, share=True)
