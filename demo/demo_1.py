# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import json
import os
# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on
from PIL import Image
import tempfile
import time
import warnings

import cv2
import numpy as np
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from maskdino import add_maskdino_config
from predictor import VisualizationDemo


# constants
WINDOW_NAME = "mask2former demo"

coco_json = {
    "info": {
        "description": "Panoptic segmentation dataset",
        "version": "1.0",
        "year": 2024,
        "contributor": "Your Name",
        "date_created": "2024-06-07"
    },
    "licenses": [],
    "images": [],
    "annotations": [],
    "categories": [
        {
            "id": 0,
            "name": "architectural-plans-kBh5",
            "supercategory": "none",
            "isthing": 0
        },
        {
            "id": 1,
            "name": "bath",
            "supercategory": "architectural-plans-kBh5",
            "isthing": 1
        },
        {
            "id": 2,
            "name": "bed_closet",
            "supercategory": "architectural-plans-kBh5",
            "isthing": 1
        },
        {
            "id": 3,
            "name": "bed_pass",
            "supercategory": "architectural-plans-kBh5",
            "isthing": 1
        },
        {
            "id": 4,
            "name": "bedroom",
            "supercategory": "architectural-plans-kBh5",
            "isthing": 1
        },
        {
            "id": 5,
            "name": "chase",
            "supercategory": "architectural-plans-kBh5",
            "isthing": 1
        },
        {
            "id": 6,
            "name": "closet",
            "supercategory": "architectural-plans-kBh5",
            "isthing": 1
        },
        {
            "id": 7,
            "name": "dining",
            "supercategory": "architectural-plans-kBh5",
            "isthing": 1
        },
        {
            "id": 8,
            "name": "entry",
            "supercategory": "architectural-plans-kBh5",
            "isthing": 1
        },
        {
            "id": 9,
            "name": "fireplace",
            "supercategory": "architectural-plans-kBh5",
            "isthing": 1
        },
        {
            "id": 10,
            "name": "flex",
            "supercategory": "architectural-plans-kBh5",
            "isthing": 1
        },
        {
            "id": 11,
            "name": "foyer",
            "supercategory": "architectural-plans-kBh5",
            "isthing": 1
        },
        {
            "id": 12,
            "name": "front_porch",
            "supercategory": "architectural-plans-kBh5",
            "isthing": 1
        },
        {
            "id": 13,
            "name": "garage",
            "supercategory": "architectural-plans-kBh5",
            "isthing": 1
        },
        {
            "id": 14,
            "name": "general",
            "supercategory": "architectural-plans-kBh5",
            "isthing": 1
        },
        {
            "id": 15,
            "name": "hall",
            "supercategory": "architectural-plans-kBh5",
            "isthing": 1
        },
        {
            "id": 16,
            "name": "hall_cased_opening",
            "supercategory": "architectural-plans-kBh5",
            "isthing": 1
        },
        {
            "id": 17,
            "name": "kitchen",
            "supercategory": "architectural-plans-kBh5",
            "isthing": 1
        },
        {
            "id": 18,
            "name": "laundry",
            "supercategory": "architectural-plans-kBh5",
            "isthing": 1
        },
        {
            "id": 19,
            "name": "living",
            "supercategory": "architectural-plans-kBh5",
            "isthing": 1
        },
        {
            "id": 20,
            "name": "master_bed",
            "supercategory": "architectural-plans-kBh5",
            "isthing": 1
        },
        {
            "id": 21,
            "name": "master_closet",
            "supercategory": "architectural-plans-kBh5",
            "isthing": 1
        },
        {
            "id": 22,
            "name": "master_hall",
            "supercategory": "architectural-plans-kBh5",
            "isthing": 1
        },
        {
            "id": 23,
            "name": "master_vestibule",
            "supercategory": "architectural-plans-kBh5",
            "isthing": 1
        },
        {
            "id": 24,
            "name": "mech",
            "supercategory": "architectural-plans-kBh5",
            "isthing": 1
        },
        {
            "id": 25,
            "name": "mudroom",
            "supercategory": "architectural-plans-kBh5",
            "isthing": 1
        },
        {
            "id": 26,
            "name": "office",
            "supercategory": "architectural-plans-kBh5",
            "isthing": 1
        },
        {
            "id": 27,
            "name": "pantry",
            "supercategory": "architectural-plans-kBh5",
            "isthing": 1
        },
        {
            "id": 28,
            "name": "patio",
            "supercategory": "architectural-plans-kBh5",
            "isthing": 1
        },
        {
            "id": 29,
            "name": "portico",
            "supercategory": "architectural-plans-kBh5",
            "isthing": 1
        },
        {
            "id": 30,
            "name": "powder",
            "supercategory": "architectural-plans-kBh5",
            "isthing": 1
        },
        {
            "id": 31,
            "name": "reach_closet",
            "supercategory": "architectural-plans-kBh5",
            "isthing": 1
        },
        {
            "id": 32,
            "name": "reading_nook",
            "supercategory": "architectural-plans-kBh5",
            "isthing": 1
        },
        {
            "id": 33,
            "name": "rear_porch",
            "supercategory": "architectural-plans-kBh5",
            "isthing": 1
        },
        {
            "id": 34,
            "name": "solarium",
            "supercategory": "architectural-plans-kBh5",
            "isthing": 1
        },
        {
            "id": 35,
            "name": "stairs_editor",
            "supercategory": "architectural-plans-kBh5",
            "isthing": 1
        },
        {
            "id": 36,
            "name": "util_hall",
            "supercategory": "architectural-plans-kBh5",
            "isthing": 1
        },
        {
            "id": 37,
            "name": "walk",
            "supercategory": "architectural-plans-kBh5",
            "isthing": 1
        },
        {
            "id": 38,
            "name": "water_closet",
            "supercategory": "architectural-plans-kBh5",
            "isthing": 1
        },
        {
            "id": 39,
            "name": "workshop",
            "supercategory": "architectural-plans-kBh5",
            "isthing": 1
        }
    ],
}


def read_image(path, format="BGR"):
    img = Image.open(path)
    if format == "BGR":
        img = img.convert("RGB")
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return img

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="maskdino demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)
    annotation_id = 0
    if args.input:
        input_path = os.path.expanduser(args.input[0])
        if os.path.isdir(input_path):
            args.input = glob.glob(os.path.join(input_path, "*"))
        else:
            args.input = glob.glob(input_path)
        assert args.input, "The input path(s) was not found"
        idx = 0
        for path in tqdm.tqdm(args.input, disable=not args.output):
            if path[-4:] == "json":
                continue
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            height, width = img.shape[:2]
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)
            panoptic_seg, segments_info = predictions['panoptic_seg']
            panoptic_seg = panoptic_seg.cpu().detach().numpy()
            
            image_info = {
                "file_name": os.path.basename(path),
                "height": height,
                "width": width,
                "id": idx
            }
            idx += 1
            coco_json['images'].append(image_info)
            segments_info_data = []
            unique_ids = np.unique(panoptic_seg)
            for segment_id in unique_ids:
                if segment_id == 0:
                    continue  # Skip the 'nothing' category

                mask = (panoptic_seg == segment_id).astype(np.uint8)
                iscrowd = 0  # or 1, based on your context
                area = int(np.sum(mask))

                # Bounding box
                y_indices, x_indices = np.where(mask)
                bbox = [int(np.min(x_indices)), int(np.min(y_indices)), int(np.max(x_indices) - np.min(x_indices)), int(np.max(y_indices) - np.min(y_indices))]

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                segmentation = []
                for contour in contours:
                    contour = contour.flatten().tolist()
                    segmentation.append(contour)
                    
                segment_info = {
                    "id": int(segment_id),
                    "category_id": int(segment_id),
                    "area": area,
                    "bbox": bbox,
                    "segmentation": segmentation,
                    "iscrowd": iscrowd
                }
                segments_info_data.append(segment_info)
                annotation_id += 1
            coco_json['annotations'].extend(segments_info_data)
            
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            if args.output:
                if os.path.isdir(args.output):
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output.save(out_filename)
                with open("/home/ubuntu/code/MaskDINO/configs/coco/_annotation_file.json", 'w') as f:
                    json.dump(coco_json, f, indent=4)
            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit
                
    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        assert args.output is None, "output not yet supported with --webcam!"
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cam.release()
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)
        codec, file_ext = (
            ("x264", ".mkv") if test_opencv_video_format("x264", ".mkv") else ("mp4v", ".mp4")
        )
        if codec == ".mp4v":
            warnings.warn("x264 codec not available, switching to mp4v")
        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + file_ext
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*codec),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()
