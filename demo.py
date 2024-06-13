# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import os
import argparse
import cv2
from pathlib import Path
from time import time

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


IMG_FORMATS = {"bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm"}  # image suffixes
VID_FORMATS = {"asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "wmv", "webm"}  # video suffixes
FORMATS_HELP_MSG = f"Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}"

MILLISECOND = 1000


def visualise(im, predictions, scale=1.0):
    # draw the predictions on the image.
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=scale)
    out = v.draw_instance_predictions(predictions)
    return out.get_image()[:, :, ::-1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, help='Path to image file/directory.')
    parser.add_argument('--output', type=str, default='runs/demo', help='Path to save directory.')
    parser.add_argument('--thres', type=float, default=0.4, help='Threshold for the model.')
    parser.add_argument('--seg', type=str, default=None, help='Path to red zone segment txt.')
    parser.add_argument('--show', action='store_true', help='Display the annotated image/video in a window.')
    parser.add_argument('--min-sz', type=int, default=800, help='Size of the smallest side of the image \
        during testing. Set to zero to disable resize in testing.')
    parser.add_argument('--max-sz', type=int, default=1333, help='Size of the largest side of the image during testing.')
    parser.add_argument('--original', action='store_true', help='Use original model.')
    args = parser.parse_args()

    source = Path(args.source)
    if os.path.isdir(source):
        file_paths = [source / f for f in os.listdir(args.source)]
    else:
        file_paths = [source]
    save_dir = Path(args.output)
    if not save_dir.exists():
        os.makedirs(save_dir)

    # create detectron2 config
    cfg = get_cfg()
    cfg.MIN_SIZE_TEST = args.min_sz
    cfg.MAX_SIZE_TEST = args.max_sz
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    if args.original:
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
    else:
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x_custom.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.thres  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")

    # default predictor for inference
    predictor = DefaultPredictor(cfg)

    for file_path in file_paths:
        if not ((file_path.suffix[1:].lower() in IMG_FORMATS) or (file_path.suffix[1:].lower() in VID_FORMATS)):
            continue
        filename = file_path.stem

        if file_path.suffix[1:].lower() in IMG_FORMATS:
            im = cv2.imread(file_path.__str__())

            start_time = time()
            outputs = predictor(im)
            inference_time = (time() - start_time) * 1000

            start_time = time()
            im0 = visualise(im, outputs)
            visualise_time = (time() - start_time) * 1000
            cv2.imwrite(str(save_dir / f'{filename}.jpg'), im0)

        if file_path.suffix[1:].lower() in VID_FORMATS:
            cap = cv2.VideoCapture(file_path.__str__())
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            vid_writer = cv2.VideoWriter(str(save_dir / f'{filename}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                         (w, h))
            prev = None
            feat_cached = False
            frame_no = 0
            while cap.isOpened():
                ret, im = cap.read()
                if not ret:
                    break
                frame_no += 1

                start_time = time()
                outputs = predictor(im)
                inference_time = (time() - start_time) * MILLISECOND

                start_time = time()
                predictions = outputs['instances'].to('cpu')
                person_count = predictions.__len__()
                im0 = visualise(im, predictions)
                visualise_time = (time() - start_time) * MILLISECOND

                if args.show:
                    cv2.imshow('Demo', im0)
                    cv2.waitKey(1)
                vid_writer.write(im0)
                prev = im
                print(
                    'Frame {0:{1}d}/{2}: inference {3:5.1f}ms  visualise {4:6.1f}  {5:2d} person detected.'
                    .format(frame_no, len(str(frame_count)), frame_count,
                            inference_time, visualise_time, person_count))
            vid_writer.release()
            cap.release()
