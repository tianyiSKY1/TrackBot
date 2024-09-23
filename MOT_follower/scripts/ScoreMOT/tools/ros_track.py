#!/usr/bin/env python3
import argparse
import os
import os.path as osp
import time
import cv2
import torch
import numpy as np



import rospy
import message_filters
from simple_follower.msg import position as PositionMsg
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from loguru import logger

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer




IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("ByteTrack")
    # parser.add_argument(
    #     "demo", default="webcam", help="demo type, eg. image, video and webcam"
    # )
    # parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    # parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # parser.add_argument(
        #"--path", default="./datasets/mot/train/MOT17-05-FRCNN/img1", help="path to images or video"
        # "--path", default="./videos/palace.mp4", help="path to images or video"
    # )
    # parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    # parser.add_argument(
    #     "--save_result",
    #     default=True,
    #     action="store_true",
    #     help="whether to save the inference result of image/video",
    # )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default="/home/rosz/wheeltec/src/simple_follower/scripts/ByteTrack/exps/example/mot/yolox_s_mix_det.py",
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default="/home/rosz/wheeltec/src/simple_follower/scripts/ByteTrack/pretrained/bytetrack_s_mot17.pth.tar", type=str, help="ckpt for eval")
    parser.add_argument(""
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=True,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        trt_file=None,
        decoder=None,
        device=torch.device("cpu"),
        fp16=False
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
            #logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

def imageflow_demo(predictor, vis_folder, current_time, args):
    # cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    cap = cv2.VideoCapture(-1)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    save_folder = os.path.join(
        vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    )
    os.makedirs(save_folder, exist_ok=True)
    # if args.demo == "video":
    #     save_path = os.path.join(save_folder, args.path.split("/")[-1])
    # else:
    #     save_path = os.path.join(save_folder, "camera.mp4")
    save_path = os.path.join(save_folder, "camera.mp4")
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    tracker = BYTETracker(args, frame_rate=30)
    timer = Timer()
    frame_id = 0
    results = []
    while True:
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.inference(frame, timer)
            online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
            timer.toc()
            results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))
            online_im = plot_tracking(img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1,
                                      fps=1. / timer.average_time)
            # if args.save_result:
            vid_writer.write(online_im)
            cv2.imshow("demo", online_im)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += 1



class ImageSubscriber(object):
    def __init__(self, predictor, vis_folder, current_time, args):
        self.bridge = CvBridge()
        self.predictor = predictor
        self.vis_folder = vis_folder
        self.current_time = current_time
        self.args = args
        im_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        dep_sub = message_filters.Subscriber('/camera/depth/image_raw', Image)

        self.timeSynchronizer = message_filters.ApproximateTimeSynchronizer([im_sub, dep_sub], 10, 0.5)
        self.timeSynchronizer.registerCallback(self.mot_callback)

        self.positionPublisher = rospy.Publisher('/object_tracker/current_position', PositionMsg, queue_size=3)


        self.targetUpper = np.array([0, 110, 80])
        self.targetLower = np.array([19, 255, 255])
        self.pictureHeight= 480
        self.pictureWidth = 640
        self.tanVertical = np.tan(0.43196898986859655)
        self.tanHorizontal = np.tan(0.5235987755982988)
        self.lastPoCsition =None
        self.targetDist = 600
        
        rospy.logwarn(self.targetUpper)

        self.tracker = BYTETracker(args, frame_rate=30)
        self.timer = Timer()
        self.frame_id = 0
        self.results = []

        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.on_mouse_click)
        self.selected_id = None
        self.selected_bbox = None
    
    def on_mouse_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for tlwh, id in zip(self.online_tlwhs, self.online_ids):
                if self.isin_bbox(x, y, tlwh):
                    self.selected_id = id
                    self.selected_bbox = tlwh
                    print(self.selected_id)
                    print([x,y])

    def isin_bbox(self, x, y, bbox):
        t, l, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        return l <= y <= l + w and t <= x <= t + h

    def mot_callback(self, image_data, depth_data):
        try:
            bridge = CvBridge()
            frame = bridge.imgmsg_to_cv2(image_data, desired_encoding='bgr8')
            depthFrame = bridge.imgmsg_to_cv2(depth_data, desired_encoding='passthrough')
        except CvBridgeError as e:
            print(e)
        if self.frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(self.frame_id, 1. / max(1e-5, self.timer.average_time)))
        outputs, img_info = self.predictor.inference(frame, self.timer)
        online_targets = self.tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
        self.online_tlwhs = []
        self.online_ids = []
        self.online_scores = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                self.online_tlwhs.append(tlwh)
                self.online_ids.append(tid)
                self.online_scores.append(t.score)
        self.timer.toc()
        self.results.append((self.frame_id + 1, self.online_tlwhs, self.online_ids, self.online_scores))
        online_im = plot_tracking(img_info['raw_img'], self.online_tlwhs, self.online_ids, frame_id=self.frame_id + 1,
                                    fps=1. / self.timer.average_time)
        cv2.imshow("image", online_im)
        ch = cv2.waitKey(1)
        self.frame_id += 1

        newPos = None

        for tlwh, id in zip(self.online_tlwhs, self.online_ids):
            if id == self.selected_id:
                self.selected_bbox = tlwh


        try:
            pos = self.analyseContour(self.selected_bbox, depthFrame)
            if newPos is None:
                newPos = pos
            self.lastPosition = pos
            self.publishPosition(pos)
            return
			#我们没有找到可信的最后一个位置，所以我们只保存了最大的等高线
            self.lastPosition = newPos #we didn't find a plossible last position, so we just save the biggest contour 
        except IndexError:
			# and publish warnings
			# 打印警告信息
            rospy.logwarn('no position found')
            posMsg = PositionMsg(0, 0,self.targetDist)
            self.positionPublisher.publish(posMsg)
    
    def publishPosition(self, pos):
		# calculate the angles from the raw position
		#从原始位置计算角度
        angleX = self.calculateAngleX(pos)
        angleY = self.calculateAngleY(pos)
        posMsg = PositionMsg(angleX, angleY, pos[1])
        self.positionPublisher.publish(posMsg)
    
    def calculateAngleX(self, pos):
        centerX = pos[0][0]
        displacement = 2*centerX/self.pictureWidth-1
        angle = -1*np.arctan(displacement*self.tanHorizontal)
        return angle
    
    def calculateAngleY(self, pos):
        centerY = pos[0][1]
        displacement = 2*centerY/self.pictureHeight-1
        angle = -1*np.arctan(displacement*self.tanVertical)
        return angle
    
    def analyseContour(self, bbox, depthFrame):
		# get a rectangle that completely contains the object
		# 获取完全包含该对象的矩形
        ret = np.asarray(bbox).copy()
        ret[:2] += ret[2:] / 2
        centerRaw = (bbox[0], bbox[1])
        size = (bbox[2], bbox[3])

		# get the center of that rounded to ints (so we can index the image)
        center = np.round(centerRaw).astype(int)

		# find out how far we can go in x/y direction without leaving the object (min of the extension of the bounding rectangle/2 (here 3 for safety)) 
		# 找出在不离开对象的情况下，我们可以在x/y方向上走多远(最小边界矩形的延伸长度/2(为安全起见，此处为3))
        minSize = int(min(size)/3)

		# get all the depth points within this area (that is within the object)
		# 获取该区域内（即对象内）的所有深度点
        depthObject = depthFrame[(center[1]-minSize):(center[1]+minSize), (center[0]-minSize):(center[0]+minSize)]

		# get the average of all valid points (average to have a more reliable distance measure)
		# 获得所有有效点的平均值（平均值以获得更可靠的距离度量）
        depthArray = depthObject[~np.isnan(depthObject)]
        averageDistance = np.mean(depthArray)
        if(averageDistance>400 or averageDistance<3000):
            pass
        else:
            averageDistance=400
        
        if len(depthArray) == 0:
            rospy.logwarn('empty depth array. all depth values are nan')

        return (centerRaw, averageDistance)






def main(exp, args):
    # if not args.experiment_name:
    #     args.experiment_name = exp.exp_name

    output_dir = osp.join(exp.output_dir, exp.exp_name)
    os.makedirs(output_dir, exist_ok=True)

    # if args.save_result:
    vis_folder = osp.join(output_dir, "track_vis")
    os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"
    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model().to(args.device)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = osp.join(output_dir, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()  # to FP16

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = osp.join(output_dir, "model_trt.pth")
        assert osp.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16)
    current_time = time.localtime()

    # imageflow_demo(predictor, vis_folder, current_time, args)

    rospy.init_node('visual_MOT', anonymous=False)
    img_sub = ImageSubscriber(predictor, vis_folder, current_time, args)
    rospy.spin()


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, 'rostrack')

    main(exp, args)
