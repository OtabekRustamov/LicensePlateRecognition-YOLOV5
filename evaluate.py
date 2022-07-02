import copy
import os
import pickle
import sys
from collections import Counter
from xml.etree.ElementTree import parse

import cv2
import numpy
import torch
import tqdm

from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
           'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
           'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
           'U', 'V', 'W', 'X', 'Y', 'Z', 'License Plate']
base_path = '../Dataset/PlateUZ'


def draw_boxes(image, boxes, scores, labels, classes):
    for b, l, s in zip(boxes, labels, scores):
        class_id = int(l)
        class_name = classes[class_id]

        x_min, y_min, x_max, y_max = list(map(int, b))
        label = class_name

        ret, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
        cv2.rectangle(image, (x_min, y_max - ret[1] - baseline), (x_min + ret[0], y_max), (0, 255, 0), -1)
        cv2.putText(image, label, (x_min, y_max - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


def inference():
    img_size = 2048
    weights = 'weights/best.pt'

    device = torch.device('cuda')
    half = device.type != 'cpu'
    model = torch.load(weights, map_location=device)['ema'].float().eval()
    if half:
        model.half()
    image = torch.zeros((1, 3, img_size, img_size), device=device)
    _ = model(image.half() if half else image) if device.type != 'cpu' else None
    file_names = []
    with open(f'{base_path}/train.txt') as reader:
        for line in reader.readlines():
            file_names.append(line.rstrip())

    result_dict = {}
    for file_name in tqdm.tqdm(file_names):
        image_path = os.path.join(base_path, 'images', file_name + '.jpg')
        label_path = os.path.join(base_path, 'labels', file_name + '.xml')

        image = cv2.imread(image_path)  # BGR
        img_np = image.copy()
        h0, w0 = image.shape[:2]
        r = img_size / max(h0, w0)
        if r != 1:
            resample = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            image = cv2.resize(image, (int(w0 * r), int(h0 * r)), interpolation=resample)
        h, w = image.shape[:2]
        image, ratio, pad = letterbox(image, img_size, scaleup=False)

        shapes = (h0, w0), ((h / h0, w / w0), pad)
        image = image[:, :, ::-1].transpose(2, 0, 1)  # rgb
        image = numpy.ascontiguousarray(image)

        image = torch.from_numpy(image).to(device)
        image = image.half() if half else image.float()
        image /= 255.0
        if image.ndimension() == 3:
            image = image.unsqueeze(0)

        pred = model(image)[0]
        pred = non_max_suppression(pred, 0.3, 0.1)

        scores = []
        pred_boxes = []
        pred_classes = []
        for det in pred:
            if det is not None and len(det):
                scale_coords(image.shape[1:], det[:, :4], shapes[0], shapes[1])

                for *xy_xy, conf, cls in reversed(det):
                    x_min = int(xy_xy[0])
                    y_min = int(xy_xy[1])
                    x_max = int(xy_xy[2])
                    y_max = int(xy_xy[3])

                    scores.append(float(f'{conf:.8f}'))
                    pred_boxes.append([x_min, y_min, x_max, y_max])
                    pred_classes.append(int(float(str(cls.cpu().numpy()))))
        # scores, pred_boxes, pred_classes
        true_boxes = []
        true_classes = []
        for element in parse(label_path).getroot().iter('object'):
            x_min = int(element.find('bndbox').find('xmin').text)
            y_min = int(element.find('bndbox').find('ymin').text)
            x_max = int(element.find('bndbox').find('xmax').text)
            y_max = int(element.find('bndbox').find('ymax').text)
            true_boxes.append([x_min, y_min, x_max, y_max])
            true_classes.append(CLASSES.index(element.find('name').text))
        draw_boxes(img_np, pred_boxes, scores, pred_classes, CLASSES)
        cv2.imwrite(f'images/{file_name}.jpg', img_np)
        result = {'pred_boxes': pred_boxes,
                  'true_boxes': true_boxes,
                  'true_class': true_classes,
                  'pred_class': pred_classes,
                  'confidence': scores}
        result_dict[f'{file_name}.jpg'] = result
    with open(os.path.join('weights/plate.pickle'), 'wb') as f:
        pickle.dump(result_dict, f)


def compute_ap(rec, pre):
    m_rec = [0]
    [m_rec.append(e) for e in rec]
    m_rec.append(1)
    m_pre = [0]
    [m_pre.append(e) for e in pre]
    m_pre.append(0)
    for i in range(len(m_pre) - 1, 0, -1):
        m_pre[i - 1] = max(m_pre[i - 1], m_pre[i])
    ii = []
    for i in range(len(m_rec) - 1):
        if m_rec[1:][i] != m_rec[0:-1][i]:
            ii.append(i + 1)
    ap = 0
    for i in ii:
        ap = ap + numpy.sum((m_rec[i] - m_rec[i - 1]) * m_pre[i])
    return [ap, m_pre[0:len(m_pre) - 1], m_rec[0:len(m_pre) - 1], ii]


def get_intersection_area(box_a, box_b):
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])
    return (x_b - x_a + 1) * (y_b - y_a + 1)


def is_boxes_intersect(box_a, box_b):
    if box_a[0] > box_b[2]:
        return False  # boxA is right of boxB
    if box_b[0] > box_a[2]:
        return False  # boxA is left of boxB
    if box_a[3] < box_b[1]:
        return False  # boxA is above boxB
    if box_a[1] > box_b[3]:
        return False  # boxA is below boxB
    return True


def get_area(box):
    return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)


def get_union_areas(box_a, box_b, inter_area=None):
    area_a = get_area(box_a)
    area_b = get_area(box_b)
    if inter_area is None:
        inter_area = get_intersection_area(box_a, box_b)
    return float(area_a + area_b - inter_area)


def compute_iou(box_a, box_b):
    # if boxes dont intersect
    if is_boxes_intersect(box_a, box_b) is False:
        return 0
    intersection_area = get_intersection_area(box_a, box_b)
    union = get_union_areas(box_a, box_b, inter_area=intersection_area)
    # intersection over union
    iou = intersection_area / union
    assert iou >= 0
    return iou


def compute_statistics(true_boxes, pred_boxes, confidence,
                       true_class, pred_class, key, threshold=0.25, single=False):
    # list containing metrics (precision, recall, average precision) of each class
    ret = []

    # list with all ground truths (Ex: [imageName,class,confidence=1, (bb coordinates XYX2Y2)])
    true = []

    # list with all detections (Ex: [imageName,class,confidence,(bb coordinates XYX2Y2)])
    pred = []

    # [image_name, class, confidence, [coordinates]]
    for class_name, box in zip(true_class, true_boxes):
        if single:
            true.append([key, 'damage', 1, box])
        else:
            true.append([key, class_name, 1, box])
    for class_name, score, box in zip(pred_class, confidence, pred_boxes):
        if score >= threshold:
            if single:
                pred.append([key, 'damage', score, box])
            else:
                pred.append([key, class_name, score, box])

    if single:
        classes = (['damage'])
    else:
        classes = CLASSES

    # Precision x Recall is obtained individually by each class
    # Loop through by classes
    for ci, c in enumerate(classes):
        num_tp = 0
        num_fp = 0
        # Get only detection of class c
        detections = []
        [detections.append(d) for d in pred if d[1] == c]
        # Get only ground truths of class c
        gts = []
        [gts.append(g) for g in true if g[1] == c]
        num_pos = len(gts)
        # sort detections by decreasing confidence
        detections = sorted(detections, key=lambda conf: conf[2], reverse=True)
        tp = numpy.zeros(len(detections))
        fp = numpy.zeros(len(detections))
        # create dictionary with amount of gts for each image
        det = Counter([cc[0] for cc in gts])

        for key, val in det.items():
            det[key] = numpy.zeros(val)

        # Loop through detections
        for d in range(len(detections)):
            # Find ground truth image
            j_max = None
            gt = [gt for gt in gts if gt[0] == detections[d][0]]
            iou_max = sys.float_info.min
            for j in range(len(gt)):
                iou = compute_iou(detections[d][3], gt[j][3])
                if iou > iou_max:
                    iou_max = iou
                    j_max = j
            # Assign detection as true positive/don't care/false positive
            if iou_max >= 0.1:
                if det[detections[d][0]][j_max] == 0:
                    tp[d] = 1  # count as true positive
                    num_tp += 1
                    det[detections[d][0]][j_max] = 1  # flag as already 'seen'
                else:
                    num_fp += 1
                    fp[d] = 1  # count as false positive
            # - A detected "object" is overlapped with a GT "object" with IOU >= IOUThreshold.
            else:
                num_fp += 1
                fp[d] = 1  # count as false positive
        # compute precision, recall and average precision

        acc_fp = numpy.cumsum(fp)
        acc_tp = numpy.cumsum(tp)
        rec = acc_tp / (num_pos + 1e-10)
        pre = numpy.divide(acc_tp, (acc_fp + acc_tp))

        # Depending on the method, call the right implementation
        [ap, _, _, _] = compute_ap(rec, pre)
        # add class result in the dictionary to be returned
        ret.append({'ap': ap, 'tp': num_tp, 'fp': num_fp, 'fn': num_pos - num_tp})
    return ret


def main():
    if not os.path.exists('weights/plate.pickle'):
        inference()
    with open('weights/plate.pickle', 'rb') as f:
        data = pickle.load(f)
    tp = 0
    fp = 0
    fn = 0
    threshold = 0.3
    test_names = []
    with open(f'{base_path}/train.txt') as reader:
        for line in reader.readlines():
            test_names.append(line.rstrip().split(' ')[0])
    for key, value in data.items():
        true_boxes = value['true_boxes']
        pred_boxes = value['pred_boxes']
        confidence = value['confidence']
        true_class = value['true_class']
        pred_class = value['pred_class']

        for result in compute_statistics(true_boxes, pred_boxes, confidence,
                                         true_class, pred_class, key, threshold, True):
            tp += result['tp']
            fp += result['fp']
            fn += result['fn']

    print(f'F1 Score: {tp / (tp + (fp + fn) / 2.0) * 100:.1f}')


if __name__ == '__main__':
    with torch.no_grad():
        main()
