from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import functools
import json
import os
import tensorflow as tf

from google.protobuf import text_format


from object_detection.protos import input_reader_pb2
from object_detection.protos import model_pb2
from object_detection.protos import pipeline_pb2
from object_detection.protos import train_pb2
import os, sys, shutil
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
import base64
from IPython.display import HTML
from IPython.display import clear_output
from IPython import display
import matplotlib.patches as patches
from matplotlib.pyplot import cm
import time
import cv2
import pickle
import json
import ast
from os.path import join

from glob import glob
from mpl_toolkits.mplot3d import Axes3D
import xml.etree.ElementTree as ET

def disp_video(fname):
    import io
    import base64
    from IPython.display import HTML
    video = io.open(fname, 'r+b').read()
    encoded = base64.b64encode(video)
    return HTML(data='''<video alt="test" width="640" height="480" controls>
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii')))


def get_info_from_DF(frame_no, frames):
    result = []
    temp = frames[frames["frame_no"] == frame_no]
    for i, box in temp.iterrows():
        result.append([int(box["xmin"]), int(box["ymin"]), int(box["xmax"]), int(box["ymax"])])
    return result

def get_info_from_xml(frame_no, frames):
    result = []
    xml_file = '{}/frame{}.xml'.format(frames, frame_no)

    print(xml_file)
    root = ET.parse(xml_file).getroot()
    for obj in root.findall('object'):
        obj_name = obj.find('name').text
        bndbox = obj.find('bndbox')
        left = bndbox.find('xmin').text
        top = bndbox.find('ymin').text
        right = bndbox.find('xmax').text
        bottom = bndbox.find('ymax').text
        result.append([int(left), int(top), int(right), int(bottom)])
    return result


def detect_frames(path_to_graph, path_to_labels,
                  data_folder, video_path, min_index, max_index, frame_rate, threshold, reference_frames):
    # We load the label maps and access category names and their associated indicies
    label_map = label_map_util.load_labelmap(path_to_labels)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=90, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Import a graph by reading it as a string, parsing this string then importing it using the tf.import_graph_def command
    print('Importing graph...')
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_graph, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    # Generate a video object
    # fourcc = cv2.VideoWriter_fourcc('h', '2', '6', '4')
    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    print('Starting session...')
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Define input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            frames_path = data_folder

            num_frames = max_index - min_index

            reference_image = os.listdir(data_folder)[0]
            image = cv2.imread(join(data_folder, reference_image))
            height, width, channels = image.shape
            out = cv2.VideoWriter(video_path, fourcc, frame_rate, (width, height))
            print('Running Inference:')
            for fdx, file_name in \
                    enumerate(sorted(os.listdir(data_folder), key=lambda fname: int(fname.split('.')[0]))):

                if fdx <= min_index or fdx >= max_index:
                    continue;
                image = cv2.imread(join(frames_path, file_name))
                image_np = np.array(image)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                bboxes = get_info_from_xml(int(file_name.split('.')[0]), reference_frames)
                # Actual detection.
                tic = time.time()
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                toc = time.time()
                t_diff = toc - tic
                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=2,
                    min_score_thresh=threshold)

                cv2.putText(image, 'frame: {}'.format(file_name), (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

                for bbox in bboxes:
                    cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
                    cv2.putText(image, 'FPS (GPU Inference) %.2f' % round(1 / t_diff, 2), (30, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

                prog = 'Completed %.2f %%' % (100 * float(fdx - min_index + 1) / num_frames)
                print('\r{}'.format(prog), end="")
                cv2.imwrite("temp/{}.jpg".format(fdx), image)
                out.write(image)
        out.release()


def detect_frames_for_comparison(path_to_graph, path_to_labels,
                                 data_folder, min_index, max_index, reference_frames):
    # We load the label maps and access category names and their associated indicies
    label_map = label_map_util.load_labelmap(path_to_labels)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=1, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Import a graph by reading it as a string, parsing this string then importing it using the tf.import_graph_def command
    print('Importing graph...')
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_graph, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    # Generate a video object

    print('Starting session...')
    output = []
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Define input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            frames_path = data_folder
            xml_path = join(data_folder, 'xml')
            num_frames = max_index - min_index
            reference_image = os.listdir(data_folder)[0]
            image = cv2.imread(join(data_folder, reference_image))
            height, width, channels = image.shape
            print('Running Inference:')
            for fdx, file_name in \
                    enumerate(sorted(os.listdir(data_folder), key=lambda fname: int(fname.split('.')[0]))):
                if fdx <= min_index or fdx >= max_index:
                    continue;
                image = cv2.imread(join(frames_path, file_name))
                image_np = np.array(image)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                bboxes = get_info_from_DF(int(file_name.split(".")[0]), reference_frames)
                # Actual detection.
                tic = time.time()
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                toc = time.time()
                t_diff = toc - tic
                fps = 1 / t_diff

                boxes = np.squeeze(boxes)
                classes = np.squeeze(classes)
                scores = np.squeeze(scores)

                vis_util.visualize_boxes_and_labels_on_image_array(
                    image,
                    boxes,
                    classes.astype(np.int32),
                    scores,
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=2,
                    min_score_thresh=0.5)

                cv2.imwrite(join('/dli/dli-v3/iv05/data/temp', file_name), image)
                prog = '\rCompleted %.2f %%' % (100 * float(fdx - min_index + 1) / num_frames)
                print('{}'.format(prog), end="")
                boxes = np.array([(i[0] * height, i[1] * width, i[2] * height, i[3] * width) for i in boxes])
                output.append((bboxes, (boxes, scores, classes, num, fps)))

    return output


# function to compute the intersection over union of these two bounding boxes
def bbox_IoU(A, B):
    # A = list(ymin,xmin,ymax,xmax)
    # B = list(ymin,xmin,ymax,xmax) - (xmin, ymin, xmax, ymax)
    # assign for readability
    yminA, xminA, ymaxA, xmaxA = A
    xminB, yminB, xmaxB, ymaxB = B

    # figure out the intersecting rectangle coordinates
    xminI = max(xminA, xminB)
    yminI = max(yminA, yminB)
    xmaxI = min(xmaxA, xmaxB)
    ymaxI = min(ymaxA, ymaxB)

    # compute the width and height of the intereseting rectangle
    wI = xmaxI - xminI
    hI = ymaxI - yminI

    # compute the area of intersection rectangle (enforce area>=0)
    areaI = max(0, wI) * max(0, hI)

    # compute areas of the input bounding boxes
    areaA = (xmaxA - xminA) * (ymaxA - yminA)
    areaB = (xmaxB - xminB) * (ymaxB - yminB)

    # if intersecting area is zero, we're done (avoids IoU=0/0 also)
    if areaI == 0: return 0, areaI, areaA, areaB

    # finally, compute and return the intersection over union
    return areaI / (areaA + areaB - areaI), areaI, areaA, areaB


def get_configs_from_pipeline_file(FLAGS):
    """Reads training configuration from a pipeline_pb2.TrainEvalPipelineConfig.

    Reads training config from file specified by pipeline_config_path flag.

    Returns:
      model_config: model_pb2.DetectionModel
      train_config: train_pb2.TrainConfig
      input_config: input_reader_pb2.InputReader
    """
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.gfile.GFile(FLAGS.pipeline_config_path, 'r') as f:
        text_format.Merge(f.read(), pipeline_config)

    model_config = pipeline_config.model
    train_config = pipeline_config.train_config
    input_config = pipeline_config.train_input_reader

    return model_config, train_config, input_config


def get_configs_from_multiple_files(FLAGS):
    """Reads training configuration from multiple config files.

    Reads the training config from the following files:
      model_config: Read from --model_config_path
      train_config: Read from --train_config_path
      input_config: Read from --input_config_path

    Returns:
      model_config: model_pb2.DetectionModel
      train_config: train_pb2.TrainConfig
      input_config: input_reader_pb2.InputReader
    """
    train_config = train_pb2.TrainConfig()
    with tf.gfile.GFile(FLAGS.train_config_path, 'r') as f:
        text_format.Merge(f.read(), train_config)

    model_config = model_pb2.DetectionModel()
    with tf.gfile.GFile(FLAGS.model_config_path, 'r') as f:
        text_format.Merge(f.read(), model_config)

    input_config = input_reader_pb2.InputReader()
    with tf.gfile.GFile(FLAGS.input_config_path, 'r') as f:
        text_format.Merge(f.read(), input_config)

    return model_config, train_config, input_config
