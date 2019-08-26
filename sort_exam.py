from utils.helper import detect_frames, disp_video, get_info_from_xml
import pandas as pd
import tensorflow as tf
import numpy as np
import time
from utils.helper import join
import os
import cv2
from sort import *

PATH_TO_INFERENCE_GRAPH= 'export/frozen_inference_graph.pb'
PATH_TO_LABELS = 'label_map.pbtxt'
PATH_TO_DATA = '/G/temp/full'
VIDEO_OUT_PATH = 'temp/output.mp4'
ANNOTATION_PATH = '/G/Temp/annotations'

# detect_frames(PATH_TO_INFERENCE_GRAPH, PATH_TO_LABELS, PATH_TO_DATA, VIDEO_OUT_PATH,0, 1467, 5, 0.5, ANNOTATION_PATH)
# disp_video(VIDEO_OUT_PATH)
# print(get_info_from_xml(int(1), ANNOTATION_PATH))

def inference_frames(path_to_graph, path_to_labels,
                     data_folder, min_index, max_index, threshold):
    # reading the original lables
    # original = pd.read_csv(config['Path_To_DF_File'], converters={2: ast.literal_eval})
    # original = original[original.apply(lambda x: x['outside'] == 0, axis=1)]

    # Import a graph by reading it as a string, parsing this string then importing it using the tf.import_graph_def command
    print('Importing graph...')
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_graph, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    # Use temp list of dictionaries to hold output data
    pre_track_data = []

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

            counter = 1

            for fdx, file_name in \
                    enumerate(sorted(os.listdir(data_folder), key=lambda fname: int(fname.split('.')[0]))):
                if int(file_name.split(".")[0]) <= min_index or int(file_name.split(".")[0]) >= max_index:
                    continue;
                counter += 1
                image = cv2.imread(join(frames_path, file_name))
                image_np = np.array(image)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # bboxes = get_info_from_DF(int(file_name.split(".")[0]), original)
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

                for i in range(int(num[0])):

                    if scores[i] < threshold:
                        continue


                    # Create the detection bounding box row
                    pre_track_data.append({
                        "frame_id": file_name.split(".")[0],
                        "box": (boxes[i][1] * width, boxes[i][0] * height, boxes[i][3] * width, boxes[i][2] * height),
                    # Values returned are in order ymin, xmin, ymax, xmax
                        "track_id": -1,
                        "label": classes[i],
                        "score": scores[i],
                        "source": "DET"
                    })

                prog = '\rCompleted %.2f %%' % (100 * float((int(file_name.split(".")[0]) - min_index) / num_frames))
                print('{}'.format(prog), end="")
    print('\rCompleted %.2f %%' % (100.00))
    print(counter)
    pre_track_data = pd.DataFrame(pre_track_data)
    return pre_track_data



def calculate_tracks_mot(bboxes_inf):
    mot_tracker = Sort()
    tracks = []

    # Get rid of any scores below 0.5
    bboxes_pass = bboxes_inf.copy()
    bboxes_pass = bboxes_pass[bboxes_pass.score > 0.5]

    # Change the model name here for testing
    for frame_id, frame_data in bboxes_pass.groupby("frame_id"):
        det_score = []

        for box_idx, box_data in frame_data.iterrows():
            # print "In: ", box_data.frame_id, (box_data.xmin, box_data.ymin, box_data.xmax, box_data.ymax), box_data.score
            det_score.append(list(np.append(box_data.box, box_data.score)))

        # Update the tracker
        mot_track_list = mot_tracker.update(np.array(det_score))

        # What is returned may not be in the same order. Need to match bounding boxes up with track ids
        for t in mot_track_list:
            # print "Out:", box_data.frame_id, (t[0], t[1], t[2], t[3]), int(t[4])

            tracks.append({
                "frame_id": frame_id,
                "box": t,
                "track_id": int(t[4]),
                "label": "",
                "score": -1.0,
                "source": "SORT"
            })

    return pd.DataFrame(tracks)


def get_RGB(n):
    max_value = 16581375 #255**3
    interval = int(max_value / n)
    colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]
    return [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]


def create_video_from_results(video_path, image_folder, input_frames, min_index, max_index, bounding_boxes):
    frame_count = 1

    data_folder = image_folder

    image = cv2.imread(join(data_folder, '{}.jpg'.format(min_index)))
    size = image.shape[1], image.shape[0]

    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    # fourcc = cv2.VideoWriter_fourcc('H', '2', '6', '4')

    out_vid = cv2.VideoWriter(video_path, fourcc, 10.0, size)

    track_data = {}

    # get unique color for each track_id
    uniques = bounding_boxes.track_id.unique()
    rand_colors = get_RGB(len(uniques))
    colors = {}
    for i, t in enumerate(uniques):
        colors.update({t: rand_colors[i]})

    list = [i for i in range(min_index, max_index)]
    # unique_frames = input_frames.frame_no.unique()
    unique_frames = np.array(list)

    frames_len = len(unique_frames)

    for fdx_1, frame_1 in enumerate(unique_frames):
        print("\rsaving frame no:{}, out of {}".format(frame_count, frames_len), end='')

        image = cv2.imread(join(data_folder, '{}.jpg'.format(frame_1)))

        items = bounding_boxes[bounding_boxes.apply(lambda x: int(x['frame_id']) == frame_1, axis=1)]
        for fdx_2, frame_2 in items.iterrows():
            xmin, ymin, xmax, ymax = frame_2.box[0], frame_2.box[1], frame_2.box[2], frame_2.box[3]
            track_id = frame_2.track_id

            cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), colors[track_id], 2)

        out_vid.write(image)
        frame_count += 1

    out_vid.release()

    print('\nCompleted!')

# detect_frames(PATH_TO_INFERENCE_GRAPH, PATH_TO_LABELS, PATH_TO_DATA, VIDEO_OUT_PATH,0, 1467, 5, 0.5, ANNOTATION_PATH)

raw_detections = inference_frames(PATH_TO_INFERENCE_GRAPH, '', PATH_TO_DATA, 755, 955, 0.5)
tracks = calculate_tracks_mot(raw_detections)
print(tracks.head())

create_video_from_results(VIDEO_OUT_PATH, PATH_TO_DATA, '', 755, 955 ,tracks)
# disp_video(VIDEO_OUT_PATH)
