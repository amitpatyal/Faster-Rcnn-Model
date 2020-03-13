import os
import cv2
import numpy as np
import tensorflow as tf
import object_detection.utils.label_map_util as label_map_util
import object_detection.utils.visualization_utils as vis_util


PATH_TO_LABELS = os.path.join('dataset', 'label.pbtxt')
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile('inference_graph/frozen_inference_graph.pb', 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=5, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


if __name__== "__main__":
    Url = 'C:/Project/AI/data_Set/videos/project_04.MP4'

    VideoCapture = cv2.VideoCapture(Url)
    with detection_graph.as_default():
        with tf.compat.v1.Session(graph=detection_graph) as sess:
            ret = True
            while(ret):
                ret, image_frame = VideoCapture.read()
                image_frame_expand = np.expand_dims(image_frame, axis=0)
                detection_graph.get_all_collection_keys()
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')

                (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],
                                                                    feed_dict = {image_tensor : image_frame_expand})
                vis_util.visualize_boxes_and_labels_on_image_array(image_frame,
                                                                   np.squeeze(boxes),
                                                                   np.squeeze(classes).astype(np.int32),
                                                                   np.squeeze(scores),
                                                                   category_index,
                                                                   use_normalized_coordinates=True,
                                                                   line_thickness=1)
                cv2.imshow('image',cv2.resize(image_frame,(680,560)))
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    VideoCapture.release()
                    break