"""
smokingdetection.py - Smoking Face Detection

Usage:

  detect and show UI
    python smokingdetection.py video_filename.mp4 <"screen"/events_filename.txt>

  for example:

    python smokingdetection.py "data\videos\Smoking & Driving.mp4" "screen"

"""
### import modules
import os
import sys
import cv2
import numpy as np
import time
import uuid
import tensorflow as tf
import subprocess
import csv
import datetime
import tiny_face_model
import pickle
import pylab as pl
from scipy.special import expit

"""
initialization
"""
# constants
REQUIRED_ARGUMENT_COUNT = 3
FRAMES_SLOW = 15 * 2
FRAMES_EACH = FRAMES_SLOW
MAX_INPUT_DIM = 5000.0

# first line output
is_first_output = True

# formatting bool
bool_dictionary = {True: 'yes', False: 'no', None: 'no'}

"""
check arguments
"""
if len(sys.argv) != REQUIRED_ARGUMENT_COUNT + 1:
  print ('Usage: python smokingdetection.py (video_filename.mp4) (events_filename.txt) (video_fps)')
  exit(1)
video_filename = sys.argv[1]
events_filename = sys.argv[2]
video_fps = int(sys.argv[3])

"""
start video processing
"""

# capture video
cap = cv2.VideoCapture(video_filename)

"""
load a graph from model_file
"""
def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

"""
read tensor from image
"""
def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(
        file_reader, channels=3, name="png_reader")
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(
        tf.image.decode_gif(file_reader, name="gif_reader"))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
  else:
    image_reader = tf.image.decode_jpeg(
        file_reader, channels=3, name="jpeg_reader")
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result

"""
load label file
"""
def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

"""
run face detection from a file
"""
def detect_faces_on_image(filename):
  fname = filename.split(os.sep)[-1]
  raw_img = cv2.imread(filename)
  raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
  raw_img_f = raw_img.astype(np.float32)

  def _calc_scales():
    raw_h, raw_w = raw_img.shape[0], raw_img.shape[1]
    min_scale = min(np.floor(np.log2(np.max(clusters_w[normal_idx] / raw_w))),
                    np.floor(np.log2(np.max(clusters_h[normal_idx] / raw_h))))
    max_scale = min(1.0, -np.log2(max(raw_h, raw_w) / MAX_INPUT_DIM))
    scales_down = pl.frange(min_scale, 0, 1.)
    scales_up = pl.frange(0.5, max_scale, 0.5)
    scales_pow = np.hstack((scales_down, scales_up))
    scales = np.power(2.0, scales_pow)
    return [1] # scales

  scales = _calc_scales()
  start = time.time()

  # initialize output
  bboxes = np.empty(shape=(0, 5))

  # process input at different scales
  for s in scales:
    print("Processing {} at scale {:.4f}".format(fname, s))
    img = cv2.resize(raw_img_f, (0, 0), fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
    img = img - average_image
    img = img[np.newaxis, :]

    # we don't run every template on every scale ids of templates to ignore
    tids = list(range(4, 12)) + ([] if s <= 1.0 else list(range(18, 25)))
    ignoredTids = list(set(range(0, clusters.shape[0])) - set(tids))

    # run through the net
    score_final_tf = face_detection_sess.run(score_final, feed_dict={face_detection_x: img})

    # collect scores
    score_cls_tf, score_reg_tf = score_final_tf[:, :, :, :25], score_final_tf[:, :, :, 25:125]
    prob_cls_tf = expit(score_cls_tf)
    prob_cls_tf[0, :, :, ignoredTids] = 0.0

    def _calc_bounding_boxes():
      # threshold for detection
      _, fy, fx, fc = np.where(prob_cls_tf > face_prob_thresh)

      # interpret heatmap into bounding boxes
      cy = fy * 8 - 1
      cx = fx * 8 - 1
      ch = clusters[fc, 3] - clusters[fc, 1] + 1
      cw = clusters[fc, 2] - clusters[fc, 0] + 1

      # extract bounding box refinement
      Nt = clusters.shape[0]
      tx = score_reg_tf[0, :, :, 0:Nt]
      ty = score_reg_tf[0, :, :, Nt:2*Nt]
      tw = score_reg_tf[0, :, :, 2*Nt:3*Nt]
      th = score_reg_tf[0, :, :, 3*Nt:4*Nt]

      # refine bounding boxes
      dcx = cw * tx[fy, fx, fc]
      dcy = ch * ty[fy, fx, fc]
      rcx = cx + dcx
      rcy = cy + dcy
      rcw = cw * np.exp(tw[fy, fx, fc])
      rch = ch * np.exp(th[fy, fx, fc])

      scores = score_cls_tf[0, fy, fx, fc]
      tmp_bboxes = np.vstack((rcx - rcw / 2, rcy - rch / 2, rcx + rcw / 2, rcy + rch / 2))
      tmp_bboxes = np.vstack((tmp_bboxes / s, scores))
      tmp_bboxes = tmp_bboxes.transpose()
      return tmp_bboxes

    tmp_bboxes = _calc_bounding_boxes()
    bboxes = np.vstack((bboxes, tmp_bboxes)) # <class 'tuple'>: (5265, 5)


  print("time {:.2f} secs for {}".format(time.time() - start, fname))

  # non maximum suppression
  # refind_idx = util.nms(bboxes, nms_thresh)
  refind_idx = tf.image.non_max_suppression(tf.convert_to_tensor(bboxes[:, :4], dtype=tf.float32),
                                               tf.convert_to_tensor(bboxes[:, 4], dtype=tf.float32),
                                               max_output_size=bboxes.shape[0], iou_threshold=face_nms_thresh)
  refind_idx = face_detection_sess.run(refind_idx)
  refined_bboxes = bboxes[refind_idx]
  print ('refined_bboxes', refined_bboxes)

  return refined_bboxes
  
"""
initialize tensorflow smoking face detection
"""

model_file = 'data/models/smoking-faces-output-graph.pb'
label_file = "data/models/smoking-faces-output-graph.txt"
graph = load_graph(model_file)

"""
initialize tensorflow face detection
"""

weight_file_path = 'data/models/hr_res101'
face_detection_graph = tf.Graph().as_default()

# placeholder of input images. Currently batch size of one is supported.
face_detection_x = tf.placeholder(tf.float32, [1, None, None, 3]) # n, h, w, c

# Create the tiny face model which weights are loaded from a pretrained model.
model = tiny_face_model.Model(weight_file_path)
score_final = model.tiny_face(face_detection_x)

# Load an average image and clusters(reference boxes of templates).
with open(weight_file_path, "rb") as f:
  _, mat_params_dict = pickle.load(f)

average_image = model.get_data_by_key("average_image")
clusters = model.get_data_by_key("clusters")
clusters_h = clusters[:, 3] - clusters[:, 1] + 1
clusters_w = clusters[:, 2] - clusters[:, 0] + 1
normal_idx = np.where(clusters[:, 4] == 1)

face_detection_sess = tf.Session()
face_detection_sess.run(tf.global_variables_initializer())

face_prob_thresh=0.5
face_nms_thresh=0.1

"""
process
"""

frames_read = 0

output_events_filename = events_filename if events_filename != 'screen' else 'sol.txt'
print ('writing events to:', output_events_filename)

with open(output_events_filename, 'w', newline='') as csv_file:

  # initialize csv_event_writer
  csv_event_writer = csv.writer(csv_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)

  sess = tf.Session(graph=graph)

  # tensorflow input/output
  input_layer = 'Placeholder'
  output_layer = 'final_result'
  input_name = "import/" + input_layer
  output_name = "import/" + output_layer
  input_operation = graph.get_operation_by_name(input_name)
  output_operation = graph.get_operation_by_name(output_name)

  input_height = 299
  input_width = 299
  input_mean = 0
  input_std = 255

  # video control
  face_img = None
  start_time = time.time()
  while(cap.isOpened()):
    ok, frame = cap.read()
    
    if not ok:
      break

    frames_read+=1

    if frames_read % FRAMES_EACH == 0:

      # calculate elapsed time
      elapsed_time = int(time.time() - start_time)

      # detect faces
      frame_image_filename = 'frame_image.jpg'
      cv2.imwrite(frame_image_filename, frame)
      detected_faces = detect_faces_on_image(frame_image_filename)

      # draw detected_faces rects
      for (x, y, x2, y2, c) in detected_faces:

        try:
          # calculate face's width and height
          x = int(x)
          y = int(y)
          w = int(x2 - x)
          h = int(y2 - y)
          print (x, y, w, h, c)

          # get face image
          frame_for_face = frame.copy()
          face_img = frame_for_face[y:y+h,x:x+w]

          # save face image
          file_name = 'detected_face.jpg'
          cv2.imwrite(file_name, face_img)

          # load tensorflow image
          t = read_tensor_from_image_file(
              file_name,
              input_height=input_height,
              input_width=input_width,
              input_mean=input_mean,
              input_std=input_std)

          # classify face_img according to smoke: yes/no
          tensorflow_results = sess.run(output_operation.outputs[0], {
              input_operation.outputs[0]: t
          })
          tensorflow_results = np.squeeze(tensorflow_results)
          top_k = tensorflow_results.argsort()[-5:][::-1]
          labels = load_labels(label_file)

          # determine if is smoking
          if labels[top_k[0]] == 'yes':
            face_color = (0,0,255)
          else:
            face_color = (255,0,0)

          # face border width
          face_border_width = 5

          # draw rectangle
          cv2.rectangle(frame, (x, y), (x+w, y+h), face_color, face_border_width)

        except:
          pass

      # show
      if events_filename == 'screen':

        # show frame
        cv2.imshow('frame', frame)

        # process screen key events
        k = cv2.waitKey(33)
        if k==27:    # Esc key to stop
          exit ('Quit')
        elif k==113:
          exit ('Quit')
        elif k==115:
          FRAMES_EACH = FRAMES_SLOW
        elif k==102:
          FRAMES_EACH = FRAMES_EACH * 2
        else:
          if k > -1:
            print (k)

# calculate elapsed time
elapsed_time = time.time() - start_time
print ('elapsed_time', elapsed_time, 'frames_read', frames_read)

### closing tool objects
cap.release()
cv2.destroyAllWindows()
