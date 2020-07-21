import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2';
import tensorflow.compat.v1 as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import queue
import threading
import time
import pickle
import sys

import coloredlogs
import cv2
import numpy as np
import dlib
import imutils
from imutils import face_utils
import deepgaze
from deepgaze.head_pose_estimation import CnnHeadPoseEstimator
import json
from img_json import im2json, json2im

from GazeML.src.datasources import Video, Webcam
from GazeML.src.models import ELG
import GazeML.src.util.gaze


imgs_list = []
if os.path.isdir('img_cap') == False:
	os.mkdir('img_cap')
else:
	path = os.getcwd() + os.sep + 'img_cap'
	for f in os.listdir(path):
		os.remove(path + os.sep + f)

if os.path.isdir('outputs') == False:
	os.mkdir('outputs')

def gaze():
	
	coloredlogs.install(
		datefmt='%d/%m %H:%M',
		fmt='%(asctime)s %(levelname)s %(message)s',
		level='INFO',
	)

	# Check if GPU is available
	from tensorflow.python.client import device_lib
	gpu_available = False
	try:
		gpus = [d for d in device_lib.list_local_devices()
				if d.device_type == 'GPU']
		gpu_available = len(gpus) > 0
	except:
		pass

	with tf.Session() as session:

		# Declare some parameters
		batch_size = 2

		# Define webcam stream data source
		# Change data_format='NHWC' if not using CUDA
		data_source = Webcam(tensorflow_session=session, batch_size=batch_size,
							 camera_id=0, fps=60,
							 data_format='NCHW' if gpu_available else 'NHWC',
							 eye_image_shape=(36, 60))

		model = ELG(
			session, train_data={'videostream': data_source},
			first_layer_stride=1,
			num_modules=2,
			num_feature_maps=32,
			learning_schedule=[
				{
					'loss_terms_to_optimize': {'dummy': ['hourglass', 'radius']},
				},
			],
		)

		# Begin visualization thread
		inferred_stuff_queue = queue.Queue()

		def _visualize_output():
			last_frame_index = 0
			last_frame_time = time.time()
			fps_history = []
			all_gaze_histories = [list() for _ in range(2)]
			gaze_history_max_len = 10
			prev = time.time() + 1
			i = 0

			print("\nGaze angle estimation started")

			def make_dict(img):
				# Store gaze data in json file
				nonlocal prev, i

				if time.time() - prev > 1:
					ghl = all_gaze_histories[0]
					if len(ghl) > gaze_history_max_len:
						ghl = ghl[-gaze_history_max_len:]
					left = np.asarray(ghl)

					ghr = all_gaze_histories[1]
					if len(ghr) > gaze_history_max_len:
						ghr = ghr[-gaze_history_max_len:]
					right = np.asarray(ghr)
					sys.stdout.write(f"\rCaptured frame {i}")
					sys.stdout.flush()
					i += 1
					prev = time.time()
					cv2.imwrite(os.getcwd() + os.sep + 'img_cap' + os.sep + f'{i}.jpg', img)

					jstr = im2json(img)
					img_dict = json.loads(jstr)
					img_dict['index'] = i
					img_dict['left_eye'] = {
						'pitch' : np.mean(left, axis=0)[0],
						'yaw' : np.mean(left, axis=0)[1]
					} if left.any() else None
					img_dict['right_eye'] = {
						'pitch' : np.mean(right, axis=0)[0],
						'yaw' : np.mean(right, axis=0)[1]
					} if right.any() else None

					imgs_list.append(img_dict)

			while True:
				# If no output to visualize, show unannotated frame
				if inferred_stuff_queue.empty():
					next_frame_index = last_frame_index + 1
					if next_frame_index in data_source._frames:
						next_frame = data_source._frames[next_frame_index]
						if 'faces' in next_frame and len(next_frame['faces']) == 0:
							cv2.imshow('vis', next_frame['bgr'])
							last_frame_index = next_frame_index
						make_dict(next_frame['bgr'])
					if cv2.waitKey(1) & 0xFF == ord('q'):
						prev = time.time() - 1
						make_dict(next_frame['bgr']*0)
						print()
						cv2.destroyAllWindows()
						return
					continue

				# Get output from neural network and visualize
				output = inferred_stuff_queue.get()
				bgr = None
				for j in range(batch_size):
					frame_index = output['frame_index'][j]
					if frame_index not in data_source._frames:
						continue
					frame = data_source._frames[frame_index]
					if j == 0 and output['eye_index'][j] == 0:
						img = frame['bgr'].copy()

					# Decide which landmarks are usable
					heatmaps_amax = np.amax(output['heatmaps'][j, :].reshape(-1, 18), axis=0)
					can_use_eye = np.all(heatmaps_amax > 0.7)
					can_use_eyelid = np.all(heatmaps_amax[0:8] > 0.75)
					can_use_iris = np.all(heatmaps_amax[8:16] > 0.8)

					start_time = time.time()
					eye_index = output['eye_index'][j]
					bgr = frame['bgr']
					eye = frame['eyes'][eye_index]
					eye_image = eye['image']
					eye_side = eye['side']
					eye_landmarks = output['landmarks'][j, :]
					eye_radius = output['radius'][j][0]
					if eye_side == 'left':
						eye_landmarks[:, 0] = eye_image.shape[1] - eye_landmarks[:, 0]
						eye_image = np.fliplr(eye_image)

					# Embed eye image and annotate for picture-in-picture
					eye_upscale = 2
					eye_image_raw = cv2.cvtColor(cv2.equalizeHist(eye_image), cv2.COLOR_GRAY2BGR)
					eye_image_raw = cv2.resize(eye_image_raw, (0, 0), fx=eye_upscale, fy=eye_upscale)
					eye_image_annotated = np.copy(eye_image_raw)
					if can_use_eyelid:
						cv2.polylines(
							eye_image_annotated,
							[np.round(eye_upscale*eye_landmarks[0:8]).astype(np.int32)
																	 .reshape(-1, 1, 2)],
							isClosed=True, color=(255, 255, 0), thickness=1, lineType=cv2.LINE_AA,
						)
					if can_use_iris:
						cv2.polylines(
							eye_image_annotated,
							[np.round(eye_upscale*eye_landmarks[8:16]).astype(np.int32)
																	  .reshape(-1, 1, 2)],
							isClosed=True, color=(0, 255, 255), thickness=1, lineType=cv2.LINE_AA,
						)
						cv2.drawMarker(
							eye_image_annotated,
							tuple(np.round(eye_upscale*eye_landmarks[16, :]).astype(np.int32)),
							color=(0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=4,
							thickness=1, line_type=cv2.LINE_AA,
						)
					face_index = int(eye_index / 2)
					eh, ew, _ = eye_image_raw.shape
					v0 = face_index * 2 * eh
					v1 = v0 + eh
					v2 = v1 + eh
					u0 = 0 if eye_side == 'left' else ew
					u1 = u0 + ew
					bgr[v0:v1, u0:u1] = eye_image_raw
					bgr[v1:v2, u0:u1] = eye_image_annotated

					# Visualize preprocessing results
					frame_landmarks = (frame['smoothed_landmarks']
									   if 'smoothed_landmarks' in frame
									   else frame['landmarks'])
					for f, face in enumerate(frame['faces']):
						for landmark in frame_landmarks[f][:-1]:
							cv2.drawMarker(bgr, tuple(np.round(landmark).astype(np.int32)),
										  color=(0, 0, 255), markerType=cv2.MARKER_STAR,
										  markerSize=2, thickness=1, line_type=cv2.LINE_AA)
						cv2.rectangle(
							bgr, tuple(np.round(face[:2]).astype(np.int32)),
							tuple(np.round(np.add(face[:2], face[2:])).astype(np.int32)),
							color=(0, 255, 255), thickness=1, lineType=cv2.LINE_AA,
						)

					# Transform predictions
					eye_landmarks = np.concatenate([eye_landmarks,
													[[eye_landmarks[-1, 0] + eye_radius,
													  eye_landmarks[-1, 1]]]])
					eye_landmarks = np.asmatrix(np.pad(eye_landmarks, ((0, 0), (0, 1)),
													   'constant', constant_values=1.0))
					eye_landmarks = (eye_landmarks *
									 eye['inv_landmarks_transform_mat'].T)[:, :2]
					eye_landmarks = np.asarray(eye_landmarks)
					eyelid_landmarks = eye_landmarks[0:8, :]
					iris_landmarks = eye_landmarks[8:16, :]
					iris_centre = eye_landmarks[16, :]
					eyeball_centre = eye_landmarks[17, :]
					eyeball_radius = np.linalg.norm(eye_landmarks[18, :] -
													eye_landmarks[17, :])

					# Smooth and visualize gaze direction
					num_total_eyes_in_frame = len(frame['eyes'])
					if len(all_gaze_histories) < num_total_eyes_in_frame:
						all_gaze_histories = [list() for _ in range(num_total_eyes_in_frame)]
					gaze_history = all_gaze_histories[eye_index]
					if can_use_eye:
						# Visualize landmarks
						cv2.drawMarker(  # Eyeball centre
							bgr, tuple(np.round(eyeball_centre).astype(np.int32)),
							color=(0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=4,
							thickness=1, line_type=cv2.LINE_AA,
						)
						i_x0, i_y0 = iris_centre
						e_x0, e_y0 = eyeball_centre
						theta = -np.arcsin(np.clip((i_y0 - e_y0) / eyeball_radius, -1.0, 1.0))
						phi = np.arcsin(np.clip((i_x0 - e_x0) / (eyeball_radius * -np.cos(theta)),
												-1.0, 1.0))
						current_gaze = np.array([theta, phi])
						gaze_history.append(current_gaze)
						if len(gaze_history) > gaze_history_max_len:
							gaze_history = gaze_history[-gaze_history_max_len:]
						GazeML.src.util.gaze.draw_gaze(bgr, iris_centre, np.mean(gaze_history, axis=0),
											length=120.0, thickness=1)

					# else:
					# 	gaze_history.clear()

					if can_use_eyelid:
						cv2.polylines(
							bgr, [np.round(eyelid_landmarks).astype(np.int32).reshape(-1, 1, 2)],
							isClosed=True, color=(255, 255, 0), thickness=1, lineType=cv2.LINE_AA,
						)

					if can_use_iris:
						cv2.polylines(
							bgr, [np.round(iris_landmarks).astype(np.int32).reshape(-1, 1, 2)],
							isClosed=True, color=(0, 255, 255), thickness=1, lineType=cv2.LINE_AA,
						)
						cv2.drawMarker(
							bgr, tuple(np.round(iris_centre).astype(np.int32)),
							color=(0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=4,
							thickness=1, line_type=cv2.LINE_AA,
						)

					dtime = 1e3*(time.time() - start_time)
					if 'visualization' not in frame['time']:
						frame['time']['visualization'] = dtime
					else:
						frame['time']['visualization'] += dtime

					def _dtime(before_id, after_id):
						return int(1e3 * (frame['time'][after_id] - frame['time'][before_id]))

					def _dstr(title, before_id, after_id):
						return '%s: %dms' % (title, _dtime(before_id, after_id))

					if eye_index == len(frame['eyes']) - 1:
						# Calculate timings
						frame['time']['after_visualization'] = time.time()
						fps = int(np.round(1.0 / (time.time() - last_frame_time)))
						fps_history.append(fps)
						if len(fps_history) > 60:
							fps_history = fps_history[-60:]
						fps_str = '%d FPS' % np.mean(fps_history)
						last_frame_time = time.time()
						fh, fw, _ = bgr.shape
						cv2.putText(bgr, fps_str, org=(fw - 110, fh - 20),
								   fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.8,
								   color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
						cv2.putText(bgr, fps_str, org=(fw - 111, fh - 21),
								   fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.79,
								   color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
						cv2.imshow('vis', bgr)
						last_frame_index = frame_index

						# Quit?
						if cv2.waitKey(1) & 0xFF == ord('q'):
							prev = time.time() - 1
							make_dict(img)
							cv2.destroyAllWindows()
							return

				make_dict(img)

		visualize_thread = threading.Thread(target=_visualize_output, name='visualization')
		visualize_thread.daemon = True
		visualize_thread.start()

		# Do inference forever
		infer = model.inference_generator()
		while True:
			output = next(infer)
			for frame_index in np.unique(output['frame_index']):
				if frame_index not in data_source._frames:
					continue
				frame = data_source._frames[frame_index]
				if 'inference' in frame['time']:
					frame['time']['inference'] += output['inference_time']
				else:
					frame['time']['inference'] = output['inference_time']
			inferred_stuff_queue.put_nowait(output)

			if not visualize_thread.isAlive():
				break

			if not data_source._open:
				break

	path = os.getcwd() + os.sep + 'outputs' + os.sep + 'img_stats.json'
	with open(path, "w") as p: 
		json.dump(imgs_list, p, indent = 4)

	print('Gaze angle estimation complete\n')


def pose():
	models = os.getcwd() + os.sep + 'models'
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(models + os.sep + 'shape_predictor_68_face_landmarks.dat')

	with tf.Session() as sess:

		head_pose_estimator = CnnHeadPoseEstimator(sess)
		head_pose_estimator.load_pitch_variables(models + os.sep + 'pitch.tf')
		head_pose_estimator.load_yaw_variables(models + os.sep + 'yaw.tf')
		head_pose_estimator.load_roll_variables(models + os.sep + 'roll.tf')

		path = os.getcwd() + os.sep + 'outputs' + os.sep + 'img_stats.json'
		with open(path, "r") as p: 
			data_list = json.load(p)

		print("Head pose estimation started")

		for data in data_list:
			frame = json2im(json.dumps(data))
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			(fh, fw) = frame.shape[:2]

			faces = detector(gray, 0)

			for face in faces:
				(x, y, w, h) = face_utils.rect_to_bb(face)
				cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
				image = frame[y:y + h, x:x + w]

				try:
					image = cv2.resize(image, (480,480))
				except:
					print('Exception')
					continue

				pitch = head_pose_estimator.return_pitch(image,radians=True)[0][0][0]
				yaw = head_pose_estimator.return_yaw(image,radians=True)[0][0][0]
				roll = head_pose_estimator.return_roll(image,radians=True)[0][0][0]
				
				sys.stdout.write(f"\rProcessed frame {data['index']}")
				sys.stdout.flush()
				
				FONT = cv2.FONT_HERSHEY_DUPLEX

				data['pose'] = {
					'pitch' : float(pitch),
					'yaw' : float(yaw),
					'roll' : float(roll)
				}

			if not faces:
				data['pose'] = None
	
	print("\nHead pose estimation complete\n")
	
	path = os.getcwd() + os.sep + 'outputs' + os.sep + 'img_stats.json'
	with open(path, "w") as p: 
		json.dump(data_list, p, indent = 4)


gaze()
pose()

print('Saved frame statistics to outputs/img_stats.json')
print('Saved images to img_cap/')
print('Done\n')