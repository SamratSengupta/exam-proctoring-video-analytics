import cv2
import json
from img_json import im2json, json2im
import time
import numpy as np
import imutils
import sys
import time
import os

path = os.getcwd() + os.sep + 'outputs'
with open(path + os.sep + 'img_stats.json', "r") as p: 
	data_list = json.load(p)

video_codec = cv2.VideoWriter_fourcc(*'mp4v')
fps=30
out = None
filename = 'video' + str(time.time()) + '.mp4'

bad_pose = 0
bad_left = True
bad_right = True
bad_gaze = 0
prev_pitch = 0
prev_yaw = 0
prev_left_eye_pitch = 0
prev_left_eye_yaw = 0
prev_right_eye_pitch = 0
prev_right_eye_yaw = 0

FONT = cv2.FONT_HERSHEY_DUPLEX

def draw():

	fh, fw = frame.shape[:2]

	if(data['pose']):
		if -0.15 < pitch < 0:
			cv2.putText(frame, f'pitch = {pitch:.2f}', (20,60), FONT, 1, (0,255,0), 1)
		else:
			cv2.putText(frame, f'pitch = {pitch:.2f}', (20,60), FONT, 1, (0,0,255), 1)
		if -0.5 < yaw < 0.5:
			cv2.putText(frame, f'yaw = {yaw:.2f}', (20,90), FONT, 1, (0,255,0), 1)
		else:
			cv2.putText(frame, f'yaw = {yaw:.2f}', (20,90), FONT, 1, (0,0,255), 1)
	else:
		cv2.putText(frame, 'FACE NOT DETECTED', (20,30), FONT, 1, (0,0,255), 1)

	if data['pose']:
		if data['left_eye']:
			if -0.5 < left_eye_pitch < 0:
				cv2.putText(frame, f'left eye pitch = {left_eye_pitch:.2f}', (20,120), FONT, 1, (0,255,0), 1)
			else:
				cv2.putText(frame, f'left eye pitch = {left_eye_pitch:.2f}', (20,120), FONT, 1, (0,0,255), 1)
			
			if -0.5 < left_eye_yaw < 0.5:
				cv2.putText(frame, f'left eye yaw = {left_eye_yaw:.2f}', (20,150), FONT, 1, (0,255,0), 1)
			else:
				cv2.putText(frame, f'left eye yaw = {left_eye_yaw:.2f}', (20,150), FONT, 1, (0,0,255), 1)
		else:
			cv2.putText(frame, 'left eye not detected', (20,120), FONT, 1, (0,0,0), 1)

		if data['right_eye']:
			if -0.5 < right_eye_pitch < 0:
				cv2.putText(frame, f'right eye pitch = {right_eye_pitch:.2f}', (20,180), FONT, 1, (0,255,0), 1)
			else:
				cv2.putText(frame, f'right eye pitch = {right_eye_pitch:.2f}', (20,180), FONT, 1, (0,0,255), 1)
			
			if -0.5 < right_eye_yaw < 0.5:
				cv2.putText(frame, f'right eye yaw = {right_eye_yaw:.2f}', (20,210), FONT, 1, (0,255,0), 1)
			else:
				cv2.putText(frame, f'right eye yaw = {right_eye_yaw:.2f}', (20,210), FONT, 1, (0,0,255), 1)

		else:
			cv2.putText(frame, 'right eye not detected', (20,180), FONT, 1, (0,0,0), 1)

	if bad_pose >= 5:
		cv2.putText(frame, 'CHEATING DETECTED: BAD POSE', (20,fh-10), FONT, 1, (0,0,255), 1)

	if bad_gaze >= 5:
		cv2.putText(frame, 'CHEATING DETECTED: BAD GAZE', (20,fh-70), FONT, 1, (0,0,255), 1)


for data in data_list:
	frame = json2im(json.dumps(data))
	frame = imutils.resize(frame, height = 720)

	if data['pose']:
		pitch = data['pose']['pitch']		
		yaw = data['pose']['yaw']
		
		if not -0.15 < pitch < 0 or not -0.5 < yaw < 0.5:
			if abs(prev_pitch - pitch) < 0.1 and abs(prev_yaw - yaw) < 0.1:
				bad_pose += 1
		else:
			bad_pose = 0

		prev_pitch = pitch
		prev_yaw = yaw
	else:
		bad_pose += 1
	
	if data['pose']:
		if data['left_eye']:
			left_eye = data['left_eye']
			left_eye_pitch = left_eye['pitch']
			left_eye_yaw = left_eye['yaw']
			if not -0.5 < left_eye_pitch < 0 or not -0.5 < left_eye_yaw < 0.5:
				if abs(left_eye_pitch - prev_left_eye_pitch) < 0.2 and abs(left_eye_yaw - prev_left_eye_yaw) < 0.2:
					bad_left = True
			else:
				bad_left = False
			
			prev_left_eye_pitch = left_eye_pitch
			prev_left_eye_yaw = left_eye_yaw

		
		if data['right_eye']:
			right_eye = data['right_eye']
			right_eye_pitch = right_eye['pitch']
			right_eye_yaw = right_eye['yaw']
			if not -0.5 < right_eye_pitch < 0 or not -0.5 < right_eye_yaw < 0.5:
				if abs(right_eye_pitch - prev_right_eye_pitch) < 0.2 and abs(right_eye_yaw - prev_right_eye_yaw) < 0.2:
					bad_right = True
			else:
				bad_right = False

			prev_right_eye_pitch = right_eye_pitch
			prev_right_eye_yaw = right_eye_yaw

		if bad_left or bad_right:
			bad_gaze += 1
		else:
			bad_gaze = 0

	sys.stdout.write(f"\rChecked frame {data['index']}")
	sys.stdout.flush()

	# Draw frame statistics on video
	draw()

	(height, width) = frame.shape[:2]
	if out is None:
		out = cv2.VideoWriter(path + os.sep + filename, video_codec, fps, (width,height))


	frame_count = 0
	time = 1                                                  
	while(frame_count < time * fps):
		out.write(frame)
		frame_count += 1

	cv2.imshow('frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cv2.destroyAllWindows()
out.release()
print(f"\nOutput saved as outputs/{filename}")
print('Done\n')