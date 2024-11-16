# app.py
try:
    from flask import Flask, render_template, request, Response, jsonify
except ImportError as e:
    print("Error importing Flask. Make sure you don't have a file named 'flask.py' in your directory")
    print("Delete any file named 'flask.py' and try again")
    raise e

import cv2
import numpy as np
import mediapipe as mp
import os
import sys
from werkzeug.utils import secure_filename

# Global variables
width = 640
height = 480
camera = None
source_image = None
src_landmark_points = None
indexes_triangles = None
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Deepfake/Deepfake'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def get_landmark_points(src_image):
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None
        if len(results.multi_face_landmarks) > 1:
            return None

        src_face_landmark = results.multi_face_landmarks[0].landmark
        landmark_points = []
        for i in range(468):
            y = int(src_face_landmark[i].y * src_image.shape[0])
            x = int(src_face_landmark[i].x * src_image.shape[1])
            landmark_points.append((x, y))
        return landmark_points

def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index

def get_triangles(convexhull, landmarks_points, np_points):
    rect = cv2.boundingRect(convexhull)
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(landmarks_points)
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)

    indexes_triangles = []
    for t in triangles:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        index_pt1 = np.where((np_points == pt1).all(axis=1))
        index_pt1 = extract_index_nparray(index_pt1)
        index_pt2 = np.where((np_points == pt2).all(axis=1))
        index_pt2 = extract_index_nparray(index_pt2)
        index_pt3 = np.where((np_points == pt3).all(axis=1))
        index_pt3 = extract_index_nparray(index_pt3)

        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            triangle = [index_pt1, index_pt2, index_pt3]
            indexes_triangles.append(triangle)

    return indexes_triangles

def triangulation(triangle_index, landmark_points, img=None):
    tr1_pt1 = landmark_points[triangle_index[0]]
    tr1_pt2 = landmark_points[triangle_index[1]]
    tr1_pt3 = landmark_points[triangle_index[2]]
    triangle = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)

    rect = cv2.boundingRect(triangle)
    (x, y, w, h) = rect

    cropped_triangle = None
    if img is not None:
        cropped_triangle = img[y: y + h, x: x + w]

    cropped_triangle_mask = np.zeros((h, w), np.uint8)

    points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                      [tr1_pt2[0] - x, tr1_pt2[1] - y],
                      [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)

    cv2.fillConvexPoly(cropped_triangle_mask, points, 255)

    return points, cropped_triangle, cropped_triangle_mask, rect

def warp_triangle(rect, points1, points2, src_cropped_triangle, dest_cropped_triangle_mask):
    (x, y, w, h) = rect
    matrix = cv2.getAffineTransform(np.float32(points1), np.float32(points2))
    warped_triangle = cv2.warpAffine(src_cropped_triangle, matrix, (w, h))
    warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=dest_cropped_triangle_mask)
    return warped_triangle

def add_piece_of_new_face(new_face, rect, warped_triangle):
    (x, y, w, h) = rect
    new_face_rect_area = new_face[y: y + h, x: x + w]
    new_face_rect_area_gray = cv2.cvtColor(new_face_rect_area, cv2.COLOR_BGR2GRAY)
    _, mask_triangles_designed = cv2.threshold(new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
    warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)
    new_face_rect_area = cv2.add(new_face_rect_area, warped_triangle)
    new_face[y: y + h, x: x + w] = new_face_rect_area

def swap_new_face(dest_image, dest_image_gray, dest_convexHull, new_face):
    face_mask = np.zeros_like(dest_image_gray)
    head_mask = cv2.fillConvexPoly(face_mask, dest_convexHull, 255)
    face_mask = cv2.bitwise_not(head_mask)
    head_without_face = cv2.bitwise_and(dest_image, dest_image, mask=face_mask)
    result = cv2.add(head_without_face, new_face)
    (x, y, w, h) = cv2.boundingRect(dest_convexHull)
    center_face = (int((x + x + w) / 2), int((y + y + h) / 2))
    return cv2.seamlessClone(result, dest_image, head_mask, center_face, cv2.MIXED_CLONE)

def process_frame(frame):
    global width, height, source_image, indexes_triangles, src_landmark_points
    
    try:
        dest_image = cv2.resize(frame, (width, height))
        dest_image_gray = cv2.cvtColor(dest_image, cv2.COLOR_BGR2GRAY)
        
        dest_landmark_points = get_landmark_points(dest_image)
        if dest_landmark_points is None:
            return frame
            
        dest_np_points = np.array(dest_landmark_points)
        dest_convexHull = cv2.convexHull(dest_np_points)

        frame_height, frame_width, channels = dest_image.shape
        new_face = np.zeros((frame_height, frame_width, channels), np.uint8)

        for triangle_index in indexes_triangles:
            points, src_cropped_triangle, cropped_triangle_mask, _ = triangulation(
                triangle_index=triangle_index,
                landmark_points=src_landmark_points,
                img=source_image)

            points2, _, dest_cropped_triangle_mask, rect = triangulation(
                triangle_index=triangle_index,
                landmark_points=dest_landmark_points)

            warped_triangle = warp_triangle(
                rect=rect, 
                points1=points, 
                points2=points2,
                src_cropped_triangle=src_cropped_triangle,
                dest_cropped_triangle_mask=dest_cropped_triangle_mask)

            add_piece_of_new_face(new_face=new_face, rect=rect, warped_triangle=warped_triangle)

        result = swap_new_face(
            dest_image=dest_image, 
            dest_image_gray=dest_image_gray,
            dest_convexHull=dest_convexHull, 
            new_face=new_face)

        result = cv2.medianBlur(result, 3)
        return result
    except Exception as e:
        print(f"Error in process_frame: {str(e)}")
        return frame

def generate_frames():
    global camera, source_image
    while True:
        try:
            success, frame = camera.read()
            if not success or source_image is None:
                break
            else:
                processed_frame = process_frame(frame)
                ret, buffer = cv2.imencode('.jpg', processed_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            print(f"Error in generate_frames: {str(e)}")
            break

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global source_image, src_landmark_points, indexes_triangles
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            source_image = cv2.imread(filepath)
            if source_image is None:
                return jsonify({'error': 'Could not load image'}), 400
                
            src_image_gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
            src_mask = np.zeros_like(src_image_gray)
            src_landmark_points = get_landmark_points(source_image)
            if src_landmark_points is None:
                return jsonify({'error': 'No face detected in source image'}), 400
                
            src_np_points = np.array(src_landmark_points)
            src_convexHull = cv2.convexHull(src_np_points)
            cv2.fillConvexPoly(src_mask, src_convexHull, 255)
            indexes_triangles = get_triangles(
                convexhull=src_convexHull,
                landmarks_points=src_landmark_points,
                np_points=src_np_points
            )
            
            return jsonify({'success': True, 'filename': filename})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/video')
def video():
    global camera
    try:
        if camera is None:
            camera = cv2.VideoCapture(0)
            camera.set(3, width)
            camera.set(4, height)
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(f"Error in video route: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/stop')
def stop_video():
    global camera
    if camera is not None:
        camera.release()
        camera = None
    return jsonify({'success': True})

if __name__ == '__main__':
    print("Starting Flask application...")
    print("Make sure you don't have a file named 'flask.py' in your directory")
    print(f"Upload folder path: {app.config['UPLOAD_FOLDER']}")
    app.run(debug=True)