from flask import Blueprint, render_template, Response, jsonify, request
import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
import base64
import io
from PIL import Image

realtime_bp = Blueprint('realtime', __name__, url_prefix='/realtime')

# 모델 경로 설정 (프로젝트 루트 기준)
model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "model", "best.pt")

# YOLO 모델 로드
try:
    model = YOLO(model_path)
    print(f"모델 로드 성공: {model_path}")
except Exception as e:
    print(f"모델 로드 실패: {e}")
    model = None

@realtime_bp.route('/')
def index():
    return render_template('realtime/index.html')

@realtime_bp.route('/process_image', methods=['POST'])
def process_image():
    if model is None:
        return jsonify({'status': 'error', 'message': '모델 로드 실패'}), 500
        
    try:
        # 이미지 데이터 받기
        data = request.json
        image_data = data.get('image', '')
        
        # Base64 이미지 데이터 디코딩
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # 이미지 변환
        image = Image.open(io.BytesIO(image_bytes))
        image_np = np.array(image)
        
        # BGR로 변환 (OpenCV 형식)
        if len(image_np.shape) == 3 and image_np.shape[2] == 4:  # RGBA 이미지
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
        elif len(image_np.shape) == 3 and image_np.shape[2] == 3:  # RGB 이미지
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
        # 모델로 객체 감지
        results = model(image_np, conf=0.4)  # 신뢰도 임계값 설정
        
        # 결과 처리
        detections = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            cls = result.boxes.cls.cpu().numpy()
            
            for box, conf, cl in zip(boxes, confs, cls):
                x1, y1, x2, y2 = box
                detections.append({
                    'box': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': float(conf),
                    'class': int(cl),
                    'name': model.names[int(cl)]
                })
        
        return jsonify({
            'status': 'success',
            'detections': detections
        })
        
    except Exception as e:
        print(f"이미지 처리 오류: {e}")
        return jsonify({
            'status': 'error',
            'message': f'이미지 처리 중 오류 발생: {str(e)}'
        }), 500
