from flask import Blueprint, render_template, request, session, jsonify
from werkzeug.utils import secure_filename
import os
from .logic import (
    recommend_discard, 
    detect_mahjong_tiles_in_image, 
    find_best_discard, 
    explain_recommendation,
    calculate_shanten,
    calculate_effective_tiles,
    find_waiting_tiles,
    process_tile  # ✅ process_tile 임포트 확인
)
from ultralytics import YOLO
from PIL import Image
from pillow_heif import register_heif_opener
import random
import string

register_heif_opener()

model = YOLO("model/best.pt")
discard_bp = Blueprint('discard', __name__, url_prefix='/discard')

UPLOAD_FOLDER = os.path.join('static', 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

ALL_TILES = (
    [f"{n}m" for n in range(1, 10)] +
    [f"{n}p" for n in range(1, 10)] +
    [f"{n}s" for n in range(1, 10)] +
    [f"{n}z" for n in range(1, 8)] +
    ["r5m", "r5p", "r5s"]
)

def generate_request_id():
    return ''.join(random.choices(string.ascii_letters + string.digits, k=8))

def analyze_hand_and_create_result(hand, best_tile, analysis):
    explain_recommendation(hand, best_tile, analysis)
    current_shanten = calculate_shanten(hand)
    recommendation_reason = ""
    if best_tile in analysis:
        tile_analysis = analysis[best_tile]
        if 'special_form' in tile_analysis:
            if tile_analysis['special_form'] == 'chitoitsu':
                recommendation_reason = "치또이츠(7쌍)에 가까워 쌍패가 아닌 패를 우선적으로 버립니다."
            elif tile_analysis['special_form'] == 'kokushi':
                recommendation_reason = "국사무쌍에 가까워 중장패를 우선적으로 버립니다."
        elif best_tile[-1] == 'z':
            if best_tile[0] in ['1', '2', '3', '4']:
                recommendation_reason = "동, 남, 서, 북 패는 다른 패와 연결할 수 없어서 먼저 버리는 것이 좋습니다."
            else:
                recommendation_reason = "백, 발, 중 패는 다른 패와 연결되지 않아 버리는 것이 좋습니다."
        elif best_tile[0] in ['1', '9']:
            recommendation_reason = "1과 9는 한쪽으로만 연결할 수 있어서 버리기 좋은 패입니다."
        elif any(len(block) == 3 for block in tile_analysis['blocks']):
            recommendation_reason = "이미 완성된 세 패 조합이 있어 남은 패들을 발전시킬 수 있습니다."
        elif any(len(block) == 2 for block in tile_analysis['blocks']):
            recommendation_reason = "연결 가능한 두 패 조합을 유지하면서 더 좋은 패를 기다릴 수 있습니다."
        else:
            recommendation_reason = "단독으로 있는 패를 버려서 더 좋은 패 조합을 만들 수 있습니다."
    result = {
        'original_hand': hand,
        'recommended_discard': best_tile,
        'block_analysis': analysis,
        'available_tiles': ALL_TILES,
        'recommendation_reason': recommendation_reason,
        'current_shanten': current_shanten,
        'discard_analysis': {}
    }
    for tile in hand:
        temp_hand = hand.copy()
        temp_hand.remove(tile)
        shanten_after = calculate_shanten(temp_hand)
        effective_tiles = calculate_effective_tiles(temp_hand, shanten_after)
        result['discard_analysis'][str(tile)] = {
            'shanten_before': current_shanten,
            'shanten_after': shanten_after,
            'effective_tiles': effective_tiles,
            'effective_count': len(effective_tiles)
        }
    if current_shanten == 0:
        waiting_tiles = find_waiting_tiles(hand)
        result['waiting_tiles'] = waiting_tiles
    return result

@discard_bp.route('/', methods=['GET', 'POST'])
def discard_recommendation():
    if request.method == 'GET':
        return render_template('discard/upload.html')
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('discard/result.html', result={'error': '이미지를 업로드해주세요'})
        file = request.files['file']
        if file.filename == '':
            return render_template('discard/result.html', result={'error': '파일이 선택되지 않았습니다'})
        try:
            filepath = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
            file.save(filepath)
            if filepath.lower().endswith('.heic'):
                image = Image.open(filepath)
                png_path = os.path.join(UPLOAD_FOLDER, 'temp.png')
                image.save(png_path, format='PNG')
                filepath = png_path
            detect_mahjong_tiles_in_image(model, filepath)
            json_path = 'detected_tiles.json'
            detected_result = recommend_discard(json_path)
            detected_tiles = []
            if 'original_hand' in detected_result:
                detected_tiles = detected_result['original_hand']
            session['detected_tiles'] = detected_tiles
            request_id = generate_request_id()
            session['last_request_id'] = request_id
            return render_template('discard/discard_edit_tiles.html', 
                                 tiles=detected_tiles,
                                 available_tiles=ALL_TILES,
                                 request_id=request_id)
        except Exception as e:
            return render_template('discard/result.html', result={'error': f'예상치 못한 오류 발생: {str(e)}'})
        finally:
            if 'filepath' in locals() and os.path.exists(filepath):
                os.remove(filepath)

@discard_bp.route('/next', methods=['POST'])
def next_tile():
    try:
        request_id = request.form.get('request_id')
        if 'last_request_id' in session and session['last_request_id'] == request_id:
            return jsonify({'error': 'Duplicate request'})
        session['last_request_id'] = request_id
        
        # 현재 패 정규화
        current_hand = [process_tile(t) for t in session.get('current_hand', [])]  # ✅ 핵심 수정
        
        # 버릴 패 정규화
        discard_tile = process_tile(request.form.get('discard_tile'))  # ✅ 핵심 수정
        if discard_tile in current_hand:
            current_hand.remove(discard_tile)
        
        # 새로 뽑은 패 정규화
        new_tile = process_tile(request.form.get('selected_tile'))  # ✅ 핵심 수정
        if new_tile:
            current_hand.append(new_tile)
        
        best_tile, analysis = find_best_discard(current_hand)
        session['current_hand'] = current_hand  # 정규화된 패 저장
        result = analyze_hand_and_create_result(current_hand, best_tile, analysis)
        new_request_id = generate_request_id()
        session['last_request_id'] = new_request_id
        return render_template('discard/result.html', result=result, request_id=new_request_id)
    except KeyError as ke:
        return render_template('discard/result.html', result={'error': f'분석 오류: {str(ke)}'})
    except Exception as e:
        return render_template('discard/result.html', result={'error': f'다음 패 분석 중 오류 발생: {str(e)}'})

@discard_bp.route('/edit_tiles', methods=['POST'])
def discard_edit_tiles():
    edited_tiles = request.form.getlist('edited_tiles[]')
    request_id = generate_request_id()
    session['last_request_id'] = request_id
    return render_template('discard/discard_edit_tiles.html', 
                         tiles=edited_tiles,
                         available_tiles=ALL_TILES,
                         request_id=request_id)

@discard_bp.route('/analyze_edited', methods=['POST'])
def analyze_edited():
    edited_tiles = request.form.getlist('edited_tiles[]')
    validated_tiles = [process_tile(t) for t in edited_tiles]  # ✅ 정규화
    warning = None
    if len(validated_tiles) != 14:
        warning = f"현재 {len(validated_tiles)}개의 패가 선택되었습니다. 정확한 분석을 위해 14개의 패를 선택해주세요."
    
    if validated_tiles:
        best_tile, analysis = find_best_discard(validated_tiles)
        session['current_hand'] = validated_tiles  # ✅ 정규화된 패 저장
        result = analyze_hand_and_create_result(validated_tiles, best_tile, analysis)
        current_shanten = calculate_shanten(validated_tiles)
        result['current_shanten'] = current_shanten
        if current_shanten == 0:
            waiting_tiles = find_waiting_tiles(validated_tiles)
            result['waiting_tiles'] = waiting_tiles
        if warning:
            result['warning'] = warning
        request_id = generate_request_id()
        session['last_request_id'] = request_id
        return render_template('discard/result.html', result=result, request_id=request_id)
    else:
        return render_template('discard/result.html', result={'error': '패가 선택되지 않았습니다'})

@discard_bp.errorhandler(500)
def internal_error(error):
    return render_template('discard/error.html', 
                          error_message="예상치 못한 오류가 발생했습니다. 패 개수를 확인해주세요."), 500
