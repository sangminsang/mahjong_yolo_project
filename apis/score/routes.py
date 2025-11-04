import os
from flask import Blueprint, render_template, request, session
from werkzeug.utils import secure_filename
from .logic import detect_mahjong_tiles_in_image, calculate_final_score, model
import cv2
import json
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener
from flask import current_app

# HEIF 포맷 지원 등록
register_heif_opener()

score_bp = Blueprint('score', __name__, url_prefix='/score')

# 가능한 모든 마작패 리스트 (뒷면 포함)
ALL_TILES = (
    [f"{n}m" for n in range(1, 10)] +  # 만수패
    [f"{n}p" for n in range(1, 10)] +  # 핀즈패
    [f"{n}s" for n in range(1, 10)] +  # 소우즈패
    [f"{n}z" for n in range(1, 8)] +   # 자패
    ["r5m", "r5p", "r5s"] +            # 적도라
    ["Back"]                           # 뒷면 추가
)

@score_bp.route('/', methods=['GET', 'POST'])
def score_calculation():
    if request.method == 'POST':
        # 디버깅 코드 추가
        print("\n=== 초기 폼 데이터 디버깅 ===")
        print("폼 데이터:", request.form)
        print("쯔모 체크 여부:", request.form.get('tsumo'))
        print("리치 체크 여부:", request.form.get('riichi'))
        print("더블리치 체크 여부:", request.form.get('double_riichi'))
        print("일발 체크 여부:", request.form.get('one_shot'))
        print("해저/하저 체크 여부:", request.form.get('last_tile_win'))
        print("천화/지화 체크 여부:", request.form.get('tian_hu_di_hu'))    
    if request.method == 'GET':
        return render_template('score/upload.html')
    
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('score/result.html', result={'error': '이미지를 업로드해주세요'})
        
        file = request.files['file']
        if file.filename == '':
            return render_template('score/result.html', result={'error': '파일이 선택되지 않았습니다'})
        
        try:
            # 파일을 임시로 저장
            filename = secure_filename(file.filename)
            filepath = os.path.join('static', 'uploads', filename)
            file.save(filepath)
            
            print("이미지 파일 저장 완료:", filepath)
            
            # HEIC 파일 처리
            if filename.lower().endswith('.heic'):
                try:
                    image = Image.open(filepath)
                    png_path = os.path.join('static', 'uploads', 'temp.png')
                    image.save(png_path, format='PNG')
                    filepath = png_path
                    print("HEIC 파일을 PNG로 변환 완료")
                except Exception as e:
                    raise Exception(f"HEIC 파일 변환 중 오류 발생: {str(e)}")
            
            # 이미지 읽기
            image = cv2.imread(filepath)
            if image is None:
                raise Exception("이미지를 읽을 수 없습니다")
            
            detected_tiles = detect_mahjong_tiles_in_image(model, image)
            print("감지된 타일:", detected_tiles)
            
            # JSON 파일로 저장
            json_path = 'detected_tiles.json'
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(detected_tiles, f, ensure_ascii=False, indent=2)
            
            print("JSON 파일 저장 완료:", json_path)
            
            # 2단계: 점수 계산을 위한 추가 정보 수집
            additional_info = {
                'main_round': request.form.get('main_round', 'east'),
                'round_count': int(request.form.get('round_count', 0)),
                'seat': request.form.get('seat', 'east'),
                'tsumo': request.form.get('tsumo') == 'on',
                'riichi': request.form.get('riichi') == 'on',
                'one_shot': request.form.get('one_shot') == 'on',
                'flower_on_mount': request.form.get('flower_on_mount') == 'on',
                'double_riichi': request.form.get('double_riichi') == 'on',
                'huro': request.form.get('huro') == 'on',
                'last_tile_win': request.form.get('last_tile_win') == 'on',
                'steal_kang': request.form.get('steal_kang') == 'on',
                'tian_hu_di_hu': request.form.get('tian_hu_di_hu') == 'on',
                'dora_indicators': request.form.getlist('dora_indicators'),
                'ankan_tiles': request.form.getlist('ankan_tiles'),
                'winning_tile': request.form.get('winning_tile', ''),
                'ming_tiles': request.form.getlist('ming_tiles')
            }
            
            # 세션에 정보 저장 (패 편집 후 사용)
            session['detected_tiles'] = detected_tiles[0] if detected_tiles else []
            session['additional_info'] = additional_info
            
            # 패 편집 페이지로 이동
            return render_template('score/score_edit_tiles.html', 
                                 tiles=detected_tiles[0] if detected_tiles else [],
                                 available_tiles=ALL_TILES,
                                 additional_info=additional_info)
            
        except Exception as e:
            print("오류 발생:", str(e))
            return render_template('score/result.html', result={'error': str(e)})
            
        #finally:
            # 임시 파일 정리
            #if 'filepath' in locals() and os.path.exists(filepath):
                #os.remove(filepath)

@score_bp.route('/score_edit_tiles', methods=['POST'])
def score_edit_tiles():
    # 사용자가 편집한 패 목록 받기
    edited_tiles = request.form.getlist('edited_tiles[]')
    
    # 세션에서 추가 정보 가져오기
    additional_info = session.get('additional_info', {})
    
    # 패 편집 페이지로 이동
    return render_template('score/score_edit_tiles.html', 
                         tiles=edited_tiles,
                         available_tiles=ALL_TILES,
                         additional_info=additional_info)

@score_bp.route('/score_analyze_edited', methods=['POST'])
def score_analyze_edited():
    
    # 사용자가 편집한 패 목록 받기
    edited_tiles = request.form.getlist('edited_tiles[]')
    
    # 세션에서 추가 정보 가져오기
    additional_info = session.get('additional_info', {})

    # 단일 값 필드와 리스트 필드 구분
    single_value_fields = {
        'main_round': 'east',
        'seat': 'east',
        'winning_tile': ''
    }

    list_fields = {
        'dora_indicators': [],
        'ankan_tiles': [],
        'ming_tiles': []
    }

    # 단일 값 필드는 직접 가져오기 (첫 번째 값 사용)
    for key, default in single_value_fields.items():
        existing_value = session.get('additional_info', {}).get(key, default)
        additional_info[key] = request.form.get(key, existing_value)

    # 리스트 필드는 getlist() 사용
    for key, default in list_fields.items():
        values = request.form.getlist(key + '[]') or request.form.getlist(key)
        additional_info[key] = values if values else default

    # 정수 필드 검증
    try:
        additional_info['round_count'] = int(request.form.get('round_count', 0))
    except (ValueError, TypeError):
        additional_info['round_count'] = 0
    
    # 패 데이터 저장
    json_path = 'detected_tiles.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump([edited_tiles], f, ensure_ascii=False, indent=2)
    
    # 패 개수 경고 로직
    warning = None
    if len(edited_tiles) not in (0, 14):
        warning = {
            'level': 'warning' if len(edited_tiles) > 14 else 'danger',
            'message': f"현재 {len(edited_tiles)}개의 패가 선택되었습니다. 정확한 분석을 위해 14개의 패를 선택해주세요."
        }
    
    # 점수 계산 실행
    try:
        result = calculate_final_score(additional_info)
    except Exception as e:
        return render_template('score/error.html', error=str(e))
    
    # 핸드 타입 분류 (치또이츠/국사무쌍/일반)
    if result.get('type') == 'chitoitsu':
        result['hand_type'] = 'chitoitsu'
    elif result.get('type') in ('kokushi', 'kokushi_13wait'):
        result['hand_type'] = 'kokushi'
    else:
        result['hand_type'] = 'standard'
    
    # 적도라 개수 필드 보정
    result.setdefault('akadora_count', 0)
    
    # 경고 메시지 병합
    if warning:
        result['warnings'] = result.get('warnings', []) + [warning]
    
    return render_template('score/result.html', 
                         result=result,
                         debug_mode=current_app.config.get('DEBUG', False))


@score_bp.route('/score_recalculate', methods=['POST'])
def score_recalculate():
    # 결과 페이지에서 패 수정하기 버튼을 눌렀을 때 처리
    from flask import session
    
    # 세션에서 정보 가져오기
    detected_tiles = session.get('detected_tiles', [])
    additional_info = session.get('additional_info', {})
    
    # 패 편집 페이지로 이동
    return render_template('score/score_edit_tiles.html', 
                         tiles=detected_tiles,
                         available_tiles=ALL_TILES,
                         additional_info=additional_info)
