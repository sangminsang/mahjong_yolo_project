import cv2
import numpy as np
import json
import os
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog
import math
from collections import Counter
from collections import defaultdict

# 모델 경로 설정
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'model', 'best.pt')
model = YOLO(MODEL_PATH)

def detect_mahjong_tiles_in_image(model, image, exclude_classes=[]):
    detected_tiles = []
    frame_tiles = []
    
    # 모델로 이미지에서 마작 패 감지
    results = model(image)
    
    # NMS (Non-Maximum Suppression) 임계값 설정
    nms_threshold = 0.45
    
    # 감지된 박스들을 저장할 리스트
    valid_detections = []
    
    for det in results[0].boxes:
        x1, y1, x2, y2 = det.xyxy[0]
        conf = det.conf[0]
        cls = int(det.cls[0])
        label = model.names[cls]
        
        if label in exclude_classes:
            continue

        # 박스 크기 계산
        box_width = x2 - x1
        box_height = y2 - y1
        box_area = box_width * box_height
        
        # 최소 크기 임계값 (작은 박스 필터링)
        min_area_threshold = 800  # 적절한 값으로 조정 필요
        
        # 신뢰도 임계값
        confidence_threshold = 0.1  # 높은 신뢰도만 허용
        
        # 크기와 신뢰도 조건을 만족하는 경우만 처리
        if box_area > min_area_threshold and conf > confidence_threshold:
            # 중복 검출 확인
            is_duplicate = False
            for existing_det in valid_detections:
                iou = calculate_iou(
                    (x1, y1, x2, y2),
                    (existing_det[0], existing_det[1], existing_det[2], existing_det[3])
                )
                if iou > nms_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                valid_detections.append((x1, y1, x2, y2, conf, cls, label))
    
    # 유효한 감지 결과만 저장
    for x1, y1, x2, y2, conf, cls, label in valid_detections:
        frame_tiles.append(label)

    detected_tiles.append(frame_tiles)
    
    return detected_tiles

def calculate_iou(box1, box2):
    # IoU (Intersection over Union) 계산
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0
    
def handle_ming_tiles(tiles, ming_tiles):
    print("\n=== 밍깡/밍커 처리 시작 ===")
    print(f"초기 타일: {tiles}")
    print(f"밍 타일 목록: {ming_tiles}")
    
    ming_sets = []
    remaining_tiles = tiles.copy()
    
    for ming_tile in ming_tiles:
        tile_count = remaining_tiles.count(ming_tile)
        print(f"타일 '{ming_tile}' 개수: {tile_count}")
        ming_set = []
        
        if tile_count == 4:
            print(f"밍깡 발견: {ming_tile}")
            for _ in range(4):
                remaining_tiles.remove(ming_tile)
                ming_set.append(ming_tile)
            ming_sets.append(('kang', ming_set))
            print(f"밍깡 발견: {ming_set}")
        elif tile_count == 3:
            print(f"밍커 발견: {ming_tile}")
            for _ in range(3):
                remaining_tiles.remove(ming_tile)
                ming_set.append(ming_tile)
            ming_sets.append(('pon', ming_set))
            print(f"밍커 발견: {ming_set}")
    
    print(f"최종 밍 세트: {ming_sets}")
    print(f"남은 타일: {remaining_tiles}")
    return ming_sets, remaining_tiles

def handle_back_tiles(tiles):
    """뒷면 타일 처리"""
    processed_tiles = []
    for tile in tiles:
        if tile != 'back':
            processed_tiles.append(tile)
    return processed_tiles

def handle_ankan(tiles, ankan_tiles):
    # 안깡 타일 목록 처리
    if isinstance(ankan_tiles, str):
        ankan_tiles = [ankan_tiles] if ankan_tiles else []
    
    print("=== 안깡 처리 시작 ===")
    print(f"초기 타일: {tiles}")
    print(f"안깡 타일 목록: {ankan_tiles}")
    
    ankan_sets = []
    used_tiles = set()
    remaining_tiles = tiles.copy()
    
    # 안깡 타일 처리
    for ankan_tile in ankan_tiles:
        if not ankan_tile or ankan_tile == 'Back':
            continue
            
        # 안깡 세트 생성 (예: 2p -> [2p, 2p, 2p, 2p])
        ankan_set = [ankan_tile] * 4
        ankan_sets.append(ankan_set)
        
        # 타일 개수 확인
        tile_count = remaining_tiles.count(ankan_tile)
        back_count = remaining_tiles.count('Back')
        
        # 안깡 타일 2개와 뒷면 2개를 제거
        # 실제 타일 2개 제거
        for _ in range(min(2, tile_count)):
            remaining_tiles.remove(ankan_tile)
            used_tiles.add(ankan_tile)
        
        # 뒷면 2개 제거
        for _ in range(min(2, back_count)):
            remaining_tiles.remove('Back')
            used_tiles.add('Back')
    
    print(f"최종 안깡 세트: {ankan_sets}")
    print(f"사용된 타일: {used_tiles}")
    print(f"남은 타일: {remaining_tiles}")
    
    return ankan_sets, remaining_tiles, used_tiles

def count_akadora(tiles):
    akadora_count = 0
    processed_tiles = []
    for tile in tiles:
        if tile.startswith('r'):
            akadora_count += 1
            processed_tiles.append(tile[1:])  # 'r' 제거 후 저장
        else:
            processed_tiles.append(tile)
    print(f"[적도라 검증] 원본: {tiles} → 처리 후: {processed_tiles}")
    return akadora_count, processed_tiles

def find_mahjong_sets(tiles, ankan_tiles, ming_sets, winning_tile=None):
    # 1. 초기 처리: 뒷면 패 제거 + 적도라 정규화
    processed_tiles = [t for t in tiles if t != 'Back']
    akadora_count, normalized_tiles = count_akadora(processed_tiles)
    
    # 2. 치또이쯔 판정 (7페어)
    if is_chitoitsu(normalized_tiles):
        print("[특수패 인식] 치또이쯔")
        return {
            'type': 'chitoitsu',
            'head': [],
            'bodies': [],
            'ankan': ankan_tiles,
            'ming': ming_sets,
            'akadora': akadora_count
        }
    
    # 3. 국사무쌍 13면대기 판정 (Double Yakuman)
    if is_kokushi_13wait(normalized_tiles, winning_tile):
        print("[특수패 인식] 국사무쌍 13면대기")
        return {
            'type': 'kokushi_13wait',
            'head': [],
            'bodies': [],
            'ankan': ankan_tiles,
            'ming': ming_sets,
            'akadora': akadora_count
        }
    
    # 4. 일반 국사무쌍 판정 (Single Yakuman)
    elif is_kokushi_musou(normalized_tiles):
        print("[특수패 인식] 국사무쌍")
        return {
            'type': 'kokushi',
            'head': [],
            'bodies': [],
            'ankan': ankan_tiles,
            'ming': ming_sets,
            'akadora': akadora_count
        }
    
    # 5. 기존 세트 수집 (핵심 수정 부분)
    existing_bodies = []
    for ming_type, ming_tiles in ming_sets:
        existing_bodies.append(ming_tiles)  # 밍깡/밍커 추가
    existing_bodies.extend(ankan_tiles)     # 안깡 추가
    
    total_existing = len(existing_bodies)
    remaining_body_count = 4 - total_existing
    
    # 6. 남은 몸통이 없는 경우 머리만 확인
    if remaining_body_count == 0:
        heads = [t for t, cnt in Counter(normalized_tiles).items() if cnt >=2]
        if heads:
            return {
                'type': 'standard',
                'head': [heads[0], heads[0]],
                'bodies': existing_bodies,
                'ankan': ankan_tiles,
                'ming': ming_sets,
                'akadora': akadora_count
            }
        else:
            return {'type': 'invalid', 'error': "유효한 패 조합을 찾을 수 없음"}
    
    # 7. 일반 패턴 검색 (기존 로직 유지)
    print("\n=== 일반 패 조합 검색 ===")
    heads = [t for t, cnt in Counter(normalized_tiles).items() if cnt >= 2]
    
    for head in heads:
        remaining = normalized_tiles.copy()
        try:
            remaining.remove(head)
            remaining.remove(head)
        except ValueError:
            continue
        
        # 몸통 검색 재귀 함수 (기존 로직 유지)
        def find_bodies(tiles, bodies):
            if len(bodies) == remaining_body_count:
                return bodies if not tiles else None
            if len(tiles) < 3:
                return None
            
            # 커쯔 검색
            tile_counts = Counter(tiles)
            for tile, count in tile_counts.items():
                if count >= 3:
                    new_tiles = [t for t in tiles if t != tile]
                    result = find_bodies(new_tiles, bodies + [[tile]*3])
                    if result:
                        return result
            
            # 슌쯔 검색
            for t in (x for x in set(tiles) if not x.endswith('z')):
                num = int(t[0])
                if num > 7: continue
                next1 = f"{num+1}{t[1]}"
                next2 = f"{num+2}{t[1]}"
                if next1 in tiles and next2 in tiles:
                    new_tiles = tiles.copy()
                    new_tiles.remove(t)
                    new_tiles.remove(next1)
                    new_tiles.remove(next2)
                    result = find_bodies(new_tiles, bodies + [[t, next1, next2]])
                    if result: return result
            return None
        
        bodies = find_bodies(remaining, [])
        if bodies:
            # 기존 세트와 새로 찾은 세트 병합 (핵심 수정 부분)
            return {
                'type': 'standard',
                'head': [head, head],
                'bodies': existing_bodies + bodies,
                'ankan': ankan_tiles,
                'ming': ming_sets,
                'akadora': akadora_count
            }
    
    print("유효한 패 조합 없음")
    return {
        'type': 'invalid',
        'error': "유효한 패 조합을 찾을 수 없음"
    }

def is_chitoitsu(tiles):
    """적도라 포함 치또이쯔 판정"""
    return len(tiles) == 14 and all(cnt == 2 for cnt in Counter(tiles).values())

def is_kokushi_musou(tiles):
    """적도라 포함 국사무쌍 판정"""
    required = {'1m','9m','1p','9p','1s','9s','1z','2z','3z','4z','5z','6z','7z'}
    return set(tiles) == required and len(tiles) == 14 and any(tiles.count(t)==2 for t in required)

def check_yaku(head, bodies, tiles, winning_tile, call, reach, seat, main_round, 
               isTianHuDiHu, dora_marker, one_shot, tsumo, steal_kang, 
               flower_on_mount, ming_tiles, lastTileWin, ankan_sets, ming_sets, double_riichi=False, akadora_count=0):
    print("\n=== 역 판정 시작 ===")
    print(f"머리: {head}")
    print(f"몸통: {bodies}")
    print(f"안깡: {ankan_sets}")
    print(f"화료패: {winning_tile}")
    print(f"울음: {call}")
    print(f"리치: {reach}")
    print(f"더블리치: {double_riichi}")

    seat_value = seat[0] if isinstance(seat, list) and seat else seat
    main_round_value = main_round[0] if isinstance(main_round, list) and main_round else main_round

    seat_tile = f"{['east', 'south', 'west', 'north'].index(seat_value) + 1}z"
    main_round_tile = f"{['east', 'south', 'west', 'north'].index(main_round_value) + 1}z"

    han = 0  # 한수 초기화
    yakuman_count = 0  # 역만 카운트 초기화
    is_pinghu_flag = False
    yaku_list = []  # 성립된 역들을 저장할 리스트
    total_tiles = tiles + [body[0] for body in bodies if len(body) == 4]
    # 멘젠 상태 확인
    is_call = call is True or call == True or (isinstance(call, str) and call.lower() == 'true')
    is_menzen = not is_call and not ming_sets
    print(f"멘젠 상태: {is_menzen}")

    # ming_tiles 처리
    ming_tiles_list = ming_tiles if isinstance(ming_tiles, list) else ming_tiles.split(', ') if ming_tiles else []

    # 천화/지화 처리
    is_tian_hu_di_hu = isTianHuDiHu is True or isTianHuDiHu == True or (isinstance(isTianHuDiHu, str) and isTianHuDiHu.lower() == 'true')
    if is_tian_hu_di_hu:
        if seat == 'east':
            yakuman_count += 1
            yaku_list.append({
                'name': '천화',
                'han': 13,
                'description': '천화가 성립하여 1배 역만 추가'
            })
            print("천화가 성립하여 1배 역만 추가")
        else:
            yakuman_count += 1
            yaku_list.append({
                'name': '지화',
                'han': 13,
                'description': '지화가 성립하여 1배 역만 추가'
            })
            print("지화가 성립하여 1배 역만 추가")
    # 역만 체크
    if is_kokushi_13wait(tiles, winning_tile):
        yakuman_count = 2  # 2배 역만
        yaku_list.append({
            'name': '국사무쌍 13면 대기',
            'han': 26,
            'description': '국사무쌍 13면 대기가 성립하여 2배 역만 추가'
        })
        print("국사무쌍 13면 대기가 성립하여 2배 역만 추가")
        han = yakuman_count * 13
        return han, is_pinghu_flag, yakuman_count, yaku_list
    elif is_kokushi_musou(tiles):
        yakuman_count = 1  # 1배 역만
        yaku_list.append({
            'name': '국사무쌍',
            'han': 13,
            'description': '국사무쌍이 성립하여 1배 역만 추가'
        })
        print("국사무쌍이 성립하여 1배 역만 추가")
        han = yakuman_count * 13
        return han, is_pinghu_flag, yakuman_count, yaku_list

    # 스안커 단기를 먼저 체크
    suuankou_tanki = is_suuankou_tanki(head, bodies, winning_tile, ming_sets, ankan_sets)
    if suuankou_tanki:
        yakuman_count += 2
        yaku_list.append({
            'name': '스안커 단기',
            'han': 26,
            'description': '스안커 단기가 성립하여 2배 역만 추가'
        })
        print("스안커 단기가 성립하여 2배 역만 추가")
    # 스안커 단기가 아닐 경우에만 일반 스안커 체크
    elif is_suuankou(bodies, ankan_sets, ming_sets, winning_tile, tsumo) and not ming_sets:
        yakuman_count += 1
        yaku_list.append({
            'name': '스안커',
            'han': 13,
            'description': '스안커가 성립하여 1배 역만 추가'
        })
        print("스안커가 성립하여 1배 역만 추가")
    if is_daisangen(bodies, ankan_sets, ming_sets):
        yakuman_count +=1
        yaku_list.append({
            'name': '대삼원',
            'han':13,
            'description':'대삼원이 성립하여 1배 역만 추가'
        })
        print("대삼원이 성립하여 1배 역만 추가")

    if is_tsuuiisou(tiles, ankan_sets, ming_sets):
        yakuman_count += 1
        yaku_list.append({
            'name': '자일색',
            'han': 13,
            'description': '자일색이 성립하여 1배 역만 추가'
        })
        print("자일색이 성립하여 1배 역만 추가")

    if is_ryuuiisou(tiles):
        yakuman_count += 1
        yaku_list.append({
            'name': '녹일색',
            'han': 13,
            'description': '녹일색이 성립하여 1배 역만 추가'
        })
        print("녹일색이 성립하여 1배 역만 추가")

    if is_chinroutou(head, bodies, ankan_sets, ming_sets, tiles):
        yakuman_count += 1
        yaku_list.append({
            'name': '청노두',
            'han': 13,
            'description': '청노두가 성립하여 1배 역만 추가'
        })
        print("청노두가 성립하여 1배 역만 추가")

    if is_shousuushii(bodies, ankan_sets, ming_sets):
        yakuman_count += 1
        yaku_list.append({
            'name': '소사희',
            'han': 13,
            'description': '소사희가 성립하여 1배 역만 추가'
        })
        print("소사희가 성립하여 1배 역만 추가")
    if is_daisuushii(bodies, ankan_sets, ming_sets):
        yakuman_count += 2
        yaku_list.append({
            'name': '대사희',
            'han': 26,
            'description': '대사희가 성립하여 2배 역만 추가'
        })
        print("대사희가 성립하여 2배 역만 추가")

    if is_sukantsu(bodies):
        yakuman_count += 1
        yaku_list.append({
            'name': '스깡즈',
            'han': 13,
            'description': '스깡즈가 성립하여 1배 역만 추가'
        })
        print("스깡즈가 성립하여 1배 역만 추가")

    # 순정구련보등을 먼저 체크
    if is_chuuren_poutou_9wait(head, tiles, winning_tile):
        yakuman_count += 2
        yaku_list.append({
            'name': '순정구련보등',
            'han': 26,
            'description': '순정구련보등이 성립하여 2배 역만 추가'
        })
        print("순정구련보등이 성립하여 2배 역만 추가")
    # 순정구련보등이 아닐 경우에만 일반 구련보등 체크
    elif is_chuuren_poutou(head, tiles):
        yakuman_count += 1
        yaku_list.append({
            'name': '구련보등',
            'han': 13,
            'description': '구련보등이 성립하여 1배 역만 추가'
        })
        print("구련보등이 성립하여 1배 역만 추가")

    if yakuman_count > 0:
        print(f"역만이 성립하여 도라 계산 생략")
        han = yakuman_count * 13
        return han, is_pinghu_flag, yakuman_count, yaku_list
    # 일반 역 판정
    if is_menzen:
        # 리치 관련
        is_double_riichi = double_riichi is True or double_riichi == True or (isinstance(double_riichi, str) and double_riichi.lower() == 'true')
        is_reach = reach is True or reach == True or (isinstance(reach, str) and reach.lower() == 'true')
        if double_riichi:
            han += 2
            yaku_list.append({
                'name': '더블리치',
                'han': 2,
                'description': '더블리치가 성립하여 2판 추가'
            })
            print("더블리치가 성립하여 2판 추가")
        elif reach is True:
            han += 1
            yaku_list.append({
                'name': '리치',
                'han': 1,
                'description': '리치가 성립하여 1판 추가'
            })
            print("리치가 성립하여 1판 추가")

            # 핑후 (멘젠 필수)
            if is_pinghu(head, bodies, winning_tile, call, reach, seat, main_round, ankan_sets):
                han += 1
                yaku_list.append({
                    'name': '핑후',
                    'han': 1,
                    'description': '핑후가 성립하여 1판 추가'
                })
                print("핑후가 성립하여 1판 추가")
                is_pinghu_flag = True

        # 이페코 (멘젠 필수)
        if is_peiko(bodies):
            han += 1
            yaku_list.append({
                'name': '이페코',
                'han': 1,
                'description': '이페코가 성립하여 1판 추가'
            })
            print("이페코가 성립하여 1판 추가")
        # 량페코 (멘젠 필수)
        if is_ryanpeiko(bodies):
            han += 3
            yaku_list.append({
                'name': '량페코',
                'han': 3,
                'description': '량페코가 성립하여 3판 추가'
            })
            print("량페코가 성립하여 3판 추가")

        # 멘젠쯔모
        is_tsumo = tsumo is True or tsumo == True or (isinstance(tsumo, str) and tsumo.lower() == 'true')
        if is_tsumo:
            han += 1
            yaku_list.append({
                'name': '멘젠쯔모',
                'han': 1,
                'description': '멘젠쯔모가 성립하여 1판 추가'
            })
            print("멘젠쯔모가 성립하여 1판 추가")


        # 치또이 (멘젠 필수)
        if is_chitoitsu(tiles):
            han += 2
            yaku_list.append({
                'name': '치또이',
                'han': 2,
                'description': '치또이가 성립하여 2판 추가'
            })
            print("치또이가 성립하여 2판 추가")
    # 일반 역들 (멘젠 불필요)
    if is_tanyao(tiles, ankan_sets, ming_sets):
        han += 1
        yaku_list.append({
            'name': '탕야오',
            'han': 1,
            'description': '탕야오가 성립하여 1판 추가'
        })
        print("탕야오가 성립하여 1판 추가")
    
    # 일발 (리치 상태에서만)
    if one_shot is True and (reach is True or double_riichi) and is_menzen:
        han += 1
        yaku_list.append({
            'name': '일발',
            'han': 1,
            'description': '일발이 성립하여 1판 추가'
        })
        print("일발이 성립하여 1판 추가")

    # 자풍/장풍패와 삼원패 체크
    wind_counter = defaultdict(int)
    dragon_counter = defaultdict(int)
    dragons = ['5z', '6z', '7z']  # 백발중
    seat_tile = f"{['east', 'south', 'west', 'north'].index(seat) + 1}z"
    main_round_tile = f"{['east', 'south', 'west', 'north'].index(main_round) + 1}z"

    # 1. 일반 몸통에서 체크
    for body in bodies:
        if len(body) >= 3 and body[0] == body[1]:  # 커쯔/깡
            if body[0].endswith('z'):
                if body[0] in dragons:
                    dragon_counter[body[0]] += len(body)
                else:
                    wind_counter[body[0]] += len(body)

    # 2. 밍커/밍깡 체크
    for ming_type, ming_set in ming_sets:
        tile = ming_set[0]
        if tile.endswith('z'):
            if tile in dragons:
                dragon_counter[tile] += len(ming_set)
            else:
                wind_counter[tile] += len(ming_set)

    # 3. 안깡 체크
    for ankan in ankan_sets:
        tile = ankan[0]
        if tile.endswith('z'):
            if tile in dragons:
                dragon_counter[tile] += 4
            else:
                wind_counter[tile] += 4

    # 자풍패/장풍패 판정
    if wind_counter[seat_tile] >= 3:
        han += 1
        yaku_list.append({
            'name': '자풍패',
            'han': 1,
            'description': '자풍패가 성립하여 1판 추가'
        })
        print("자풍패가 성립하여 1판 추가")

    if wind_counter[main_round_tile] >= 3:
        han += 1
        yaku_list.append({
            'name': '장풍패',
            'han': 1,
            'description': '장풍패가 성립하여 1판 추가'
        })
        print("장풍패가 성립하여 1판 추가")

    # 삼원패 판정
    for dragon in dragons:
        if dragon_counter[dragon] >= 3:
            han += 1
            yaku_list.append({
                'name': f'{dragon} 삼원패',
                'han': 1,
                'description': f'{dragon} 삼원패가 성립하여 1판 추가'
            })
            print(f"{dragon} 삼원패가 성립하여 1판 추가")

    # 상황 역
    if steal_kang is True:
        han += 1
        yaku_list.append({
            'name': '창깡',
            'han': 1,
            'description': '창깡이 성립하여 1판 추가'
        })
        print("창깡이 성립하여 1판 추가")
    
    if flower_on_mount is True:
        han += 1
        yaku_list.append({
            'name': '영상개화',
            'han': 1,
            'description': '영상개화가 성립하여 1판 추가'
        })
        print("영상개화가 성립하여 1판 추가")
    if lastTileWin is True:
        han += 1
        is_tsumo = tsumo is True or tsumo == True or (isinstance(tsumo, str) and tsumo.lower() == 'true')
        if is_tsumo:
            yaku_list.append({
                'name': '해저로월',
                'han': 1,
                'description': '해저로월이 성립하여 1판 추가'
            })
            print("해저로월이 성립하여 1판 추가")
        else:
            yaku_list.append({
                'name': '하저로어',
                'han': 1,
                'description': '하저로어가 성립하여 1판 추가'
            })
            print("하저로어가 성립하여 1판 추가")

    # 또이또이
    if "치또이" not in [yaku['name'] for yaku in yaku_list] and is_toitoi(bodies):
        han += 2
        yaku_list.append({
            'name': '또이또이',
            'han': 2,
            'description': '또이또이가 성립하여 2판 추가'
        })
        print("또이또이가 성립하여 2판 추가")

    # 일기통관
    if is_ikkitsuukan(bodies):
        han_value = 2 if is_menzen else 1
        yaku_list.append({
            'name': '일기통관',
            'han': han_value,
            'description': f'{"멘젠" if is_menzen else "후로"} 일기통관이 성립하여 {han_value}판 추가'
        })
        han += han_value
        print(f"{'멘젠' if is_menzen else '후로'} 일기통관이 성립하여 {han_value}판 추가")
    # 삼색동각
    if is_sanshoku_doukou(bodies):
        han += 2
        yaku_list.append({
            'name': '삼색동각',
            'han': 2,
            'description': '삼색동각이 성립하여 2판 추가'
        })
        print("삼색동각이 성립하여 2판 추가")

    # 삼색동순
    if is_sanshoku_doujun(bodies):
        han_value = 2 if is_menzen else 1
        yaku_list.append({
            'name': '삼색동순',
            'han': han_value,
            'description': f'{"멘젠" if is_menzen else "후로"} 삼색동순이 성립하여 {han_value}판 추가'
        })
        han += han_value
        print(f"{'멘젠' if is_menzen else '후로'} 삼색동순이 성립하여 {han_value}판 추가")

    # 산깡즈
    if is_sankantsu(bodies):
        han += 2
        yaku_list.append({
            'name': '산깡즈',
            'han': 2,
            'description': '산깡즈가 성립하여 2판 추가'
        })
        print("산깡즈가 성립하여 2판 추가")

    # 산안커
    if is_sananko(bodies, winning_tile, tsumo, ming_tiles, ankan_sets):
        han += 2
        yaku_list.append({
            'name': '산안커',
            'han': 2,
            'description': '산안커가 성립하여 2판 추가'
        })
        print("산안커가 성립하여 2판 추가")
    # 소삼원
    if is_shousangen(head ,tiles, ankan_sets, ming_sets):
        han += 2
        yaku_list.append({
            'name': '소삼원',
            'han': 2,
            'description': '소삼원이 성립하여 2판 추가'
        })
        print("소삼원이 성립하여 2판 추가")

    # 혼노두
    if is_honroutou(tiles, ankan_sets, ming_sets):
        han += 2
        yaku_list.append({
            'name': '혼노두',
            'han': 2,
            'description': '혼노두가 성립하여 2판 추가'
        })
        print("혼노두가 성립하여 2판 추가")

    # 준찬타/찬타 판정 부분
    if is_junchanta(bodies, ming_sets, ankan_sets) and is_chanta(head, bodies, ming_sets, ankan_sets):
        han_value = 3 if is_menzen else 2
        yaku_list.append({
            'name': '준찬타',
            'han': han_value,
            'description': f'준찬타가 {"멘젠" if is_menzen else "후로"} 상태에서 성립하여 {han_value}판 추가'
        })
        han += han_value
        print(f"준찬타가 {'멘젠' if is_menzen else '후로'} 상태에서 성립하여 {han_value}판 추가")
    elif is_chanta(head, bodies, ming_sets, ankan_sets):
        han_value = 2 if is_menzen else 1
        yaku_list.append({
            'name': '찬타',
            'han': han_value,
            'description': f'찬타가 {"멘젠" if is_menzen else "후로"} 상태에서 성립하여 {han_value}판 추가'
        })
        han += han_value
        print(f"찬타가 {'멘젠' if is_menzen else '후로'} 상태에서 성립하여 {han_value}판 추가")

    # 청일색 판정 먼저
    if is_chinitsu(head, tiles, ankan_sets, ming_sets):
        han_value = 6 if is_menzen else 5
        yaku_list.append({
            'name': '청일색',
            'han': han_value,
            'description': f'청일색이 {"멘젠" if is_menzen else "후로"} 상태에서 성립하여 {han_value}판 추가'
        })
        han += han_value
        print(f"청일색이 {'멘젠' if is_menzen else '후로'} 상태에서 성립하여 {han_value}판 추가")

    # 혼일색은 청일색이 아닐 때만 체크 (elif 사용)
    elif is_honitsu(tiles, head, ankan_sets, ming_sets):
        han_value = 3 if is_menzen else 2
        yaku_list.append({
            'name': '혼일색',
            'han': han_value,
            'description': f'혼일색이 {"멘젠" if is_menzen else "후로"} 상태에서 성립하여 {han_value}판 추가'
        })
        han += han_value
        print(f"혼일색이 {'멘젠' if is_menzen else '후로'} 상태에서 성립하여 {han_value}판 추가")

    return han, is_pinghu_flag, yakuman_count, yaku_list

def is_yaochuhai(tile):
    """요구패(노두패+자패) 판정"""
    return tile.endswith('z') or tile[0] in '19'

def calculate_fu(head, bodies, tsumo, call, ming_tiles, hidden_kong, is_pinghu, winning_tile):
    if not head or not bodies:
        return 0, ["특수패형(국사무쌍/치또이츠)은 부수 계산 생략"]
    
    print("\n=== 부수 계산 시작 ===")
    fu_details = []  # 부수 계산 과정을 저장할 리스트
    
    # 치또이 체크
    if isinstance(head, str) and head == 'chitoitsu':
        fu = 25
        print(f"치또이 고정 부수: {fu}")
        fu_details.append(f"치또이 고정 부수: {fu}")
        return fu, fu_details
        
    fu = 20  # 기본 부수
    print(f"기본 부수: {fu}")
    fu_details.append(f"기본 부수: {fu}")
    
    # 머리 부수
    if head[0].endswith('z'):  # 자패 머리
        fu += 2
        print(f"자패 머리 +2: {fu}")
        fu_details.append(f"자패 머리 +2: {fu}")
    
    # 몸통 부수와 대기 형태 판정
    waiting_type = None
    for body in bodies:
        print(f"\n몸통 검사: {body}")
        if len(body) == 3:  # 순자/커쯔
            if body[0] == body[1]:  # 커쯔
                if is_yaochuhai(body[0]):  # 요구패(자패/노두패)
                    fu += 8
                    print(f"요구패 커쯔 +8: {fu}")
                    fu_details.append(f"요구패 커쯔 +8: {fu}")
                else:  # 중장패
                    fu += 4
                    print(f"중장패 커쯔 +4: {fu}")
                    fu_details.append(f"중장패 커쯔 +4: {fu}")
            else:  # 순자
                if winning_tile in body:
                    # 대기 형태 판정
                    idx = body.index(winning_tile)
                    if not body[0].endswith('z'):  # 수패만 해당
                        num = int(body[0][0])
                        if idx == 1:  # 간짱
                            waiting_type = 'kanchan'
                        elif (idx == 0 and num == 7) or (idx == 2 and num == 1):  # 변짱
                            waiting_type = 'penchan'
                        elif (idx == 0 and num != 7) or (idx == 2 and num != 1):  # 양면
                            waiting_type = 'ryanmen'
    
    # 대기 형태에 따른 부수 추가
    if waiting_type and not is_pinghu:
        if waiting_type in ['kanchan', 'penchan']:
            fu += 2
            print(f"{waiting_type} 대기 +2: {fu}")
            fu_details.append(f"{waiting_type} 대기 +2: {fu}")
        elif head[0] == winning_tile:  # 단기
            fu += 2
            print(f"단기 대기 +2: {fu}")
            fu_details.append(f"단기 대기 +2: {fu}")
    
    # 밍깡/밍커 부수
    if ming_tiles:
        print("\n밍깡/밍커 검사:")
        fu_details.append("\n밍깡/밍커 검사:")
        for ming_type, ming_set in ming_tiles:
            tile = ming_set[0]
            print(f"- {ming_type}: {ming_set}")
            fu_details.append(f"- {ming_type}: {ming_set}")
            if ming_type == 'kang':  # 밍깡
                if is_yaochuhai(tile):  # 요구패
                    fu += 16
                    print(f"요구패 밍깡 +16: {fu}")
                    fu_details.append(f"요구패 밍깡 +16: {fu}")
                else:  # 중장패
                    fu += 8
                    print(f"중장패 밍깡 +8: {fu}")
                    fu_details.append(f"중장패 밍깡 +8: {fu}")
            elif ming_type == 'pon':  # 밍커
                if is_yaochuhai(tile):  # 요구패
                    fu += 4
                    print(f"요구패 밍커 +4: {fu}")
                    fu_details.append(f"요구패 밍커 +4: {fu}")
                else:  # 중장패
                    fu += 2
                    print(f"중장패 밍커 +2: {fu}")
                    fu_details.append(f"중장패 밍커 +2: {fu}")
    
    # 안깡 부수
    if hidden_kong:
        print("\n안깡 검사:")
        fu_details.append("\n안깡 검사:")
        if isinstance(hidden_kong, list):
            kong_tiles = hidden_kong
        else:
            kong_tiles = hidden_kong.split(',')
        for tile in kong_tiles:
            print(f"- 안깡 타일: {tile}")
            fu_details.append(f"- 안깡 타일: {tile}")
            if is_yaochuhai(tile):  # 요구패
                fu += 32
                print(f"요구패 안깡 +32: {fu}")
                fu_details.append(f"요구패 안깡 +32: {fu}")
            else:  # 중장패
                fu += 16
                print(f"중장패 안깡 +16: {fu}")
                fu_details.append(f"중장패 안깡 +16: {fu}")
                
    # 핑후가 아닐 경우 추가 부수
    if not is_pinghu:
        # 쯔모 여부 확인 (불리언 또는 문자열 'true' 모두 처리)
        is_tsumo = tsumo is True or tsumo == True or (isinstance(tsumo, str) and tsumo.lower() == 'true')
        
        if is_tsumo:
            fu += 2
            print(f"쯔모 +2: {fu}")
            fu_details.append(f"쯔모 +2: {fu}")
        else:
            fu += 10
            print(f"론 +10: {fu}")
            fu_details.append(f"론 +10: {fu}")

    
    # 부수 올림
    original_fu = fu
    fu = ((fu + 9) // 10) * 10
    print(f"\n부수 올림: {original_fu} -> {fu}")
    fu_details.append(f"\n부수 올림: {original_fu} -> {fu}")
    
    return fu, fu_details

def calculate_score(han, fu, seat, tsumo, round=0, akadora_count=0):
    print(f"점수 계산 상세 정보:")
    print(f"한 수(han): {han}, 타입: {type(han)}")
    print(f"부수(fu): {fu}, 타입: {type(fu)}")
    print(f"자리(seat): {seat}")
    print(f"쯔모 여부(tsumo): {tsumo}, 타입: {type(tsumo)}")
    print(f"본장 수(round): {round}")
    print(f"적도라 개수(akadora_count): {akadora_count}")

    # tsumo를 문자열로 변환
    is_tsumo = str(tsumo).lower() == 'true' or tsumo is True or tsumo == True
    print(f"쯔모 판정: {is_tsumo}")

    # 역만 판정 (13판 이상)
    if han >= 13:
        yakuman_count = han // 13
        print(f"역만 배수: {yakuman_count}")

        if seat == "east":  # 친
            base = 48000 * yakuman_count
            if is_tsumo:
                each_payment = 16000 * yakuman_count
                each_payment += 100 * round
                return each_payment, f"{each_payment} 올 역만"
            else:
                final_score = base + (300 * round)
                return final_score, f"{final_score} 역만"
        else:  # 자
            base = 32000 * yakuman_count
            if is_tsumo:
                ko_score = 8000 * yakuman_count
                oya_score = 16000 * yakuman_count
                ko_score += 100 * round
                oya_score += 100 * round
                return ko_score, f"{ko_score},{oya_score} 역만"
            else:
                final_score = base + (300 * round)
                return final_score, f"{final_score} 역만"

    # 만관 판정
    is_mangan = (han >= 5 or (han == 4 and fu >= 40) or (han == 3 and fu >= 70) or fu > 110)
    print(f"\n=== 만관 판정: {is_mangan} ===")
    print(f"조건: 한수 {han}, 부수 {fu}")

    # 점수 판정 순서 수정 (높은 순서대로)
    if han >= 11:  # 삼배만
        print("=== 삼배만 판정 ===")
        if seat == "east":  # 친
            if is_tsumo:
                each_payment = 12000
                each_payment += 100 * round
                return each_payment, f"{each_payment} 올 삼배만"
            else:
                final_score = 36000 + (300 * round)
                return final_score, f"{final_score} 삼배만"
        else:  # 자
            if is_tsumo:
                ko_score = 6000
                oya_score = 12000
                ko_score += 100 * round
                oya_score += 100 * round
                return ko_score, f"{ko_score},{oya_score} 삼배만"
            else:
                final_score = 24000 + (300 * round)
                return final_score, f"{final_score} 삼배만"
    elif han >= 8:  # 배만
        print("=== 배만 판정 ===")        
        if seat == "east":  # 친
            if is_tsumo:
                each_payment = 8000
                each_payment += 100 * round
                return each_payment, f"{each_payment} 올 배만"
            else:
                final_score = 24000 + (300 * round)
                return final_score, f"{final_score} 배만"
        else:  # 자
            if is_tsumo:
                ko_score = 4000
                oya_score = 8000
                ko_score += 100 * round
                oya_score += 100 * round
                return ko_score, f"{ko_score},{oya_score} 배만"
            else:
                final_score = 16000 + (300 * round)
                return final_score, f"{final_score} 배만"
    elif han >= 6:  # 하네만
        print("=== 하네만 판정 ===")
        if seat == "east":  # 친
            if is_tsumo:
                each_payment = 6000
                each_payment += 100 * round
                return each_payment, f"{each_payment} 올 하네만"
            else:
                final_score = 18000 + (300 * round)
                return final_score, f"{final_score} 하네만"
        else:  # 자
            if is_tsumo:
                ko_score = 3000
                oya_score = 6000
                ko_score += 100 * round
                oya_score += 100 * round
                return ko_score, f"{ko_score},{oya_score} 하네만"
            else:
                final_score = 12000 + (300 * round)
                return final_score, f"{final_score} 하네만"
    elif is_mangan:  # 만관
        print("=== 만관 판정 ===")
        if seat == "east":  # 친
            if is_tsumo:
                each_payment = 4000
                each_payment += 100 * round
                return each_payment, f"{each_payment} 올 만관"
            else:
                final_score = 12000 + (300 * round)
                return final_score, f"{final_score} 만관"
        else:  # 자
            if is_tsumo:
                ko_score = 2000
                oya_score = 4000
                ko_score += 100 * round
                oya_score += 100 * round
                return ko_score, f"{ko_score},{oya_score} 만관"
            else:
                final_score = 8000 + (300 * round)
                return final_score, f"{final_score} 만관"
    else:
        print("=== 만관 미만 판정 ===")
        # 만관 미만 점수 처리
        score_table = tsumo_score_table if is_tsumo else ron_score_table
        if fu in score_table and han in score_table[fu]:
            score_info = score_table[fu][han]
            if seat == "east" and is_tsumo:
                base_score = score_info['oya']
            else:
                base_score = score_info['oya'] if seat == "east" else score_info['ko']
            
            if base_score is None:
                return 0, "점수를 계산할 수 없습니다"
                 
            try:
                if is_tsumo:
                    if seat == "east":
                        each_payment = int(base_score)
                        each_payment += 100 * round
                        return each_payment, f"{each_payment} 올 {fu}부 {han}판"
                    else:
                        ko_score, oya_score = map(int, base_score.split(','))
                        ko_score += 100 * round
                        oya_score += 100 * round
                        return ko_score, f"{ko_score},{oya_score} {fu}부 {han}판"
                else:
                    total_score = int(base_score) + (300 * round)
                    return total_score, f"{total_score} {fu}부 {han}판"
            except ValueError:
                return 0, "점수를 계산할 수 없습니다"
        
        return 0, "점수를 계산할 수 없습니다"

# 점수 테이블 정의
tsumo_score_table = {
    20: {
        1: {'ko': '100,200', 'oya': '200'},
        2: {'ko': '200,400', 'oya': '400'},
        3: {'ko': '400,800', 'oya': '800'},
        4: {'ko': '800,1600', 'oya': '1600'}
    },
    25: {
        1: {'ko': '200,300', 'oya': '300'},
        2: {'ko': '300,600', 'oya': '600'},
        3: {'ko': '600,1200', 'oya': '1200'},
        4: {'ko': '1200,2400', 'oya': '2400'}
    },
    30: {
        1: {'ko': '200,400', 'oya': '400'},
        2: {'ko': '400,800', 'oya': '800'},
        3: {'ko': '800,1600', 'oya': '1600'},
        4: {'ko': '1600,3200', 'oya': '3200'}
    },
    40: {
        1: {'ko': '300,500', 'oya': '500'},
        2: {'ko': '500,1000', 'oya': '1000'},
        3: {'ko': '1000,2000', 'oya': '2000'},
        4: {'ko': '2000,4000', 'oya': '4000'}
    },
    50: {
        1: {'ko': '400,700', 'oya': '700'},
        2: {'ko': '700,1300', 'oya': '1300'},
        3: {'ko': '1300,2600', 'oya': '2600'},
        4: {'ko': '2600,5200', 'oya': '5200'}
    },
    60: {
        1: {'ko': '400,800', 'oya': '800'},
        2: {'ko': '800,1600', 'oya': '1600'},
        3: {'ko': '1600,3200', 'oya': '3200'},
        4: {'ko': '3200,6400', 'oya': '6400'}
    },
    70: {
        1: {'ko': '500,1000', 'oya': '1000'},
        2: {'ko': '1000,2000', 'oya': '2000'},
        3: {'ko': '2000,4000', 'oya': '4000'}
    },
    80: {
        1: {'ko': '600,1200', 'oya': '1200'},
        2: {'ko': '1200,2400', 'oya': '2400'},
        3: {'ko': '2400,4800', 'oya': '4800'}
    },
    90: {
        1: {'ko': '700,1300', 'oya': '1300'},
        2: {'ko': '1300,2600', 'oya': '2600'},
        3: {'ko': '2600,5200', 'oya': '5200'}
    },
    100: {
        1: {'ko': '800,1500', 'oya': '1500'},
        2: {'ko': '1500,3000', 'oya': '3000'},
        3: {'ko': '3000,6000', 'oya': '6000'}
    },
    110: {
        1: {'ko': '800,1600', 'oya': '1600'},
        2: {'ko': '1600,3200', 'oya': '3200'},
        3: {'ko': '3200,6400', 'oya': '6400'}
    }
}

ron_score_table = {
    20: {
        1: {'ko': '300', 'oya': '500'},
        2: {'ko': '600', 'oya': '1000'},
        3: {'ko': '1200', 'oya': '2000'},
        4: {'ko': '2400', 'oya': '4000'}
    },
    25: {
        1: {'ko': '400', 'oya': '700'},
        2: {'ko': '800', 'oya': '1400'},
        3: {'ko': '1600', 'oya': '2800'},
        4: {'ko': '3200', 'oya': '5600'}
    },
    30: {
        1: {'ko': '500', 'oya': '800'},
        2: {'ko': '1000', 'oya': '1600'},
        3: {'ko': '2000', 'oya': '3200'},
        4: {'ko': '4000', 'oya': '6400'}
    },
    40: {
        1: {'ko': '700', 'oya': '1000'},
        2: {'ko': '1300', 'oya': '2000'},
        3: {'ko': '2600', 'oya': '4000'},
        4: {'ko': '5200', 'oya': '8000'}
    },
    50: {
        1: {'ko': '800', 'oya': '1200'},
        2: {'ko': '1600', 'oya': '2400'},
        3: {'ko': '3200', 'oya': '4800'},
        4: {'ko': '6400', 'oya': '9600'}
    },
    60: {
        1: {'ko': '1000', 'oya': '1500'},
        2: {'ko': '2000', 'oya': '3000'},
        3: {'ko': '4000', 'oya': '6000'},
        4: {'ko': '8000', 'oya': '12000'}
    },
    70: {
        1: {'ko': '1200', 'oya': '1800'},
        2: {'ko': '2400', 'oya': '3600'},
        3: {'ko': '4800', 'oya': '7200'}
    },
    80: {
        1: {'ko': '1300', 'oya': '2000'},
        2: {'ko': '2600', 'oya': '4000'},
        3: {'ko': '5200', 'oya': '8000'}
    },
    90: {
        1: {'ko': '1500', 'oya': '2300'},
        2: {'ko': '2900', 'oya': '4500'},
        3: {'ko': '5800', 'oya': '9000'}
    },
    100: {
        1: {'ko': '1600', 'oya': '2400'},
        2: {'ko': '3200', 'oya': '4800'},
        3: {'ko': '6400', 'oya': '9600'}
    },
    110: {
        1: {'ko': '1800', 'oya': '2700'},
        2: {'ko': '3600', 'oya': '5400'},
        3: {'ko': '7200', 'oya': '10800'}
    }
}

def calculate_final_score(additional_info=None):
    """
    JSON 파일에서 감지된 타일을 읽어 점수 계산
    Args:
        additional_info: 추가 입력 정보 딕셔너리
    """
    try:
        print("\n=== 점수 계산 함수 호출 시 추가 정보 디버깅 ===")
        print("추가 정보:", additional_info)
        print("쯔모 값:", additional_info.get('tsumo'))
        print("리치 값:", additional_info.get('riichi'))
        print("더블리치 값:", additional_info.get('double_riichi'))
        print("일발 값:", additional_info.get('one_shot'))
        print("해저/하저 값:", additional_info.get('last_tile_win'))
        print("천화/지화 값:", additional_info.get('tian_hu_di_hu'))
        print("쯔모 값 타입:", type(additional_info.get('tsumo')))
        
        print("\n=== 최종 점수 계산 시작 ===")
        print(f"추가 정보: {additional_info}")
        
        # JSON 파일에서 감지된 타일 읽기
        with open('detected_tiles.json', 'r') as f:
            detected_tiles = json.load(f)
        print(f"\n감지된 타일 정보: {detected_tiles}")
        
        if not detected_tiles:
            print("타일을 감지할 수 없음")
            return {
                "error": "타일을 감지할 수 없습니다", 
                "status": "error"
            }

        # 기본값 설정
        if additional_info is None:
            additional_info = {
                'main_round': 'east',
                'round_count': 0,
                'seat': 'east',
                'tsumo': 'false',
                'riichi': 'false',
                'one_shot': 'false',
                'flower_on_mount': 'false',
                'last_tile_win': 'false',
                'steal_kang': 'false',
                'tian_hu_di_hu': 'false',
                'dora_indicators': [],
                'ankan_tiles': [],
                'winning_tile': '',
                'ming_tiles': []
            }

        tiles = detected_tiles[0]
        print(f"처리할 타일 목록: {tiles}")
        # 밍깡/밍커 처리
        ming_tiles_list = additional_info.get('ming_tiles', [])
        print(f"\n=== 밍깡/밍커 처리 ===")
        print(f"밍 타일 목록: {ming_tiles_list}")
        print(f"처리 전 타일: {detected_tiles[0]}")
        ming_sets, remaining_after_ming = handle_ming_tiles(detected_tiles[0], ming_tiles_list)
        print(f"밍 세트: {ming_sets}")
        print(f"처리 후 타일: {remaining_after_ming}")
        
        # 안깡 처리
        ankan_tiles = additional_info.get('ankan_tiles', [])
        print(f"\n=== 안깡 처리 ===")
        print(f"안깡 타일 목록: {ankan_tiles}")
        print(f"처리 전 타일: {remaining_after_ming}")
        ankan_sets, remaining_tiles, used_tiles = handle_ankan(remaining_after_ming, ankan_tiles)
        print(f"안깡 세트: {ankan_sets}")
        print(f"처리 후 타일: {remaining_tiles}")
        print(f"사용된 타일: {used_tiles}")
        
        # 뒷면 타일 제거
        processed_tiles = handle_back_tiles(remaining_tiles)
        print(f"\n=== 뒷면 제거 후 타일: {processed_tiles} ===")
        
        # 적도라 처리
        akadora_count, final_tiles = count_akadora(processed_tiles)
        print(f"\n=== 적도라 처리 ===")
        print(f"적도라 개수: {akadora_count}")
        print(f"최종 타일: {final_tiles}")
        
        # 패 조합 찾기
        print("\n=== 패 조합 찾기 ===")
        result = find_mahjong_sets(
            final_tiles,
            ankan_sets,
            ming_sets,
            additional_info.get('winning_tile')
        )
        if not result or result.get('type') == 'invalid':
            print("유효한 패 조합을 찾을 수 없음")
            return {
                "error": "유효한 패 조합을 찾을 수 없습니다", 
                "status": "error",
                "detected_tiles": detected_tiles[0]
            }
        
        # 특수패턴 분기
        if result['type'] == 'chitoitsu':
            # 1. 기본 변수 초기화
            head = []
            bodies = []
            yakuman_count = 0  # 치또이츠는 역만이 아님
            han = 2
            fu = 25
            yaku_list = [{'name': '치또이츠', 'han': 2, 'description': '7개의 서로 다른 페어'}]
            fu_details = ["치또이츠 고정 부수: 25"]
            
            # 2. 추가 역 검사 (치또이츠와 호환되는 역만)
            han_from_check, _, _, yaku_from_check = check_yaku(
                head=head,
                bodies=bodies,
                tiles=final_tiles,
                winning_tile=additional_info.get('winning_tile'),
                call='false',  # 치또이츠는 항상 멘젠
                reach=additional_info.get('riichi', 'false'),
                seat=additional_info.get('seat'),
                main_round=additional_info.get('main_round'),
                isTianHuDiHu=additional_info.get('tian_hu_di_hu', 'false'),
                dora_marker=additional_info.get('dora_indicators', []),
                one_shot=additional_info.get('one_shot', 'false'),
                tsumo=additional_info.get('tsumo', 'false'),
                steal_kang=additional_info.get('steal_kang', 'false'),
                flower_on_mount=additional_info.get('flower_on_mount', 'false'),
                ming_tiles=[],
                lastTileWin=additional_info.get('last_tile_win', 'false'),
                ankan_sets=[],
                ming_sets=[],
                double_riichi=additional_info.get('double_riichi', False)
            )
            # 3. 탕야오 수동 검사 (중장패만 있는지)
            if all(tile[0] not in '19' and not tile.endswith('z') for tile in final_tiles):
                han += 1
                yaku_list.append({'name': '탕야오', 'han': 1, 'description': '중장패만 사용'})

            # 4. 혼일색/청일색 수동 검사
            suits = {tile[1] for tile in final_tiles if not tile.endswith('z')}
            if len(suits) == 1:  # 단일 수패 종류
                has_jihai = any(t.endswith('z') for t in final_tiles)
                if has_jihai:
                    # 혼일색 (수패 1종 + 자패)
                    han += 3
                    yaku_list.append({'name': '혼일색', 'han': 3, 'description': f'혼일색이 멘젠 상태에서 성립하여 3판 추가'})
                else:
                    # 청일색 (수패 1종 + 자패 없음 → 역만)
                    han += 6
                    yaku_list.append({'name': '청일색', 'han': 6, 'description': f'청일색이 멘젠 상태에서 성립하여 3판 추가'})
            
            # 3. 결과 병합 (치또이츠 + 추가 역)
            han += han_from_check
            yaku_list += yaku_from_check
            
            # 4. 도라/적도라 처리
            dora_count = calculate_dora_count(final_tiles, additional_info.get('dora_indicators', []))
            if dora_count > 0:
                yaku_list.append({'name': '도라', 'han': dora_count, 'description': f'도라 {dora_count}개로 {dora_count}판 추가'})
            if akadora_count > 0:
                yaku_list.append({'name': '적도라', 'han': akadora_count, 'description': f'적도라 {akadora_count}개로 {akadora_count}판 추가'})

        elif result['type'] == 'kokushi_13wait':  # 13면대기 먼저 체크
            head = []
            bodies = []
            han = 26  # 2배 역만
            fu = 0
            yaku_list = [{
                'name': '국사무쌍 13면 대기',
                'han': 26,
                'description': '13종 요구패 + 모든 패가 대기 가능'
            }]
            yakuman_count = 2  # 2배 역만
            is_pinghu_flag = False
            fu_details = []
            dora_count = 0

        elif result['type'] == 'kokushi':
            head = []
            bodies = []
            han = 13
            fu = 0
            yaku_list = [{
                'name': '국사무쌍',
                'han': 13,
                'description': '13종의 요구패 + 1장을 모음'
            }]
            yakuman_count = 1
            is_pinghu_flag = False
            fu_details = []
            dora_count = 0  # 역만은 도라 무효

        else:
            # 일반패턴
            head = result.get('head', [])
            bodies = result.get('bodies', [])
            print(f"찾은 패 조합 - 머리: {head}, 몸통: {bodies}")
            dora_indicators = additional_info.get('dora_indicators', [])
            print("\n=== 점수 계산 시작 ===")
            han, is_pinghu_flag, yakuman_count, yaku_list = check_yaku(
                head=head,
                bodies=bodies,
                tiles=final_tiles,
                winning_tile=additional_info.get('winning_tile'),
                call='true' if ming_sets else 'false',
                reach=additional_info.get('riichi', 'false'),
                seat=additional_info.get('seat'),
                main_round=additional_info.get('main_round'),
                isTianHuDiHu=additional_info.get('tian_hu_di_hu', 'false'),
                dora_marker=dora_indicators,
                one_shot=additional_info.get('one_shot', 'false'),
                tsumo=additional_info.get('tsumo', 'false'),
                steal_kang=additional_info.get('steal_kang', 'false'),
                flower_on_mount=additional_info.get('flower_on_mount', 'false'),
                ming_tiles=additional_info.get('ming_tiles', []),
                lastTileWin=additional_info.get('last_tile_win', 'false'),
                ankan_sets=ankan_sets,
                ming_sets=ming_sets,
                double_riichi=additional_info.get('double_riichi', False)
            )
            print(f"기본 한수: {han}")
            dora_count = calculate_dora_count(
                final_tiles,
                additional_info.get('dora_indicators', []),
                ankan_sets,  # 안깡 세트 전달
                ming_sets     # 밍깡/밍커 세트 전달
            )
            if yakuman_count == 0:  # 역만일 경우 도라 무효
                # 도라 추가
                if dora_count > 0:
                    yaku_list.append({
                        'name': '도라',
                        'han': dora_count,
                        'description': f'도라 {dora_count}개로 {dora_count}판 추가'
                    })
                    han += dora_count
                
                # 적도라 추가
                if akadora_count > 0:
                    yaku_list.append({
                        'name': '적도라',
                        'han': akadora_count,
                        'description': f'적도라 {akadora_count}개로 {akadora_count}판 추가'
                    })
                    han += akadora_count
            print("\n=== 부수 계산 시작 ===")
            fu, fu_details = calculate_fu(
                head=head,
                bodies=bodies,
                tsumo=additional_info.get('tsumo', 'false'),
                call='true' if ming_sets else 'false',
                ming_tiles=ming_sets,
                hidden_kong=ankan_tiles,
                is_pinghu=is_pinghu_flag,
                winning_tile=additional_info.get('winning_tile')
            )
            print(f"계산된 부수: {fu}")
            print("부수 계산 상세:", fu_details)

        # 최종 점수 계산
        print("\n=== 최종 점수 계산 ===")
        if yakuman_count > 0:
            final_score, payment_details = calculate_score(
                han=han,
                fu=0,
                seat=additional_info.get('seat', 'east'),
                tsumo=additional_info.get('tsumo', 'false'),
                round=additional_info.get('round_count', 0),
                akadora_count=0
            )
        else:
            final_score, payment_details = calculate_score(
                han=han,
                fu=fu,
                seat=additional_info.get('seat', 'east'),
                tsumo=additional_info.get('tsumo', 'false'),
                round=additional_info.get('round_count', 0),
                akadora_count=akadora_count
            )
        print(f"최종 점수: {final_score}")
        print(f"지불 방식: {payment_details}")

        return {
            "status": "success",
            "tiles": detected_tiles[0],
            "head": head,
            "bodies": bodies,
            "yakus": yaku_list,
            "han": han,
            "fu": fu,
            "fu_details": fu_details,
            "dora_count": dora_count,
            "final_score": final_score,
            "payment_details": payment_details,
            "combinations": {
                "head": head,
                "bodies": bodies
            },
            "additional_info": {
                "dora_indicators": additional_info.get('dora_indicators', []),
                "ankan_tiles": ankan_tiles,
                "ming_sets": ming_sets,
                "winning_tile": additional_info.get('winning_tile', ''),
                "tsumo": additional_info.get('tsumo', 'false'),
                "riichi": additional_info.get('riichi', 'false')
            }
        }
        
    except Exception as e:
        print(f"\n=== 전체 처리 과정 중 오류 발생 ===")
        print(f"오류 종류: {type(e).__name__}")
        print(f"오류 내용: {str(e)}")
        import traceback
        error_details = traceback.format_exc()
        print(f"상세 오류:\n{error_details}")
        return {
            "error": f"점수 계산 중 오류 발생: {str(e)}", 
            "details": error_details,
            "status": "error",
            "detected_tiles": detected_tiles[0] if 'detected_tiles' in locals() else None
        }

def calculate_dora_count(tiles, dora_indicators, ankan_tiles=[], ming_sets=[]):
    """도라 개수 계산 (적도라 표시패 처리 추가)"""
    print("\n=== 도라 개수 계산 시작 ===")
    
    # 1. 적도라 표시패 클리닝 (r 제거)
    cleaned_indicators = []
    for indicator in dora_indicators:
        if indicator.startswith('r'):
            cleaned = indicator[1:]
            if cleaned:  # 유효한 표시패만 추가
                cleaned_indicators.append(cleaned)
        else:
            cleaned_indicators.append(indicator)
    
    if not cleaned_indicators:  # ✅ 클리닝된 리스트 사용
        return 0
    
    # 2. 클리닝된 표시패로 도라 계산
    dora_count = 0
    for indicator in cleaned_indicators:  # ✅ 수정된 부분
        # 3. 다음 도라 계산 함수 호출 (get_next_dora도 수정 필요)
        next_dora = get_next_dora(indicator)
        if not next_dora:
            continue
            
        # 4. 도라 개수 카운트 (안깡/밍커 포함)
        total_tiles = tiles + [t for ankan in ankan_tiles for t in ankan] 
        for ming_set in ming_sets:
            total_tiles.extend(ming_set[1])
            
        dora_count += total_tiles.count(next_dora)
    
    return dora_count


def get_next_dora(dora):
    """적도라 표시패 처리 추가"""
    if not dora or len(dora) < 2:
        return None
        
    # 적도라 표시패 처리 (r 제거)    
    clean_dora = dora[1:] if dora.startswith('r') else dora  # ✅ 추가된 처리
    
    number = int(clean_dora[0])  # ✅ 이제 숫자 변환 가능
    suit = clean_dora[1]
    
    # 나머지 로직은 동일
    if suit == 'z':
        if number <= 4: return f"{(number%4)+1}z"
        else: return f"{(number-3)%3+5}z"
    else:
        return f"{number%9+1}{suit}"

def is_pinghu(head, bodies, winning_tile, call, reach, seat, main_round, ankan_sets=None):
    """핑후 판정 (최종 검증 버전)"""
    # 1. 울음/안깡 차단
    if call or (ankan_sets and len(ankan_sets) > 0):
        return False

    # 2. 머리 풍패 조건 검증
    head_tile = head[0]
    if head_tile.endswith('z'):
        num = head_tile[0]
        if num in {'5','6','7'}:  # 삼원패 차단
            return False
        wind_num = int(num)
        current_wind = ['east','south','west','north'][wind_num-1]
        if current_wind in {seat, main_round}:  # 자풍/장풍패 차단
            return False

    # 3. 화료패가 머리에 포함된 경우 핑후 불가
    if winning_tile in head:
        return False

    # 4. 모든 몸통 슌쯔 확인
    for body in bodies:
        if len(body)!=3 or body[0]==body[1]:
            return False

    # 5. 양면대기 검증 (정확한 숫자 범위 체크)
    ryanmen_flag = False
    for body in bodies:
        if winning_tile in body and not body[0].endswith('z'):
            idx = body.index(winning_tile)
            num = int(body[0][0])  # 슌쯔 첫 번째 타일 숫자
            
            # 양면대기 가능 조건 (핵심 수정 부분)
            if (idx == 0 and num >= 2) or (idx == 2 and num <= 6):
                ryanmen_flag = True
                break

    return ryanmen_flag

def is_tanyao(tiles, ankan_sets, ming_sets):
    """탕야오 판정 (전체 타일 검증)"""
    # 모든 타일 수집
    all_tiles = []
    # 1. 기본 타일
    all_tiles.extend(tiles)
    # 2. 안깡 타일 추가
    for ankan in ankan_sets:
        all_tiles.extend(ankan)
    # 3. 밍커/밍깡 타일 추가
    for ming_type, ming_tiles in ming_sets:
        all_tiles.extend(ming_tiles)
    
    # 노두패(1/9)나 자패 존재 여부 확인
    return not any(
        tile.endswith('z') or tile[0] in '19'
        for tile in all_tiles
    )

def get_next_dora(dora):
    if not dora or len(dora) < 2:  # 입력값 검증 추가
        return None
        
    number = int(dora[0])
    suit = dora[1]

    # 적도라 표시패 처리 (r 제거)
    clean_dora = dora[1:] if dora.startswith('r') else dora
    number = int(clean_dora[0])
    suit = clean_dora[1]
    
    if suit == 'z':  # 자패
        if number <= 4:  # 동남서북
            return f"{1 if number == 4 else number + 1}z"
        else:  # 백발중
            return f"{5 if number == 7 else number + 1}z"
    else:  # 수패
        return f"{1 if number == 9 else number + 1}{suit}"

def count_dora(tiles, dora_marker):
    cleaned_markers = [m[1:] if m.startswith('r') else m for m in dora_marker]
    dora_count = 0
    dora_tiles = [get_next_dora(dora) for dora in dora_marker if get_next_dora(dora)]
    for dora_tile in dora_tiles:
        dora_count += tiles.count(dora_tile)
    return dora_count

# 치또이츠 (7쌍의 또이츠)
def is_chitoitsu(tiles):
    """치또이 판정"""
    if len(tiles) != 14:
        return False
    
    pairs = []
    current_pair = []
    
    sorted_tiles = sorted(tiles)
    for tile in sorted_tiles:
        if not current_pair:
            current_pair.append(tile)
        else:
            if current_pair[0] == tile:
                pairs.append(current_pair + [tile])
                current_pair = []
            else:
                return False
    
    return len(pairs) == 7

# 또이또이 (치또이가 없을 때만 인정) 수정
def is_toitoi(bodies):
    """또이또이 판정"""
    # 모든 몸통이 커쯔인지 확인
    for body in bodies:
        # 슌쯔가 있으면 또이또이가 아님
        if len(body) == 3 and not (body[0] == body[1] == body[2]):
            return False
    return True

# 일기통관 (같은 습패에서 순서대로 슌쯔가 있는지 확인)
def is_ikkitsuukan(bodies):
    """일기통관 판정"""
    # 슌쯔만 추출하고 정렬
    shuntsu = sorted([body for body in bodies if len(body) == 3 and body[0] != body[1]])
    if len(shuntsu) < 3:
        return False
    
    # 같은 종류의 123, 456, 789 슌쯔가 있는지 확인
    for i in range(len(shuntsu)-2):
        first = shuntsu[i]
        if not first[0].endswith('z') and first[0][0] == '1':
            suit = first[0][-1]
            second = [f"4{suit}", f"5{suit}", f"6{suit}"]
            third = [f"7{suit}", f"8{suit}", f"9{suit}"]
            if second in shuntsu and third in shuntsu:
                return True
    return False

# 삼색동각 (세 개의 다른 습패에서 동일한 커쯔가 있는지 확인)
def is_sanshoku_doukou(bodies):
    # 커쯔(3장) 또는 깡쯔(4장) 추출
    kotsu = [body for body in bodies 
             if (len(body) == 3 or len(body) == 4) 
             and body[0] == body[1] == body[2]]
    
    # 숫자별로 종류 수집
    number_suits = defaultdict(set)
    for body in kotsu:
        num = body[0][0]
        suit = body[0][-1]
        number_suits[num].add(suit)
    
    # 세 종류 모두 존재하는 숫자 확인
    return any(len(suits) == 3 for suits in number_suits.values())

# 삼색동순 (세 개의 다른 습패에서 동일한 슌쯔가 있는지 확인)
def is_sanshoku_doujun(bodies):
    """삼색동순 판정"""
    # 슌쯔만 추출
    shuntsu = [body for body in bodies if len(body) == 3 and body[0] != body[1]]
    if len(shuntsu) < 3:
        return False
    
    # 같은 숫자의 만/삭/통 슌쯔가 있는지 확인
    for i in range(len(shuntsu)):
        first = shuntsu[i]
        if not first[0].endswith('z'):
            num = first[0][0]
            # 다른 두 종류의 같은 슌쯔 찾기
            found_suits = {first[0][-1]}
            for j in range(len(shuntsu)):
                if i != j:
                    other = shuntsu[j]
                    if (other[0][0] == num and 
                        other[0][-1] not in found_suits):
                        found_suits.add(other[0][-1])
            if len(found_suits) == 3:
                return True
    return False

# 산깡즈 (세 개의 깡쯔가 있는지 확인)
def is_sankantsu(bodies):
    """산깡즈 판정"""
    kong_count = sum(1 for body in bodies if len(body) == 4)
    return kong_count >= 3

# 산안커 (세 개의 안커가 있는지 확인)
def is_sananko(bodies, winning_tile, tsumo, ming_tiles, ankan_sets):
    """삼안커 판정 (안깡 포함)"""
    ming_tiles_list = ming_tiles if isinstance(ming_tiles, list) else []
    anko_count = 0
    
    # 1. 일반 몸통에서 안커 카운트
    for body in bodies:
        if len(body)>=3 and body[0]==body[1]==body[2]:
            if body[0] not in ming_tiles_list:
                # 론 화료시 해당 트리플릿 제외
                if not tsumo and winning_tile in body:
                    continue
                anko_count +=1
                
    # 2. 안깡 추가 (4장→3장으로 카운트)
    anko_count += len(ankan_sets)
    
    return anko_count >= 3


# 소삼원 (백/발/중 중 두 개 이상의 커쯔 또는 깡쯔가 있는지 확인)
def is_shousangen(head, bodies, ankan_sets, ming_sets):
    dragons = {'5z':0, '6z':0, '7z':0}
    
    # 1. 머리 처리 (2장 추가)
    if head and head[0] in dragons:
        dragons[head[0]] += 2
        
    # 2. 일반 몸통 처리
    for body in bodies:
        tile = body[0]
        if tile in dragons:
            dragons[tile] += len(body)
    
    # 3. 안깡 처리 (4장 추가)
    for ankan in ankan_sets:
        tile = ankan[0]
        if tile in dragons:
            dragons[tile] +=4
    
    # 4. 밍커 처리 (3장 추가)
    for ming_type, ming_set in ming_sets:
        tile = ming_set[0]
        if tile in dragons:
            dragons[tile] += len(ming_set)
    
    # 최종 조건: 2종 3장 + 1종 2장
    return sum(v >=3 for v in dragons.values()) >=2 and sum(v ==2 for v in dragons.values()) >=1

# 혼노두 (모든 패가 자패 또는 노두패로만 구성되어 있는지 확인)
def is_honroutou(tiles, ankan_sets, ming_sets):
    """혼노두 판정 (밍깡/안깡 포함)"""
    all_tiles = tiles.copy()
    
    # 안깡 추가
    for ankan in ankan_sets:
        all_tiles.extend(ankan)
    
    # 밍깡/밍커 추가
    for ming_type, ming_set in ming_sets:
        all_tiles.extend(ming_set)
    
    return all(t.endswith('z') or t[0] in '19' for t in all_tiles)

# 찬타 (모든 몸통에 자패 또는 노두패 포함, 머리도 포함)
def is_chanta(head, bodies, ming_sets, ankan_sets):
    """찬타 판정 (밍커/안깡 포함 검증)"""
    all_groups = bodies.copy()
    
    if not head:  # ✅ 치또이츠 경우 처리
        return False
    
    # 1. 밍커 추가
    for ming_type, ming_tiles in ming_sets:
        all_groups.append(ming_tiles)
    
    # 2. 안깡 추가
    for ankan in ankan_sets:
        all_groups.append(ankan)
    
    # 3. 머리 검사
    if not is_yaochuhai(head[0]):
        return False
    
    # 4. 각 그룹 검증
    for group in all_groups:
        has_yaochu = any(is_yaochuhai(t) for t in group)
        if not has_yaochu:
            return False
    
    return True

# 량페코 (두 개의 이페코 - 멘젠이어야 함)
def is_ryanpeiko(bodies):
    """량페코 판정"""
    # 슌쯔만 추출
    shuntsu = [body for body in bodies if len(body) == 3 and body[0] != body[1]]
    
    # 같은 슌쯔 쌍이 2개 있는지 확인
    if len(shuntsu) < 4:
        return False
        
    pairs = []
    used = set()
    for i in range(len(shuntsu)):
        if i in used:
            continue
        for j in range(i + 1, len(shuntsu)):
            if j in used:
                continue
            if shuntsu[i] == shuntsu[j]:
                pairs.append((i, j))
                used.add(i)
                used.add(j)
                break
    
    return len(pairs) == 2

# 준찬타 (모든 몸통에 자패 없이 노두패 포함)
def is_junchanta(bodies, ming_sets, ankan_sets):
    """준찬타 판정 (밍커/안깡 포함 검증)"""
    all_groups = bodies.copy()
    
    # 1. 밍커 추가
    for ming_type, ming_tiles in ming_sets:
        all_groups.append(ming_tiles)
    
    # 2. 안깡 추가
    for ankan in ankan_sets:
        all_groups.append(ankan)
    
    # 3. 각 그룹 검증
    for group in all_groups:
        # 자패 존재 시 실패
        if any(t.endswith('z') for t in group):
            return False
        
        # 슌쯔: 1-2-3 또는 7-8-9만 허용
        if len(group) == 3 and group[0] != group[1]:
            nums = sorted(int(t[0]) for t in group)
            if nums not in [[1,2,3], [7,8,9]]:
                return False
        
        # 커쯔: 1/9만 허용
        elif len(group) >=3 and group[0] == group[1]:
            if group[0][0] not in ('1','9'):
                return False
    
    return True

def is_honitsu(tiles, head, ankan_sets, ming_sets):
    """혼일색 판정 (적도라 변환 및 뒷면 제거 후)"""
    if not head:  # ✅ 치또이츠 경우 처리
        return False
    # 모든 타일 수집
    all_tiles = []
    # 1. 기본 타일
    all_tiles.extend(tiles)
    # 2. 안깡 타일 추가
    for ankan in ankan_sets:
        all_tiles.extend(ankan)
    # 3. 밍커/밍깡 타일 추가
    for ming_type, ming_tiles in ming_sets:
        all_tiles.extend(ming_tiles)
    
    # 수패 추출 및 종류 확인
    number_tiles = [t for t in all_tiles if not t.endswith('z')]
    return len({t[-1] for t in number_tiles}) == 1  # 모든 수패가 동일 종류

def is_chinitsu(head, tiles, ankan_sets, ming_sets):
    if not head:  # ✅ 치또이츠 경우 처리
        return False
    
    """청일색 판정"""
    # 모든 타일 수집
    all_tiles = []
    # 1. 기본 타일
    all_tiles.extend(tiles)
    # 2. 안깡 타일 추가
    for ankan in ankan_sets:
        all_tiles.extend(ankan)
    # 3. 밍커/밍깡 타일 추가
    for ming_type, ming_tiles in ming_sets:
        all_tiles.extend(ming_tiles)
    
    # 자패 존재 여부 확인
    if any(t.endswith('z') for t in all_tiles):
        return False
    
    # 모든 수패가 동일 종류인지 확인
    suits = {t[-1] for t in all_tiles}
    return len(suits) == 1

def is_kokushi_13wait(tiles, winning_tile):
    """국사무쌍 13면대기 (Double Yakuman)"""
    required = {'1m','9m','1p','9p','1s','9s','1z','2z','3z','4z','5z','6z','7z'}
    counts = Counter(tiles)
    
    # 1. 13종 모두 존재
    if not (counts.keys() >= required):
        return False
    
    # 2. 화료패가 required에 포함되고 2장 존재
    if winning_tile not in required or counts[winning_tile] != 2:
        return False
    
    # 3. 나머지 12종은 1장씩
    for t in required:
        if t != winning_tile and counts.get(t, 0) != 1:
            return False
    
    return len(tiles) == 14

def is_kokushi_musou(tiles):
    """일반 국사무쌍 (Single Yakuman)"""
    required = {'1m','9m','1p','9p','1s','9s','1z','2z','3z','4z','5z','6z','7z'}
    counts = Counter(tiles)
    return (counts.keys() == required) and any(v == 2 for v in counts.values())

def is_suuankou(bodies, ankan_sets, ming_sets, winning_tile, tsumo):
    """스안커 판정 로직 개선"""
    ming_tiles_list = [t for _, tiles in ming_sets for t in tiles]
    anko_count = 0

    # 1. 일반 몸통 검사 (커쯔만 카운트)
    for body in bodies:
        # 슌쯔(順子)는 제외
        if len(body) == 3 and body[0] == body[1] == body[2]:  # 커쯔만 처리
            if body[0] not in ming_tiles_list:
                # 론 화료시 해당 트리플릿 제외
                if not tsumo and winning_tile in body:
                    continue
                anko_count += 1

    # 2. 안깡 추가 (반드시 커쯔여야 함)
    anko_count += len(ankan_sets)

    # 3. 총 4개 커쯔 + 멘젠 조건
    return anko_count >= 4 and not ming_sets  # 멘젠 조건 추가

def is_suuankou_tanki(head, bodies, winning_tile, ming_sets, ankan_sets):
    """스안커 단기 판정 (머리 검사 추가)"""
    ming_tiles_list = ming_sets if isinstance(ming_sets, list) else []
    ankan_list = ankan_sets if isinstance(ankan_sets, list) else []
    
    anko_count = 0
    has_tanki = False
    
    # 1. 안커 개수 세기 (몸통 검사)
    for body in bodies:
        if len(body) == 3 and body[0] == body[1] == body[2]:
            if body[0] not in ming_tiles_list:
                if winning_tile in body:
                    return False  # 화료패가 커쯔에 포함될 경우 제외
                anko_count += 1
    
    # 2. 안깡 개수 추가
    anko_count += len(ankan_list)
    
    # 3. 머리에서 단기 대기 확인 (핵심 수정 부분)
    if len(head) == 2 and head[0] == head[1] and winning_tile in head:
        has_tanki = True
    
    return anko_count >= 4 and has_tanki

def is_daisangen(bodies, ankan_sets, ming_sets):
    """대삼원 판정 (개정판)"""
    dragons = {'5z':0, '6z':0, '7z':0}
    
    # 1. 일반 몸통에서 카운트
    for body in bodies:
        tile = body[0]
        if tile in dragons:
            dragons[tile] += len(body)
    
    # 2. 안깡 추가
    for ankan in ankan_sets:
        tile = ankan[0]
        if tile in dragons:
            dragons[tile] +=4
    
    # 3. 밍깡 추가
    for ming_type, ming_set in ming_sets:
        tile = ming_set[0]
        if tile in dragons:
            dragons[tile] += len(ming_set)
    
    return all(count >=3 for count in dragons.values())

def is_tsuuiisou(tiles, ankan_sets, ming_sets):
    """자일색 판정 (밍깡/안깡 포함)"""
    all_tiles = tiles.copy()
    
    # 안깡 추가
    for ankan in ankan_sets:
        all_tiles.extend(ankan)
    
    # 밍깡/밍커 추가
    for ming_type, ming_set in ming_sets:
        all_tiles.extend(ming_set)
    
    return all(t.endswith('z') for t in all_tiles)

def is_ryuuiisou(tiles):
    """녹일색 판정"""
    # 녹일색 가능 패: 2s, 3s, 4s, 6s, 8s, 6z(발)
    green_tiles = {'2s', '3s', '4s', '6s', '8s', '6z'}
    return all(tile in green_tiles for tile in tiles)

def is_chinroutou(head, bodies, ankan_sets, ming_sets, tiles):
    """청노두 판정 (전체 패 검증)"""
    # 모든 패 수집
    all_tiles = tiles.copy()
    
    # 1. 머리 추가
    all_tiles.extend(head)
    
    # 2. 몸통 추가
    for body in bodies:
        all_tiles.extend(body)
    
    # 3. 안깡 추가
    for ankan in ankan_sets:
        all_tiles.extend(ankan)
    
    # 4. 밍깡/밍커 추가
    for ming_type, ming_tiles in ming_sets:
        all_tiles.extend(ming_tiles)
    
    # 5. 자패 존재 여부 확인
    if any(t.endswith('z') for t in all_tiles):
        return False
    
    # 6. 모든 수패가 1/9인지 확인
    return all(t[0] in '19' for t in all_tiles if not t.endswith('z'))

def is_shousuushii(bodies, ankan_sets, ming_sets):
    """소사희 판정 (밍깡/안깡 포함)"""
    winds = {'1z':0, '2z':0, '3z':0, '4z':0}
    
    # 1. 일반 몸통
    for body in bodies:
        tile = body[0]
        if tile in winds:
            winds[tile] += len(body)
    
    # 2. 안깡
    for ankan in ankan_sets:
        tile = ankan[0]
        if tile in winds:
            winds[tile] +=4
    
    # 3. 밍깡/밍커
    for ming_type, ming_set in ming_sets:
        tile = ming_set[0]
        if tile in winds:
            winds[tile] += len(ming_set)
    
    return sum(1 for v in winds.values() if v >=3) ==3

def is_daisuushii(bodies, ankan_sets, ming_sets):
    """대사희 판정 (밍깡/안깡 포함)"""
    winds = {'1z':0, '2z':0, '3z':0, '4z':0}
    
    # 1. 일반 몸통
    for body in bodies:
        tile = body[0]
        if tile in winds:
            winds[tile] += len(body)
    
    # 2. 안깡
    for ankan in ankan_sets:
        tile = ankan[0]
        if tile in winds:
            winds[tile] +=4
    
    # 3. 밍깡/밍커
    for ming_type, ming_set in ming_sets:
        tile = ming_set[0]
        if tile in winds:
            winds[tile] += len(ming_set)
    
    return all(v >=3 for v in winds.values())

def is_sukantsu(bodies, ankan_sets=None):
    kant_count = 0
    
    # 일반 깡 확인
    for body in bodies:
        if len(body) == 4:
            kant_count += 1
            
    # 안깡 세트 확인
    if ankan_sets:
        kant_count += len(ankan_sets)
    
    return kant_count == 4

def is_chuuren_poutou(head, tiles):
    """구련보등 판정"""
    # 청일색 확인
    if not is_chinitsu(head, tiles, [], []):
        return False
    
    # 패 종류 확인
    suit = next(tile[-1] for tile in tiles if not tile.endswith('z'))
    required_counts = {'1': 3, '2': 1, '3': 1, '4': 1, '5': 1, '6': 1, '7': 1, '8': 1, '9': 3}
    
    # 실제 패 개수 세기
    tile_counts = defaultdict(int)
    for tile in tiles:
        number = tile[0]
        tile_counts[number] += 1
    
    # 필요한 패가 모두 있는지 확인
    for number, required_count in required_counts.items():
        if tile_counts[number] < required_count:
            return False
    
    return True

def is_chuuren_poutou_9wait(head, tiles, winning_tile):
    """순정구련보등 판정 (9면 대기 검증 추가)"""
    # 1. 청일색 확인
    if not is_chinitsu(head, tiles, [], []):
        return False

    suit = tiles[0][-1]  # 패 종류 추출 (m/p/s)
    required_structure = {
        '1': 3, '2': 1, '3': 1, '4': 1, 
        '5': 1, '6': 1, '7': 1, '8': 1, '9': 3
    }

    # 2. 실제 패 개수 세기 (적도라 변환 후)
    counts = defaultdict(int)
    for tile in tiles:
        num = tile[0] if not tile.startswith('r') else tile[1]
        counts[num] += 1

    # 3. 구조 검증 (4장짜리 패 존재 시 실패)
    for num, req_count in required_structure.items():
        if counts.get(num, 0) != req_count + (1 if num == winning_tile[0] else 0):
            return False

    # 4. 9면 대기 확인
    winning_num = winning_tile[0]
    return all(
        counts.get(str(i), 0) >= (1 if i != int(winning_num) else 2)
        for i in range(1, 10)
    )

def is_peiko(bodies):
    """이페코 판정"""
    # 슌쯔만 추출
    shuntsu = [body for body in bodies if len(body) == 3 and body[0] != body[1]]
    
    # 같은 슌쯔가 있는지 확인
    for i in range(len(shuntsu)):
        for j in range(i + 1, len(shuntsu)):
            if shuntsu[i] == shuntsu[j]:
                return True
    return False
