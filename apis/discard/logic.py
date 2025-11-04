import cv2
import json
import tkinter as tk
from tkinter import filedialog
from collections import Counter
from ultralytics import YOLO

def detect_mahjong_tiles_in_image(model, image_path, exclude_classes=[]):
    image = cv2.imread(image_path)
    detected_tiles = []

    # 모델로 이미지에서 마작 패 감지
    results = model(image)  
    frame_tiles = []
    
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
    
    # 유효한 감지 결과만 표시 및 저장
    for x1, y1, x2, y2, conf, cls, label in valid_detections:
        frame_tiles.append(label)

    detected_tiles.append(frame_tiles)
    
    save_detected_tiles(detected_tiles)

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

def save_detected_tiles(detected_tiles, filename="detected_tiles.json"):
    with open(filename, 'w') as f:
        json.dump(detected_tiles, f, indent=4)

def load_detected_tiles(filename="detected_tiles.json"):
    with open(filename, 'r') as f:
        return json.load(f)

def select_image():
    root = tk.Tk()
    root.withdraw()

    image_path = filedialog.askopenfilename(
        initialdir="c", 
        title="Select Image File",
        filetypes=(("Image files", "*.jpg;*.png"), ("All files", "*.*"))
    )
    return image_path

def recommend_discard(filepath):
    detected_tiles = load_detected_tiles(filepath)
    
    if not detected_tiles:
        return {"error": "타일을 감지할 수 없습니다"}
    
    hand = detected_tiles[0]
    
    # 5블록 이론 기반 버림패 추천
    best_tile, analysis = find_best_discard(hand)
    
    # explain_recommendation 호출 제거
    
    return {
        "original_hand": hand,
        "recommended_discard": best_tile,
        "block_analysis": analysis
    }

def process_tile(tile: str) -> str:
    """적도라(r5m 등)를 일반 패로 변환 (예: 'r5m' → '5m')"""
    return tile[1:] if tile.startswith('r') else tile

def check_chitoitsu(hand):
    """치또이츠 가능성 체크"""
    pairs = []
    tile_counts = Counter(hand)
    
    # 쌍패 개수 확인
    for tile, count in tile_counts.items():
        if count >= 2:
            pairs.append(tile)
    
    # 4쌍 이상이면 치또이츠 가능성 높음
    if len(pairs) >= 4:
        return True, pairs
    return False, []

def check_kokushi(hand):
    """국사무쌍 가능성 체크"""
    terminals_and_honors = ['1m', '9m', '1p', '9p', '1s', '9s', 
                           '1z', '2z', '3z', '4z', '5z', '6z', '7z']
    
    unique_yaochu = set(tile for tile in hand 
                       if any(tile.startswith(t) for t in terminals_and_honors))
    
    # 8종류 이상의 야오츄패가 있으면 국사무쌍 가능성 높음
    if len(unique_yaochu) >= 8:
        return True, unique_yaochu
    return False, []

def find_best_tile_to_discard(hand):
    best_tile = None
    best_score = float('-inf')
    
    # 각 패에 대한 평가
    for tile in set(hand):
        simulated_hand = hand.copy()
        simulated_hand.remove(tile)
        
        # 기본 점수 계산
        score = evaluate_tile_safety(tile, hand)
        
        # 텐파이 가능성 평가
        waiting_tiles = find_waiting_tiles(simulated_hand)
        if waiting_tiles:
            score += len(waiting_tiles) * 20
            
        if score > best_score:
            best_score = score
            best_tile = tile
            
    return best_tile

def is_tenpai(hand):
    # 치또이 텐파이 먼저 체크
    if is_chiitoi_tenpai(hand):
        return True
    
    # 일반적인 텐파이 체크
    waiting_tiles = find_waiting_tiles(hand)
    return len(waiting_tiles) > 0

def find_waiting_tiles(hand):
    waiting_tiles = set()
    
    # 치또이 텐파이 체크
    if is_chiitoi_tenpai(hand):
        tile_count = Counter(hand)
        for tile, count in tile_count.items():
            if count == 1:  # 단패인 경우, 이 패가 오면 치또이 완성
                waiting_tiles.add(tile)
                return waiting_tiles
            
    all_possible_tiles = []
    
    # 모든 가능한 패 생성
    suits = ['m', 'p', 's']
    for suit in suits:
        for i in range(1, 10):
            all_possible_tiles.append(f'{i}{suit}')
    for i in range(1, 8):
        all_possible_tiles.append(f'{i}z')
    
    # 각 가능한 패에 대해 테스트
    for test_tile in all_possible_tiles:
        test_hand = hand.copy()
        test_hand.append(test_tile)
        test_hand.sort()
        
        # 재귀적으로 모든 가능한 조합을 시도
        def try_combinations(tiles, used_tiles=None, depth=0):
            if used_tiles is None:
                used_tiles = []
            
            if depth >= 4:  # 4개의 면자/코츠를 찾았다면
                if len(tiles) == 0:  # 모든 패를 사용했다면
                    waiting_tiles.add(test_tile)
                return
            
            tiles_copy = tiles.copy()
            
            # 코츠(triplet) 시도
            for tile in set(tiles_copy):
                if tiles_copy.count(tile) >= 3:
                    new_tiles = tiles_copy.copy()
                    for _ in range(3):
                        new_tiles.remove(tile)
                    try_combinations(new_tiles, used_tiles + [[tile]*3], depth+1)
            
            # 슌쯔(sequence) 시도
            for tile in set(tiles_copy):
                if not tile.endswith('z'):  # 자패가 아닌 경우만
                    tile_val = int(tile[0])
                    tile_suit = tile[1]
                    if (tile_val <= 7 and 
                        f'{tile_val+1}{tile_suit}' in tiles_copy and 
                        f'{tile_val+2}{tile_suit}' in tiles_copy):
                        new_tiles = tiles_copy.copy()
                        new_tiles.remove(tile)
                        new_tiles.remove(f'{tile_val+1}{tile_suit}')
                        new_tiles.remove(f'{tile_val+2}{tile_suit}')
                        try_combinations(new_tiles, used_tiles + [[tile, f'{tile_val+1}{tile_suit}', f'{tile_val+2}{tile_suit}']], depth+1)
        
        # 머리(pair) 찾기
        for head_tile in set(test_hand):
            if test_hand.count(head_tile) >= 2:
                remaining_tiles = test_hand.copy()
                remaining_tiles.remove(head_tile)
                remaining_tiles.remove(head_tile)
                try_combinations(remaining_tiles)
    
    return waiting_tiles

def is_chiitoi_tenpai(hand):
    tile_count = Counter(hand)
    pairs = 0
    singles = 0
    
    for tile, count in tile_count.items():
        if count >= 2:
            pairs += 1
        elif count == 1:
            singles += 1
    
    # 치또이 텐파이 조건: 
    # 1. 6쌍이 있고 단패가 1개인 경우
    return pairs == 6 and singles == 1

def evaluate_tile_safety(tile, hand):
    score = 0
    tile_counts = Counter(hand)
    
    # 4개 모인 패 처리
    if tile_counts[tile] == 4:
        # 다른 몸통 구성 가능성 확인
        other_sets = check_other_possible_sets(hand, tile)
        if not other_sets:
            score += 15  # 다른 몸통을 만들기 어려우면 하나 버리도록 높은 점수 부여

    # 또이쯔(대자) 평가
    if tile_counts[tile] >= 2 and tile_counts[tile] < 4:
        score -= 10  # 대자는 보호
        
    if tile[-1] in ['m', 'p', 's']:
        value = int(tile[0])
        suit = tile[-1]
        connections = 0
        for i in [-2, -1, 1, 2]:
            next_value = value + i
            if 1 <= next_value <= 9:
                if f'{next_value}{suit}' in hand:
                    connections += 1
        if connections == 0:
            score += 8
    else:
        if tile_counts[tile] == 1:
            score += 10
            
    return score

def check_other_possible_sets(hand, quad_tile):
    # 깡패를 제외한 나머지 패들로 몸통 구성이 가능한지 확인
    remaining_tiles = [t for t in hand if t != quad_tile]
    possible_sets = 0
    
    # 간단한 몸통 체크 (실제로는 더 정교한 로직 필요)
    tile_counts = Counter(remaining_tiles)
    
    for tile, count in tile_counts.items():
        # 커쯔 체크
        if count >= 3:
            possible_sets += 1
            
        # 슌쯔 체크 (숫패만)
        if not tile.endswith('z'):
            value = int(tile[0])
            suit = tile[-1]
            if value <= 7:
                if (f'{value+1}{suit}' in remaining_tiles and 
                    f'{value+2}{suit}' in remaining_tiles):
                    possible_sets += 1
    
    return possible_sets >= 1  # 최소 하나 이상의 몸통이 가능하면 True

def analyze_hand(hand):
    """패를 블록 단위로 분석"""
    # 적도라 처리된 패로 변환
    processed_hand = [process_tile(tile) for tile in hand]
    tile_counts = Counter(processed_hand)
    effective_tiles = set()
    
    # 각 패 종류별로 분석
    for tile in set(processed_hand):
        if tile[-1] in ['m', 'p', 's']:
            value = int(tile[0])
            suit = tile[-1]
            
            # 연속된 패 분석 최적화
            for i in range(max(1, value-2), min(9, value+3)):
                if i <= 7:
                    seq = [f"{j}{suit}" for j in range(i, i+3)]
                    existing = sum(1 for t in seq if t in processed_hand)
                    missing = set(seq) - set(processed_hand)
                    if existing >= 2 and len(missing) == 1:
                        effective_tiles.update(missing)

            # 쌍패 및 커쯔 분석
            if tile_counts[tile] == 2:
                effective_tiles.add(tile)
            elif tile_counts[tile] == 1:
                for i in [-2, -1, 1, 2]:
                    new_value = value + i
                    if 1 <= new_value <= 9:
                        effective_tiles.add(f"{new_value}{suit}")

    # 자패 처리
    for tile in set(processed_hand):
        if tile[-1] == 'z':
            if tile_counts[tile] == 2:
                effective_tiles.add(tile)
            elif tile_counts[tile] == 1:
                effective_tiles.add(tile)

    return effective_tiles

def calculate_connection_score(tile1, tile2):
    """두 패 사이의 연결 점수 계산"""
    processed_tile1 = process_tile(tile1)
    processed_tile2 = process_tile(tile2)
    
    if processed_tile1[-1] != processed_tile2[-1]:  # 다른 종류의 패
        return 0
        
    if processed_tile1[-1] == 'z':  # 자패
        return 0
        
    value1 = int(processed_tile1[0])
    value2 = int(processed_tile2[0])
    
    diff = abs(value1 - value2)
    if diff == 0:
        return 100  # 쌍패
    elif diff == 1:
        return 80  # 연속된 패
    elif diff == 2:
        return 60  # 간짱 가능성
    else:
        return 0
    
def analyze_blocks(hand):
    """패 분석 함수 (5블록 이론 기반)"""
    if not (13 <= len(hand) <= 14):
        raise ValueError("패의 개수가 올바르지 않습니다")
    
    blocks = []
    used_tiles = set()
    
    # 1. 멘쯔 찾기 (최우선)
    mentsu = find_mentsu(hand, used_tiles)
    blocks.extend(mentsu)
    
    # 2. 타쯔 찾기 (양면 > 변짱 > 간짱 순)
    # 양면 타쯔 먼저 찾기
    ryanmen_tatsu = find_tatsu(hand, used_tiles, tatsu_type='ryanmen')
    blocks.extend(ryanmen_tatsu)
    
    # 그 다음 다른 타쯔 찾기
    other_tatsu = find_tatsu(hand, used_tiles, tatsu_type='other')
    blocks.extend(other_tatsu)
    
    # 3. 남은 패는 고립패로 처리
    isolated = [[t] for t in hand if t not in used_tiles]
    blocks.extend(isolated)
    
    # 5블록 이론에 맞게 최적의 블록 조합 선택
    optimized_blocks = optimize_blocks(blocks)
    
    return optimized_blocks, len(optimized_blocks)

def optimize_blocks(blocks):
    """5블록 이론에 맞게 블록 최적화"""
    # 멘쯔는 무조건 포함
    mentsu = [b for b in blocks if len(b) == 3]
    tatsu = [b for b in blocks if len(b) == 2]
    isolated = [b for b in blocks if len(b) == 1]
    
    # 최대 5-6개 블록으로 제한
    optimized = []
    optimized.extend(mentsu)  # 멘쯔 우선
    
    # 남은 블록 수 계산
    remaining_blocks = 5 - len(optimized)
    
    # 타쯔 추가
    optimized.extend(tatsu[:remaining_blocks])
    remaining_blocks = 5 - len(optimized)
    
    # 고립패 추가
    optimized.extend(isolated[:remaining_blocks])
    
    return optimized

def find_mentsu(hand, used_tiles):
    """완성된 멘쯔(세 패) 찾기"""
    mentsu = []
    temp_hand = [t for t in hand if t not in used_tiles]
    
    # 커쯔(같은 패 3개) 찾기
    tile_counts = Counter(temp_hand)
    for tile, count in tile_counts.items():
        if count >= 3:
            # 적도라 처리
            base_tile = process_tile(tile)
            same_tiles = [t for t in temp_hand if process_tile(t) == base_tile][:3]
            mentsu.append(same_tiles)
            used_tiles.update(same_tiles)
    
    # 슌쯔(연속된 3개) 찾기
    temp_hand = [t for t in hand if t not in used_tiles]
    for tile in sorted(temp_hand):
        if tile in used_tiles:
            continue
            
        base_tile = process_tile(tile)
        if base_tile[-1] in ['m', 'p', 's']:
            value = int(base_tile[0])
            suit = base_tile[-1]
            if value <= 7:
                next_tile = f"{value+1}{suit}"
                next_next_tile = f"{value+2}{suit}"
                
                # 연속된 패 찾기
                found_tiles = []
                found_tiles.append(tile)
                
                # 다음 패 찾기
                next_candidates = [t for t in temp_hand if process_tile(t) == next_tile 
                                 and t not in used_tiles]
                if next_candidates:
                    found_tiles.append(next_candidates[0])
                    
                    # 그 다음 패 찾기
                    next_next_candidates = [t for t in temp_hand if process_tile(t) == next_next_tile 
                                         and t not in used_tiles]
                    if next_next_candidates:
                        found_tiles.append(next_next_candidates[0])
                        
                if len(found_tiles) == 3:
                    mentsu.append(found_tiles)
                    used_tiles.update(found_tiles)
    
    return mentsu

def find_tatsu(hand, used_tiles, tatsu_type='all'):
    """미완성 타쯔(두 패) 찾기"""
    tatsu = []
    temp_hand = [t for t in hand if t not in used_tiles]
    
    # 1. 쌍패 먼저 찾기 (머리 확보)
    if tatsu_type in ['all', 'other']:
        tile_counts = Counter(temp_hand)
        for tile, count in tile_counts.items():
            if count == 2:
                base_tile = process_tile(tile)
                same_tiles = [t for t in temp_hand if process_tile(t) == base_tile][:2]
                if not any(t in used_tiles for t in same_tiles):
                    tatsu.append(same_tiles)
                    used_tiles.update(same_tiles)
    
    # 2. 양면 찾기
    if tatsu_type in ['all', 'ryanmen']:
        temp_hand = [t for t in hand if t not in used_tiles]
        for tile in sorted(temp_hand):
            if tile in used_tiles:
                continue
            
            base_tile = process_tile(tile)
            if base_tile[-1] in ['m', 'p', 's']:
                value = int(base_tile[0])
                suit = base_tile[-1]
                
                if 2 <= value <= 8:
                    next_tile = f"{value+1}{suit}"
                    next_candidates = [t for t in temp_hand if process_tile(t) == next_tile 
                                     and t not in used_tiles]
                    
                    if next_candidates:
                        found_tiles = [tile, next_candidates[0]]
                        tatsu.append(found_tiles)
                        used_tiles.update(found_tiles)
    
    # 3. 간짱 찾기 추가
    if tatsu_type in ['all', 'other']:
        temp_hand = [t for t in hand if t not in used_tiles]
        for tile in sorted(temp_hand):
            if tile in used_tiles:
                continue
            
            base_tile = process_tile(tile)
            if base_tile[-1] in ['m', 'p', 's']:
                value = int(base_tile[0])
                suit = base_tile[-1]
                
                if 1 <= value <= 7:
                    kanchan_tile = f"{value+2}{suit}"
                    kanchan_candidates = [t for t in temp_hand if process_tile(t) == kanchan_tile 
                                        and t not in used_tiles]
                    
                    if kanchan_candidates:
                        found_tiles = [tile, kanchan_candidates[0]]
                        tatsu.append(found_tiles)
                        used_tiles.update(found_tiles)
    
    return tatsu

def find_isolated_tiles(hand):
    """고립된 패 찾기"""
    return list(hand)

def evaluate_block_strength(blocks):
    """블록 강도 평가 (초보자용 단순화)"""
    total_score = 0
    
    for block in blocks:
        # 멘쯔 (이미 완성)
        if len(block) == 3:
            total_score += 2
        
        # 타쯔 (한 패만 더 필요)
        elif len(block) == 2:
            tile1, tile2 = block
            processed_tile1 = process_tile(tile1)
            processed_tile2 = process_tile(tile2)
            
            # 양면/쌍패가 변짱보다 유리
            if processed_tile1 == processed_tile2:  # 쌍패
                total_score += 1
            elif (processed_tile1[-1] == processed_tile2[-1] and 
                  abs(int(processed_tile1[0]) - int(processed_tile2[0])) == 1):
                total_score += 1  # 양면
            else:
                total_score += 0.5  # 변짱
        
        # 고립패 평가
        else:
            tile = process_tile(block[0])
            if tile[-1] == 'z':  # 자패는 가장 먼저 버림
                total_score -= 2
            elif tile[0] in ['1', '9']:  # 노두패는 그 다음으로 버림
                total_score -= 1.5
            elif tile[0] in ['2', '8']:  # 2,8패는 그 다음으로 버림
                total_score -= 1
            else:  # 중간패(3-7)는 가장 나중에 버림
                total_score -= 0.5
    
    return total_score

def find_best_discard(hand):
    """
    샤텐 수 기반 버림패 추천 함수
    """
    if len(hand) != 14:
        warning = f"패 개수가 {len(hand)}개입니다. 정확한 분석을 위해 14개가 권장됩니다."
        return hand[0] if hand else None, {"warning": warning}

    try:
        # 현재 패의 샤텐 수 계산
        current_shanten = calculate_shanten(hand)
        
        # 각 패를 버렸을 때의 샤텐 수와 유효 패 계산
        discard_analysis = {}
        best_tile = None
        best_shanten = float('inf')
        best_effective_tiles = 0
        
        for i, tile in enumerate(hand):
            temp_hand = hand.copy()
            temp_hand.pop(i)
            
            # 버린 후 샤텐 수 계산
            shanten_after = calculate_shanten(temp_hand)
            
            # 유효 패(샤텐 수를 줄이는 패) 계산
            effective_tiles = calculate_effective_tiles(temp_hand, shanten_after)
            
            discard_analysis[tile] = {
                'shanten_before': current_shanten,
                'shanten_after': shanten_after,
                'effective_tiles': effective_tiles,
                'effective_count': len(effective_tiles)
            }
            
            # 최적의 버림패 선택 (샤텐 수가 가장 낮고, 유효 패가 가장 많은 것)
            if (shanten_after < best_shanten or 
                (shanten_after == best_shanten and len(effective_tiles) > best_effective_tiles)):
                best_shanten = shanten_after
                best_effective_tiles = len(effective_tiles)
                best_tile = tile
        
        # 5블록 분석도 추가
        for tile in hand:
            temp_hand = hand.copy()
            temp_hand.remove(tile)
            
            blocks, block_count = analyze_blocks(temp_hand)
            discard_analysis[tile]['blocks'] = blocks
            discard_analysis[tile]['block_count'] = block_count
            
        return best_tile, discard_analysis
        
    except Exception as e:
        print(f"버림패 분석 중 오류 발생: {e}")
        return hand[0], {'error': str(e)}

def calculate_effective_tiles(hand, current_shanten):
    """
    유효 패(샤텐 수를 줄이는 패) 계산
    """
    effective_tiles = set()
    
    # 모든 가능한 패 생성
    all_tiles = []
    for suit in ['m', 'p', 's']:
        for i in range(1, 10):
            all_tiles.append(f"{i}{suit}")
    for i in range(1, 8):
        all_tiles.append(f"{i}z")
    
    # 각 패를 추가했을 때 샤텐 수가 줄어드는지 확인
    for tile in all_tiles:
        test_hand = hand.copy()
        test_hand.append(tile)
        
        # 패 개수가 14개를 초과하면 건너뛰기
        if len(test_hand) > 14:
            test_hand.pop()
        
        shanten_after = calculate_shanten(test_hand)
        
        if shanten_after < current_shanten:
            effective_tiles.add(tile)
    
    return effective_tiles

def explain_recommendation(hand, best_tile, analysis):
    """버림패 추천 이유 설명"""
    try:
        print("\n== 버림패 추천 분석 ==")
        print(f"추천 버림패: {best_tile}")
        
        if 'error' in analysis:
            print(f"분석 중 오류 발생: {analysis['error']}")
            return
            
        if best_tile in analysis:
            tile_analysis = analysis[best_tile]
            
            # 특수형 확인
            if 'special_form' in tile_analysis:
                if tile_analysis['special_form'] == 'chitoitsu':
                    print("\n추천 이유:")
                    print("- 현재 패가 치또이츠(7쌍)에 가깝습니다.")
                    print("- 쌍패가 아닌 패를 우선적으로 버려서 치또이츠를 완성할 수 있습니다.")
                    return
                elif tile_analysis['special_form'] == 'kokushi':
                    print("\n추천 이유:")
                    print("- 현재 패가 국사무쌍에 가깝습니다.")
                    print("- 1, 9, 자패가 아닌 중간 패를 버려서 국사무쌍을 완성할 수 있습니다.")
                    return
            
            if 'blocks' in tile_analysis:
                print("\n패 구성:")
                for i, block in enumerate(tile_analysis['blocks'], 1):
                    if len(block) == 3:
                        print(f"그룹 {i}: 완성된 세 패")
                        for tile in block:
                            print(f"  - {tile}")
                    elif len(block) == 2:
                        print(f"그룹 {i}: 연결 가능한 두 패")
                        for tile in block:
                            print(f"  - {tile}")
                    else:
                        print(f"그룹 {i}: 단독 패")
                        for tile in block:
                            print(f"  - {tile}")
                
                # 추천 이유 상세 설명
                print("\n추천 이유:")
                if best_tile[-1] == 'z':
                    if best_tile[0] in ['1', '2', '3', '4']:
                        print("- 동, 남, 서, 북 패는 다른 패와 연결할 수 없어서 먼저 버리는 것이 좋습니다.")
                    else:
                        print("- 백, 발, 중 패는 다른 패와 연결되지 않아 버리는 것이 좋습니다.")
                elif best_tile[0] in ['1', '9']:
                    print("- 1과 9는 한쪽으로만 연결할 수 있어서 버리기 좋은 패입니다.")
                elif any(len(block) == 3 for block in tile_analysis['blocks']):
                    print("- 이미 완성된 세 패 조합이 있어 남은 패들을 발전시킬 수 있습니다.")
                elif any(len(block) == 2 for block in tile_analysis['blocks']):
                    print("- 연결 가능한 두 패 조합을 유지하면서 더 좋은 패를 기다릴 수 있습니다.")
                else:
                    print("- 단독으로 있는 패를 버려서 더 좋은 패 조합을 만들 수 있습니다.")
                
    except Exception as e:
        print(f"설명 생성 중 오류 발생: {e}")

def calculate_shanten(hand):
    """개선된 샤텐 수 계산 함수"""
    processed_hand = [process_tile(t) for t in hand]
    
    # 일반형 샤텐 계산 (우선순위 1)
    normal_shanten = calculate_normal_shanten(processed_hand)
    
    # 초보자 가이드: 특수형은 4패 이하일 때만 고려
    if len(hand) >= 10:
        return normal_shanten
    
    # 치또이츠 샤텐 (우선순위 2)
    chiitoi_shanten = calculate_chitoitsu_shanten(processed_hand)
    
    # 국사무쌍 샤텐 (우선순위 3)
    kokushi_shanten = calculate_kokushi_shanten(processed_hand)
    
    return min(normal_shanten, chiitoi_shanten, kokushi_shanten)

def calculate_normal_shanten(hand):
    """정확한 일반형 샤텐 계산 함수"""
    if len(hand) < 13:
        return 6  # 최대 샤텐 수 반환

    min_shanten = 8
    has_pair = False

    # 모든 가능한 머리 후보 탐색
    for i in range(len(hand)):
        if i > 0 and hand[i] == hand[i-1]:
            continue  # 중복 계산 방지
            
        # 머리 선정 시도
        if i < len(hand)-1 and hand[i] == hand[i+1]:
            pair = hand[i:i+2]
            remaining = hand[:i] + hand[i+2:]
            
            # 완성된 멘쯔와 타쯔 계산
            mentsu, tatsu = count_mentsu_and_tatsu(remaining)
            
            current_shanten = 8 - (mentsu * 2) - tatsu - 1
            min_shanten = min(min_shanten, current_shanten)
            has_pair = True

    # 머리 없는 경우 계산
    mentsu, tatsu = count_mentsu_and_tatsu(hand)
    current_shanten = 8 - (mentsu * 2) - tatsu
    min_shanten = min(min_shanten, current_shanten)

    return max(min_shanten, 0) if has_pair else min_shanten

def count_mentsu_and_tatsu(tiles):
    """멘쯔와 타쯔 개수 계산"""
    tiles = sorted(tiles, key=lambda x: (x[-1], int(x[0])))
    mentsu = 0
    tatsu = 0
    i = 0
    
    while i < len(tiles):
        # 커쯔 확인
        if i <= len(tiles)-3 and tiles[i] == tiles[i+1] == tiles[i+2]:
            mentsu += 1
            i += 3
            continue
            
        # 슌쯔 확인
        if i <= len(tiles)-3:
            current = tiles[i]
            next1 = tiles[i+1]
            next2 = tiles[i+2]
            
            if (current[0] == str(int(next1[0])-1) == str(int(next2[0])-2) and
                current[-1] == next1[-1] == next2[-1]):
                mentsu += 1
                i += 3
                continue
                
        # 타쯔 확인 (쌍패)
        if i <= len(tiles)-2 and tiles[i] == tiles[i+1]:
            tatsu += 1
            i += 2
            continue
            
        # 타쯔 확인 (연속)
        if i <= len(tiles)-2 and tiles[i][-1] == tiles[i+1][-1]:
            if int(tiles[i][0])+1 == int(tiles[i+1][0]):
                tatsu += 1
                i += 2
                continue
                
        i += 1
    
    return mentsu, tatsu

def calculate_chitoitsu_shanten(hand):
    """정확한 치또이츠 샤텐 계산"""
    pair_count = 0
    unique_tiles = set()
    
    count = Counter(hand)
    for tile, cnt in count.items():
        if cnt >= 2:
            pair_count += 1
        if cnt >= 1:
            unique_tiles.add(tile)
    
    shanten = 6 - pair_count
    needed_unique = max(7 - len(unique_tiles), 0)
    return shanten + needed_unique

def calculate_kokushi_shanten(hand):
    """정확한 국사무쌍 샤텐 계산"""
    required = {'1m','9m','1p','9p','1s','9s','1z','2z','3z','4z','5z','6z','7z'}
    unique = set()
    has_pair = False
    
    for tile in hand:
        if tile in required:
            unique.add(tile)
            if hand.count(tile) >= 2:
                has_pair = True
                
    missing = 13 - len(unique)
    return missing - (1 if has_pair else 0)

def find_sets_and_pairs(tiles):
    processed_tiles = [process_tile(t) for t in tiles]
    norm_to_orig = {process_tile(t): t for t in tiles} 
    """
    패에서 완성된 몸통(멘쯔), 미완성 몸통(타쯔), 고립패 찾기
    """
    mentsu = []  # 완성된 몸통
    tatsu = []   # 미완성 몸통
    isolated = []  # 고립패
    used_tiles = set()
    
    # 1. 정규화된 타일과 원본 매핑 생성
    norm_to_orig = {}  # {정규화타일: [원본타일1, 원본타일2, ...]}
    for t in tiles:
        norm = process_tile(t)
        if norm not in norm_to_orig:
            norm_to_orig[norm] = []
        norm_to_orig[norm].append(t)
    
    # 2. 커쯔(같은 패 3개) 찾기 (정규화 기준)
    norm_counts = Counter([process_tile(t) for t in tiles])
    for norm_tile, count in norm_counts.items():
        if count >= 3:
            # 원본 타일 3개 선택 (중복 고려)
            orig_tiles = norm_to_orig[norm_tile][:3]
            mentsu.append(orig_tiles)
            used_tiles.update(orig_tiles)
            # 남은 타일 처리
            norm_to_orig[norm_tile] = norm_to_orig[norm_tile][3:]
    
    # 3. 슌쯔(연속된 3개) 찾기 (정규화 기준)
    sorted_norms = sorted(
        [process_tile(t) for t in tiles if t not in used_tiles],
        key=lambda x: (x[-1], int(x[0]) if x[0].isdigit() else 0)
    )
    
    i = 0
    while i < len(sorted_norms) - 2:
        current = sorted_norms[i]
        next_norm = sorted_norms[i+1]
        next_next_norm = sorted_norms[i+2]
        
        if (current[-1] == next_norm[-1] == next_next_norm[-1] and
            int(current[0])+1 == int(next_norm[0]) and
            int(current[0])+2 == int(next_next_norm[0])):
            
            # 원본 타일 매핑에서 실제 타일 추출
            current_orig = norm_to_orig[current].pop(0)
            next_orig = norm_to_orig[next_norm].pop(0)
            next_next_orig = norm_to_orig[next_next_norm].pop(0)
            
            mentsu.append([current_orig, next_orig, next_next_orig])
            used_tiles.update([current_orig, next_orig, next_next_orig])
            i += 3
        else:
            i += 1
    
    # 4. 타쯔(미완성 몸통) 찾기
    # 4.1 쌍패(토이츠) - 정규화 기준
    for norm_tile, orig_list in norm_to_orig.items():
        if len(orig_list) >= 2 and norm_tile[-1] != 'z':
            pair = orig_list[:2]
            tatsu.append(pair)
            used_tiles.update(pair)
            norm_to_orig[norm_tile] = orig_list[2:]
    
    # 4.2 양면/간짱 타쯔 - 정규화 기준
    remaining_norms = [n for n in sorted_norms if process_tile(n) not in used_tiles]
    for i in range(len(remaining_norms)):
        current = remaining_norms[i]
        if current[-1] == 'z':
            continue
            
        # 양면 타쯔
        if i+1 < len(remaining_norms):
            next_norm = remaining_norms[i+1]
            if (current[-1] == next_norm[-1] and 
                int(current[0])+1 == int(next_norm[0])):
                tatsu.append([current, next_norm])
                
        # 간짱 타쯔
        if i+2 < len(remaining_norms):
            next_next_norm = remaining_norms[i+2]
            if (current[-1] == next_next_norm[-1] and 
                int(current[0])+2 == int(next_next_norm[0])):
                tatsu.append([current, next_next_norm])
    
    # 5. 고립패 계산
    isolated = [t for t in tiles if t not in used_tiles]
    
    return mentsu, tatsu, isolated

