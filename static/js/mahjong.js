document.addEventListener('DOMContentLoaded', function() {
    // 마작패 이미지 로드
    loadTileImages();

    // 화료 패 선택 이벤트
    document.getElementById('winning-tile-selection').addEventListener('click', function(e) {
        if (e.target.classList.contains('tile-image')) {
            selectWinningTile(e.target);
        }
    });

    // 도라 표시패 선택 이벤트
    document.getElementById('dora-tile-selection').addEventListener('click', function(e) {
        if (e.target.classList.contains('tile-image')) {
            selectDoraTile(e.target);
        }
    });

    // 도라 표시패 길게 누르기 이벤트 설정
    const doraTiles = document.getElementById('dora-tile-selection').getElementsByClassName('tile-image');
    Array.from(doraTiles).forEach(img => {
        img.addEventListener('mousedown', startLongPress);
        img.addEventListener('touchstart', startLongPress, { passive: true });
        img.addEventListener('mouseup', cancelLongPress);
        img.addEventListener('mouseleave', cancelLongPress);
        img.addEventListener('touchend', cancelLongPress);
        img.addEventListener('touchcancel', cancelLongPress);
    });

    // 안깡 타일 선택 이벤트
    document.getElementById('ankan-tile-selection').addEventListener('click', function(e) {
        if (e.target.classList.contains('tile-image')) {
            selectAnkanTile(e.target);
        }
    });

    // 밍 타일 선택 이벤트 추가
    document.getElementById('ming-tile-selection').addEventListener('click', function(e) {
        if (e.target.classList.contains('tile-image')) {
            selectMingTile(e.target);
        }
    });
});

function loadTileImages() {
    const tiles = [
        // 만수패
        '1m', '2m', '3m', '4m', '5m', 'r5m', '6m', '7m', '8m', '9m',
        // 핀즈패
        '1p', '2p', '3p', '4p', '5p', 'r5p', '6p', '7p', '8p', '9p',
        // 소우즈패
        '1s', '2s', '3s', '4s', '5s', 'r5s', '6s', '7s', '8s', '9s',
        // 자패
        '1z', '2z', '3z', '4z', '5z', '6z', '7z'
    ];
    
    const winningTileGrid = document.getElementById('winning-tile-selection');
    const doraTileGrid = document.getElementById('dora-tile-selection');
    const ankanTileGrid = document.getElementById('ankan-tile-selection');
    const mingTileGrid = document.getElementById('ming-tile-selection');

    tiles.forEach(tile => {
        const img = document.createElement('img');
        img.src = `/static/images/tiles/${tile}.png`;
        img.classList.add('tile-image');
        img.dataset.tile = tile;

        winningTileGrid.appendChild(img.cloneNode(true));
        doraTileGrid.appendChild(img.cloneNode(true));
        ankanTileGrid.appendChild(img.cloneNode(true));
        mingTileGrid.appendChild(img.cloneNode(true));
    });
}

// 화료 패 선택 (단일 선택)
function selectWinningTile(img) {
    // 이전에 선택된 패의 선택 해제
    const prevSelected = document.querySelector('#winning-tile-selection .selected');
    if (prevSelected) {
        prevSelected.classList.remove('selected');
    }
    img.classList.add('selected');
    updateSelectedDisplay('selected-winning-tile', [img.dataset.tile]);
}

// 도라 표시패 선택 (같은 종류 최대 4장까지)
let selectedDoraTiles = [];
function selectDoraTile(img) {
    const tile = img.dataset.tile;
    
    // 같은 종류의 타일 개수 확인
    const sameTypeCount = selectedDoraTiles.filter(t => t === tile).length;
    
    if (sameTypeCount === 4) {
        // 4개일 때 클릭하면 해당 종류 모두 제거
        selectedDoraTiles = selectedDoraTiles.filter(t => t !== tile);
        // 선택 표시 제거
        document.querySelectorAll(`#dora-tile-selection img[data-tile="${tile}"]`)
            .forEach(img => img.classList.remove('selected'));
    } else {
        // 4개 미만이면 추가
        selectedDoraTiles.push(tile);
        img.classList.add('selected');
    }
    
    updateSelectedDisplay('selected-dora-tiles', selectedDoraTiles);
}


// 길게 누르기 시작
function startLongPress(e) {
    const img = e.target;
    longPressTimer = setTimeout(() => {
        const tile = img.dataset.tile;
        // 선택된 같은 종류의 타일 모두 제거
        selectedDoraTiles = selectedDoraTiles.filter(t => t !== tile);
        // 선택 표시 제거
        document.querySelectorAll(`#dora-tile-selection img[data-tile="${tile}"]`)
            .forEach(img => img.classList.remove('selected'));
        updateSelectedDisplay('selected-dora-tiles', selectedDoraTiles);
    }, LONG_PRESS_DURATION);
}

// 길게 누르기 취소
function cancelLongPress() {
    clearTimeout(longPressTimer);
}

// 안깡 타일 선택 (세트당 4개)
let selectedAnkanTiles = [];
function selectAnkanTile(img) {
    const tile = img.dataset.tile;
    const index = selectedAnkanTiles.indexOf(tile);
    
    if (index === -1) {
        if (selectedAnkanTiles.length < 4) {
            selectedAnkanTiles.push(tile);
            img.classList.add('selected');
        }
    } else {
        selectedAnkanTiles.splice(index, 1);
        img.classList.remove('selected');
    }
    updateSelectedDisplay('selected-ankan-tiles', selectedAnkanTiles);
}

// 밍 타일 선택 (여러 개 선택 가능)
let selectedMingTiles = [];
function selectMingTile(img) {
    const tile = img.dataset.tile;
    const index = selectedMingTiles.indexOf(tile);
    
    if (index === -1) {
        selectedMingTiles.push(tile);
        img.classList.add('selected');
    } else {
        selectedMingTiles.splice(index, 1);
        img.classList.remove('selected');
    }
    updateSelectedDisplay('selected-ming-tiles', selectedMingTiles);
}

// 선택된 패 표시 업데이트
function updateSelectedDisplay(containerId, tiles) {
    const display = document.getElementById(containerId);
    display.innerHTML = '';
    tiles.forEach(tile => {
        const img = document.createElement('img');
        img.src = `/static/images/tiles/${tile}.png`;
        img.classList.add('selected-tile');
        display.appendChild(img);
    });

    // 폼에 선택된 값들을 hidden input으로 추가
    updateHiddenInputs();
}

// 폼 제출 시 선택된 값들을 hidden input으로 추가
function updateHiddenInputs() {
    const form = document.querySelector('form');
    
    // 기존 hidden input 제거
    form.querySelectorAll('input[type="hidden"]').forEach(input => input.remove());
    
    // 화료 패
    const winningTile = document.querySelector('#winning-tile-selection .selected');
    if (winningTile) {
        addHiddenInput(form, 'winning_tile', winningTile.dataset.tile);
    }
    
    // 도라 표시패
    selectedDoraTiles.forEach(tile => {
        addHiddenInput(form, 'dora_indicators', tile);
    });
    
    // 안깡 타일
    selectedAnkanTiles.forEach(tile => {
        addHiddenInput(form, 'ankan_tiles', tile);
    });
    
    // 밍 타일
    selectedMingTiles.forEach(tile => {
        addHiddenInput(form, 'ming_tiles', tile);
    });
}

function addHiddenInput(form, name, value) {
    const input = document.createElement('input');
    input.type = 'hidden';
    input.name = name;
    input.value = value;
    form.appendChild(input);
}

// 마작패 정렬 함수
function sortMahjongTiles(tiles) {
    // 타일 종류별 순서 정의
    const suitOrder = {'m': 1, 'p': 2, 's': 3, 'z': 4};
    
    return tiles.sort((a, b) => {
        // 타일 종류 추출 (마지막 문자)
        const suitA = a.slice(-1);
        const suitB = b.slice(-1);
        
        // 종류가 다르면 종류 순서대로 정렬
        if (suitA !== suitB) {
            return suitOrder[suitA] - suitOrder[suitB];
        }
        
        // 적도라 처리 (r5m, r5p, r5s)
        if (a.startsWith('r5') && b.startsWith('5')) {
            return 1; // 적도라는 일반 5 다음에 배치
        }
        if (a.startsWith('5') && b.startsWith('r5')) {
            return -1;
        }
        
        // 같은 종류 내에서는 숫자 순서대로 정렬
        const numA = a.startsWith('r') ? parseInt(a.slice(1, 2)) : parseInt(a.slice(0, -1));
        const numB = b.startsWith('r') ? parseInt(b.slice(1, 2)) : parseInt(b.slice(0, -1));
        
        return numA - numB;
    });
}
