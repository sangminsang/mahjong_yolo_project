from flask import Flask, render_template
from apis.score.routes import score_bp
from apis.discard.routes import discard_bp
from apis.realtime.routes import realtime_bp  # 새로운 블루프린트 추가
from apis.guide.routes import guide_bp  # 새로운 블루프린트 추가

app = Flask(__name__, static_folder='static')

# Secret Key 설정
app.config['SECRET_KEY'] = 'mahjong-secret-key'

# 블루프린트 등록
app.register_blueprint(score_bp, name='score')
app.register_blueprint(discard_bp, name='discard')
app.register_blueprint(realtime_bp, name='realtime')  # 새로운 블루프린트 등록
app.register_blueprint(guide_bp, name='guide')  # 새로운 블루프린트 등록

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, ssl_context='adhoc')

