"""
app.py: Server Flask cung cấp giao diện Web dịch thuật đẹp mắt chuẩn Google Translate.
Hỗ trợ:
- Dịch thời gian thực với Beam Search & Greedy Search
- Giao diện 2 cột song ngữ trực quan, hỗ trợ nút Copy và đổi chế độ
- Tự động mở trình duyệt web khi khởi động
"""

import os
import sys
import webbrowser
import threading
from flask import Flask, render_template, request, jsonify

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(CURRENT_DIR, "models")
TEMPLATES_DIR = os.path.join(CURRENT_DIR, "templates")

app = Flask(__name__, template_folder=TEMPLATES_DIR)

# Khởi tạo engine dịch thuật
try:
    from src.inference import TranslatorEngine
except ImportError:
    from inference import TranslatorEngine

MODEL_PATH = os.path.join(MODELS_DIR, "best_transformer.pt")
TOK_EN_PATH = os.path.join(MODELS_DIR, "tokenizer_en.json")
TOK_VI_PATH = os.path.join(MODELS_DIR, "tokenizer_vi.json")

engine = None
try:
    if os.path.exists(MODEL_PATH) and os.path.exists(TOK_EN_PATH) and os.path.exists(TOK_VI_PATH):
        engine = TranslatorEngine(
            model_path=MODEL_PATH,
            tokenizer_en_path=TOK_EN_PATH,
            tokenizer_vi_path=TOK_VI_PATH
        )
        print("✅ Đã nạp thành công mô hình dịch thuật vào Web App!")
    else:
        print("⚠️ Chưa tìm đủ 3 files trong thư mục models/ (best_transformer.pt, tokenizer_en.json, tokenizer_vi.json)")
except Exception as e:
    print(f"❌ Lỗi khởi động mô hình: {e}")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/translate", methods=["POST"])
def translate_api():
    if engine is None:
        return jsonify({
            "success": False,
            "error": "Mô hình chưa được nạp. Vui lòng kiểm tra 3 file trong thư mục models/."
        }), 500

    data = request.get_json() or {}
    text = data.get("text", "").strip()
    mode = data.get("mode", "beam")  # "beam" hoặc "greedy"
    beam_size = int(data.get("beam_size", 4))

    if not text:
        return jsonify({"success": True, "translation": ""})

    try:
        if mode == "greedy":
            result = engine.translate_greedy(text)
        else:
            result = engine.translate_beam_search(text, beam_size=beam_size)

        return jsonify({"success": True, "translation": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("      🌐 KHỞI CHẠY GIAO DIỆN WEB DỊCH THUẬT ANH - VIỆT")
    print("=" * 60)
    print("Đang mở trình duyệt tại: http://127.0.0.1:5000 ...")
    threading.Timer(1.2, open_browser).start()
    app.run(host="127.0.0.1", port=5000, debug=False)
