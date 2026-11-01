"""
cap_nhat_mo_hinh.py: Script Python đồng bộ mô hình mới nhất - chạy trực tiếp trong PyCharm (Miễn phí 100%).
Ưu tiên số 1: Tải trực tiếp từ Google Drive cá nhân của bạn (cấu hình trong drive_link.txt).
Dự phòng: Quét thư mục Downloads.
"""

import os
import sys
import webbrowser

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(CURRENT_DIR, "models")

try:
    from src.sync_model import run_sync
    from src.inference import TranslatorEngine
except ImportError:
    from sync_model import run_sync
    from inference import TranslatorEngine


def main():
    print("\n" + "=" * 65)
    print("      🇻🇳🇬🇧 CÔNG CỤ ĐỒNG BỘ MÔ HÌNH VÀ TOKENIZER MỚI NHẤT")
    print("=" * 65)

    # 1. Chạy tiến trình đồng bộ (Tự động ưu tiên Google Drive hàng đầu)
    run_sync()

    # 2. Tùy chọn mở Web UI hoặc Dịch Thử
    print("\n" + "=" * 65)
    print("Bạn muốn làm gì tiếp theo?")
    print("  [1] Khởi động Giao diện Web Dịch Thuật (Tự động mở trình duyệt)")
    print("  [2] Dịch thử vài câu kiểm tra ngay trên console này")
    print("  [3] Thoát")
    print("=" * 65)

    try:
        choice = input("Nhập lựa chọn của bạn (1, 2 hoặc 3): ").strip()
    except (EOFError, KeyboardInterrupt):
        choice = "3"

    if choice == "1":
        print("\n🚀 Đang mở trình duyệt tại: http://127.0.0.1:5000 ...")
        webbrowser.open("http://127.0.0.1:5000")
        import app
        app.app.run(host="127.0.0.1", port=5000, debug=False)
    elif choice == "2":
        engine = TranslatorEngine(
            model_path=os.path.join(MODELS_DIR, "best_transformer.pt"),
            tokenizer_en_path=os.path.join(MODELS_DIR, "tokenizer_en.json"),
            tokenizer_vi_path=os.path.join(MODELS_DIR, "tokenizer_vi.json")
        )
        print("\n" + "-" * 65)
        print("Gõ câu tiếng Anh bất kỳ để dịch (Gõ 'exit' để thoát):")
        while True:
            try:
                user_text = input("\n[EN] >> ").strip()
                if not user_text or user_text.lower() in ("exit", "quit", "q"):
                    break
                vi_trans = engine.translate_beam_search(user_text, beam_size=4)
                print(f"[VI] -> {vi_trans}")
            except (EOFError, KeyboardInterrupt):
                break
        print("\nĐã hoàn tất!")
    else:
        print("\nĐã hoàn tất đồng bộ! File trọng số đã sẵn sàng.")


if __name__ == "__main__":
    main()
