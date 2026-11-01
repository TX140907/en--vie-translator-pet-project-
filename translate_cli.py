"""
translate_cli.py: Giao diện dòng lệnh (CLI) để dịch trực tiếp câu tiếng Anh sang tiếng Việt.
Hỗ trợ:
- Dịch từng câu truyền từ tham số dòng lệnh
- Chế độ Interactive: Gõ câu tiếng Anh và nhận kết quả dịch tiếng Việt tức thì
- Tự động chọn Greedy Search hoặc Beam Search
"""

import os
import sys
import argparse

# Hỗ trợ in tiếng Việt chuẩn UTF-8 trên Windows Terminal
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(CURRENT_DIR, "models")

from inference import TranslatorEngine


def main():
    parser = argparse.ArgumentParser(description="🇻🇳🇬🇧 Công cụ dịch máy Anh - Việt (Transformer from Scratch)")
    parser.add_argument("text", nargs="?", help="Câu tiếng Anh cần dịch (đặt trong dấu ngoặc kép)")
    parser.add_argument("--beam", "-b", type=int, default=4, help="Kích thước chùm tia Beam Search (Mặc định: 4)")
    parser.add_argument("--greedy", "-g", action="store_true", help="Sử dụng thuật toán Greedy Search thay cho Beam Search")
    parser.add_argument("--model", "-m", default="best_transformer.pt", help="Tên file trọng số trong thư mục models/")
    args = parser.parse_args()

    model_path = os.path.join(MODELS_DIR, args.model)
    tok_en_path = os.path.join(MODELS_DIR, "tokenizer_en.json")
    tok_vi_path = os.path.join(MODELS_DIR, "tokenizer_vi.json")

    if not os.path.exists(model_path):
        print(f"❌ [Lỗi] Không tìm thấy file trọng số: {model_path}")
        print("Vui lòng tải file 'best_transformer.pt' từ Google Drive về thư mục Translator/models/.")
        sys.exit(1)

    print("=" * 65)
    print("      🇻🇳🇬🇧 CỘNG CỤ DỊCH ANH - VIỆT (TRANSFORMER FROM SCRATCH)")
    print("=" * 65)
    print("⏳ Đang khởi tạo mô hình và nạp trọng số...")

    engine = TranslatorEngine(
        model_path=model_path,
        tokenizer_en_path=tok_en_path,
        tokenizer_vi_path=tok_vi_path
    )

    def do_translate(input_text: str) -> str:
        if args.greedy:
            return engine.translate_greedy(input_text)
        return engine.translate_beam_search(input_text, beam_size=args.beam)

    # 1. Nếu người dùng truyền câu trực tiếp từ dòng lệnh
    if args.text:
        print(f"\n[EN] Tiếng Anh:  {args.text}")
        output_vi = do_translate(args.text)
        print(f"[VI] Tiếng Việt: {output_vi}\n")
        return

    # 2. Chế độ nhập tương tác liên tục
    print("\n✅ Mô hình đã sẵn sàng!")
    mode_name = "Greedy Search" if args.greedy else f"Beam Search (k={args.beam})"
    print(f"Chế độ dịch: {mode_name}")
    print("Gõ câu tiếng Anh bất kỳ và bấm Enter để dịch (Gõ 'exit' hoặc 'quit' để thoát):")
    print("-" * 65)

    while True:
        try:
            user_input = input("\n[EN] >> ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Tạm biệt!")
                break

            result = do_translate(user_input)
            print(f"[VI] -> {result}")
        except (KeyboardInterrupt, EOFError):
            print("\nTạm biệt!")
            break


if __name__ == "__main__":
    main()
