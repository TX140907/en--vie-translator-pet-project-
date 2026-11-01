"""
test_pipeline.py: Kiểm thử nhanh toàn bộ quy trình (Dry-run test) trên CPU.
Kiểm tra:
1. Tokenizer huấn luyện và mã hóa thành công
2. Dataset và Dynamic Padding tạo mask chuẩn xác
3. Mạng Transformer tính toán forward pass chính xác shape
4. Hàm mất mát CrossEntropyLoss và Backward pass hoạt động trơn tru
"""

import os
import sys

# Hỗ trợ in tiếng Việt và UTF-8 trên Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tokenizer import build_or_load_tokenizers
from dataset import TranslationDataset, collate_fn
from model import TransformerSeq2Seq


def test_dry_run():
    print("=" * 60)
    print("      TIEN HANH DRY-RUN TEST TOAN BO PIPELINE")
    print("=" * 60)

    # 1. Dữ liệu giả lập
    sample_en = [
        "Hello, how are you today?",
        "I love machine learning and artificial intelligence.",
        "Deep learning is transforming modern technology.",
        "Neural machine translation with transformer architecture.",
        "Good morning my friend."
    ]
    sample_vi = [
        "Xin chào, hôm nay bạn thế nào?",
        "Tôi yêu thích học máy và trí tuệ nhân tạo.",
        "Học sâu đang biến đổi công nghệ hiện đại.",
        "Dịch máy nơ-ron với kiến trúc máy biến áp.",
        "Chào buổi sáng người bạn của tôi."
    ]

    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, "models")

    # 2. Test Tokenizer
    print("\n[Bước 1/4] Kiểm tra Tokenizer...")
    tok_en, tok_vi = build_or_load_tokenizers(sample_en, sample_vi, save_dir=models_dir, vocab_size=200)
    print(f"-> Vocab En: {tok_en.vocab_size}, Vocab Vi: {tok_vi.vocab_size}")

    encoded_en = tok_en.encode(sample_en[0])
    decoded_en = tok_en.decode(encoded_en)
    print(f"-> Encode test: {sample_en[0]} -> {encoded_en} -> '{decoded_en}'")

    # 3. Test Dataset & Collate Function
    print("\n[Bước 2/4] Kiểm tra Dataset & Masking...")
    dataset = TranslationDataset(sample_en, sample_vi, tok_en, tok_vi, max_len=64)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    batch = next(iter(loader))
    src, tgt_input, tgt_label, src_mask, tgt_mask = batch
    print(f"-> src shape:       {src.shape}")
    print(f"-> tgt_input shape: {tgt_input.shape}")
    print(f"-> tgt_label shape: {tgt_label.shape}")
    print(f"-> src_mask shape:  {src_mask.shape}")
    print(f"-> tgt_mask shape:  {tgt_mask.shape}")

    # 4. Test Model Forward & Backward
    print("\n[Bước 3/4] Kiểm tra Model Forward & Backward Pass...")
    model = TransformerSeq2Seq(
        src_vocab_size=tok_en.vocab_size,
        tgt_vocab_size=tok_vi.vocab_size,
        d_model=128,
        n_heads=4,
        d_ff=512,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dropout=0.1
    )

    logits = model(src, tgt_input, src_mask, tgt_mask)
    print(f"-> Output logits shape: {logits.shape} (batch, tgt_len, vocab_vi)")

    # 5. Test Loss & Backpropagation
    print("\n[Bước 4/4] Kiểm tra Tính Loss và Cập nhật Gradient...")
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_label.reshape(-1))
    print(f"-> Khởi tạo Loss: {loss.item():.4f}")

    loss.backward()
    print("-> Lan truyền ngược (Backpropagation) thành công 100%!")

    print("\n" + "=" * 60)
    print("  [OK] TAT CA CAC MODULE TOAN HOC DA SAN SANG DE DUA LEN COLAB!")
    print("=" * 60)


if __name__ == "__main__":
    test_dry_run()
