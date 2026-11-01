# 🇻🇳🇬🇧 English - Vietnamese Neural Machine Translation (From-Scratch Transformer)

Dự án dịch máy nơ-ron (NMT) Anh - Việt tự xây dựng hoàn chỉnh từ đầu bằng **PyTorch thuần** dựa trên kiến trúc của bài báo kinh điển *"Attention Is All You Need"* (Vaswani et al., 2017).

Dự án kết hợp mạng nơ-ron Transformer với **Cơ chế Phân tích Cú pháp Thông minh (POS-Aware Entity Preservation)** và **Từ điển chuyên sâu 104.829 từ** để giải quyết triệt để các bài toán khó nhất trong dịch thuật:
* Phân biệt chính xác **Danh từ riêng vs Danh từ chung** (Apple, Turkey, May, Rose, Bill).
* Bảo toàn 100% tên riêng người Châu Á và quốc tế (*Thanh, Thành, Liu Ching, Tanaka, Kim Min-soo, Elon Musk*).
* Bắt trọn vẹn ngữ nghĩa của **Từ đơn** và **Cụm động từ (Phrasal Verbs)**.
* Chặn lặp từ vô tận bằng thuật toán **Beam Search + No-repeat 3-gram Blocking**.

---

## 📁 Cấu trúc thư mục

```text
├── models/                     # Thư mục lưu trọng số & Tokenizer
│   ├── tokenizer_en.json       # BPE Tokenizer tiếng Anh (24.000 subwords)
│   ├── tokenizer_vi.json       # BPE Tokenizer tiếng Việt (24.000 subwords)
│   └── best_transformer.pt     # Trọng số tối ưu (Tải tự động qua 1 click)
├── templates/                  # Giao diện Web (HTML5/CSS/JS phong cách Google Translate)
├── notebooks/                  # Sổ tay huấn luyện GPU trên Google Colab
│   └── train_colab.ipynb       # Huấn luyện với VinAI PhoMT + CSDL từ điển
├── model.py                    # Kiến trúc Transformer Seq2Seq (Encoder, Decoder, Attention)
├── tokenizer.py                # Mã hóa/giải mã BPE Tokenizer
├── dataset.py                  # PyTorch Dataset, Dynamic Padding & Look-ahead Mask
├── inference.py                # Bộ suy luận thông minh (POS Tagging + Beam Search + Hybrid Dict)
├── translate_cli.py            # Công cụ dịch dòng lệnh (Interactive CLI)
├── app.py                      # Ứng dụng Web Flask dịch thuật thời gian thực
├── cap_nhat_mo_hinh.py         # 🚀 1-Click tự động đồng bộ Model mới nhất từ Google Drive
├── requirements.txt            # Danh sách thư viện cần thiết
└── README.md                   # Tài liệu hướng dẫn
```

---

## ⚡ Hướng Dẫn Cài Đặt & Chạy Ngay (Dành Cho Người Mới)

Chỉ cần 3 bước đơn giản là bạn có thể trải nghiệm toàn bộ hệ thống dịch thuật:

### Bước 1: Clone Repository & Cài đặt thư viện
```bash
git clone https://github.com/TX140907/en--vie-translator-pet-project-.git
cd en--vie-translator-pet-project-
pip install -r requirements.txt
```

### Bước 2: Tự động tải Trọng số Mô hình (1-Click)
Mô hình `best_transformer.pt` (>250MB) được lưu trữ trên Google Drive. Bạn chỉ cần chạy script sau để tự động tải về thư mục `models/`:
```bash
python cap_nhat_mo_hinh.py
```

### Bước 3: Trải nghiệm Dịch thuật

* **Cách 1: Mở Giao diện Web (như Google Translate)**
  ```bash
  python app.py
  ```
  Truy cập trình duyệt tại: **http://127.0.0.1:5000** để trải nghiệm giao diện dịch song ngữ thời gian thực.

* **Cách 2: Dịch nhanh trên Terminal (CLI)**
  ```bash
  # Dịch 1 câu bất kỳ:
  python translate_cli.py "Artificial intelligence is changing the world."

  # Chế độ gõ dịch liên tục:
  python translate_cli.py
  ```

---

## 🚀 Huấn Luyện Mới / Train Tiếp Nối (Dành Cho Người Phát Triển)

Toàn bộ quy trình huấn luyện GPU được đóng gói sẵn trong:
👉 [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb)

* Hỗ trợ nạp các tập ngữ liệu hàng đầu: **VinAI PhoMT** (3 triệu câu), **Helsinki-NLP/opus-100**, và **CSDL từ điển 104.829 từ**.
* Tối ưu hóa phần cứng: FP16 Mixed Precision, Dynamic Padding, và tự động Resume Training từ Google Drive.
