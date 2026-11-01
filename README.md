# 🇻🇳🇬🇧 English - Vietnamese Neural Machine Translation (From-Scratch Transformer)

Dự án dịch máy nơ-ron (NMT) Anh - Việt tự xây dựng hoàn chỉnh từ đầu bằng **PyTorch thuần** dựa trên kiến trúc của bài báo kinh điển *"Attention Is All You Need"* (Vaswani et al., 2017).

Dự án kết hợp mạng nơ-ron Transformer với **Cơ chế Phân tích Cú pháp Thông minh (POS-Aware Entity Preservation)** và **Từ điển chuyên sâu 104.829 từ** để giải quyết triệt để các bài toán khó nhất trong dịch thuật:
* Phân biệt chính xác **Danh từ riêng vs Danh từ chung** (*Apple, Turkey, May, Rose, Bill*).
* Bảo toàn 100% tên riêng người Châu Á và quốc tế (*Thanh, Thành, Liu Ching, Tanaka, Kim Min-soo, Elon Musk*).
* Bắt trọn vẹn ngữ nghĩa của **Từ đơn** và **Cụm động từ (Phrasal Verbs)**.
* Chặn lặp từ vô tận bằng thuật toán **Beam Search + No-repeat 3-gram Blocking**.

---

## ⚡ Hướng Dẫn Cài Đặt & Chạy Ngay (3 Bước Đơn Giản)

Dự án đã được đóng gói sẵn sàng. Người dùng clone về là có thể chạy ngay lập tức mà không cần bất kỳ bước huấn luyện hay cấu hình phức tạp nào:

### Bước 1: Clone Repository & Cài đặt thư viện
```bash
git clone https://github.com/TX140907/en--vie-translator-pet-project-.git
cd en--vie-translator-pet-project-
pip install -r requirements.txt
```

### Bước 2: Tải Trọng số Mô hình (1-Click Tự Động)
Trọng số mô hình `best_transformer.pt` (>250MB) được lưu trữ trên Google Drive. Bạn chỉ cần chạy script sau, hệ thống sẽ tự động tải weights từ Drive về thư mục `models/`:
```bash
python cap_nhat_mo_hinh.py
```
*(Link Google Drive chứa mô hình đã được cấu hình sẵn trong dự án: `https://drive.google.com/drive/folders/1S37n0rTYb1ANVz0hmQJNwXK37sSrr5jB?hl=vi`)*

### Bước 3: Trải nghiệm Dịch thuật

* **Cách 1: Khởi động Giao diện Web (Khuyên dùng)**
  ```bash
  python app.py
  ```
  Trình duyệt sẽ tự động mở tại: **http://127.0.0.1:5000** để trải nghiệm giao diện dịch song ngữ thời gian thực phong cách Google Translate.

* **Cách 2: Dịch nhanh trên Terminal (CLI)**
  ```bash
  # Dịch 1 câu bất kỳ:
  python translate_cli.py "Artificial intelligence is changing the world."

  # Chế độ gõ dịch liên tục:
  python translate_cli.py
  ```

---

## 📁 Cấu Trúc Thư Mục Dự Án

```text
├── app.py                      # 🌐 [Chạy trực tiếp] Giao diện Web Flask dịch thuật thời gian thực
├── translate_cli.py            # 💻 [Chạy trực tiếp] Công cụ dịch dòng lệnh Terminal tương tác
├── cap_nhat_mo_hinh.py         # 🚀 [Chạy trực tiếp] 1-Click tự động tải mô hình từ Google Drive
├── drive_link.txt              # 🔗 Link Google Drive chứa mô hình tối ưu đã huấn luyện
├── requirements.txt            # 📦 Danh sách thư viện Python cần thiết
├── README.md                   # 📖 Tài liệu hướng dẫn sử dụng
│
├── src/                        # 🧠 [Mã nguồn backend lõi]
│   ├── __init__.py             # Package marker
│   ├── model.py                # Kiến trúc Transformer Seq2Seq (Encoder, Decoder, Attention)
│   ├── tokenizer.py            # Lớp bọc bộ mã hóa BPE Tokenizer
│   ├── dataset.py              # Xử lý Dynamic Padding & Attention Masks
│   ├── inference.py            # Bộ suy luận thông minh (POS Tagging, Beam Search, sửa tên riêng)
│   └── sync_model.py           # Module tải tự động và xác minh tính toàn vẹn của mô hình
│
├── models/                     # 📂 Thư mục chứa Tokenizer & Weights
│   ├── tokenizer_en.json       # BPE Tokenizer tiếng Anh (24.000 subwords) - [Có sẵn trong repo]
│   ├── tokenizer_vi.json       # BPE Tokenizer tiếng Việt (24.000 subwords) - [Có sẵn trong repo]
│   └── best_transformer.pt     # Trọng số mô hình (Tự động tải về qua cap_nhat_mo_hinh.py)
│
└── templates/                  # 🎨 Giao diện người dùng
    └── index.html              # Giao diện Web dịch thuật song ngữ (HTML5/CSS3/JavaScript)
```
