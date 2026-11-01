# 🇻🇳🇬🇧 English - Vietnamese Neural Machine Translation (From-Scratch Transformer)

Dự án dịch máy nơ-ron (NMT) Anh - Việt tự xây dựng hoàn chỉnh từ đầu bằng **PyTorch thuần** dựa trên kiến trúc của bài báo kinh điển *"Attention Is All You Need"* (Vaswani et al., 2017).

Dự án kết hợp mạng nơ-ron Transformer với **Cơ chế Phân tích Cú pháp Thông minh (POS-Aware Entity Preservation)** và **Từ điển chuyên sâu 104.829 từ** để giải quyết triệt để các bài toán khó nhất trong dịch thuật:
* Phân biệt chính xác **Danh từ riêng vs Danh từ chung** (*Apple, Turkey, May, Rose, Bill*).
* Bảo toàn 100% tên riêng người Châu Á và quốc tế (*Thanh, Thành, Liu Ching, Tanaka, Kim Min-soo, Elon Musk*).
* Bắt trọn vẹn ngữ nghĩa của **Từ đơn** và **Cụm động từ (Phrasal Verbs)**.
* Chặn lặp từ vô tận bằng thuật toán **Beam Search + No-repeat 3-gram Blocking**.

---

## 📁 Cấu Trúc Thư Mục & Chức Năng Từng Tệp Tin

Dự án được phân cấp rõ ràng giữa các tệp **giao diện thực thi trực tiếp** và các **module mã nguồn lõi** bên trong thư mục `src/`:

```text
├── app.py                      # 🌐 [Chạy trực tiếp] Khởi động Giao diện Web Flask dịch thuật
├── translate_cli.py            # 💻 [Chạy trực tiếp] Dịch tương tác qua dòng lệnh Terminal
├── cap_nhat_mo_hinh.py         # 🚀 [Chạy trực tiếp] 1-Click đồng bộ trọng số từ Google Drive về máy
├── drive_link.txt              # ⚙️ [Cấu hình] Nơi dán Link Google Drive cá nhân của người huấn luyện
├── requirements.txt            # 📦 Danh sách thư viện Python cần thiết
├── README.md                   # 📖 Tài liệu hướng dẫn chi tiết
│
├── src/                        # 🧠 [Mã nguồn lõi nội bộ - Không cần chạy trực tiếp]
│   ├── __init__.py             # Đánh dấu package Python
│   ├── model.py                # Kiến trúc Transformer Seq2Seq (Encoder, Decoder, Attention, Positional Encoding)
│   ├── tokenizer.py            # Lớp bọc mã hóa / giải mã Byte-Pair Encoding (BPE Tokenizer)
│   ├── dataset.py              # PyTorch Dataset, Collate Function, Dynamic Padding & Attention Masks
│   ├── inference.py            # Bộ suy luận thông minh (POS Tagging NLTK, Beam Search, sửa tên riêng, từ điển)
│   ├── sync_model.py           # Module kết nối Google Drive qua gdown và kiểm tra tính toàn vẹn trọng số
│   └── test_pipeline.py        # Kiểm thử nhanh toàn bộ luồng tính toán toán học (Dry-run test)
│
├── models/                     # 📂 Thư mục chứa Tokenizer & Trọng số
│   ├── tokenizer_en.json       # BPE Tokenizer tiếng Anh (24.000 subwords) - [Đã có sẵn trên Git]
│   ├── tokenizer_vi.json       # BPE Tokenizer tiếng Việt (24.000 subwords) - [Đã có sẵn trên Git]
│   └── best_transformer.pt     # Trọng số mô hình (>250MB) - [Tự động tải về qua cap_nhat_mo_hinh.py]
│
├── templates/                  # 🎨 Giao diện Web
│   └── index.html              # Trang Web dịch song ngữ phong cách Google Translate (HTML5/CSS3/JS)
│
├── notebooks/                  # ⚡ Huấn luyện GPU
│   └── train_colab.ipynb       # Sổ tay huấn luyện Google Colab (VinAI PhoMT + Opus-100 + CSDL Từ điển)
│
└── data/                       # 📊 Dữ liệu huấn luyện nội bộ
    └── .gitkeep                # Giữ thư mục data trên Git (Dữ liệu nặng được bỏ qua bởi .gitignore)
```

---

## ⚡ Hướng Dẫn Dành Cho Người Dùng (Chạy Mô Hình & Dịch Thuật)

### Bước 1: Clone Repository & Cài đặt thư viện
```bash
git clone https://github.com/TX140907/en--vie-translator-pet-project-.git
cd en--vie-translator-pet-project-
pip install -r requirements.txt
```

### Bước 2: Chuẩn bị file trọng số `best_transformer.pt`
File trọng số mô hình `best_transformer.pt` (>250MB) không đưa lên Git để tránh vượt hạn mức 100MB của GitHub:
* Nếu bạn được tác giả chia sẻ link hoặc bạn đã train mô hình: Hãy xem mục bên dưới để tải tự động.
* Hoặc nếu bạn đã có file `best_transformer.pt`, chỉ cần copy nó đặt vào thư mục `models/`.

### Bước 3: Khởi chạy dịch thuật

* **Cách 1: Khởi động Giao diện Web (Khuyên dùng)**
  ```bash
  python app.py
  ```
  Trình duyệt sẽ tự động mở tại địa chỉ: **http://127.0.0.1:5000** để bạn trải nghiệm dịch song ngữ thời gian thực.

* **Cách 2: Dịch nhanh trên Terminal (CLI)**
  ```bash
  # Dịch 1 câu bất kỳ:
  python translate_cli.py "Artificial intelligence is changing the world."

  # Chế độ gõ dịch liên tục:
  python translate_cli.py
  ```

---

## 🚀 Hướng Dẫn Dành Cho Người Huấn Luyện (Train & Tự Lấy Model Từ Drive)

Nếu bạn muốn tự huấn luyện mô hình của riêng mình:

1. Mở file [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb) trên **Google Colab** (chọn GPU T4/A100).
2. Khi huấn luyện, notebook sẽ tự động lưu file `best_transformer.pt` trực tiếp vào **Google Drive cá nhân của bạn**.
3. **Đồng bộ về máy tính cá nhân chỉ với 1 click:**
   * Vào Google Drive của bạn, click chuột phải vào file (hoặc thư mục chứa) `best_transformer.pt` -> Chọn **Chia sẻ** -> Đổi sang chế độ **"Bất kỳ ai có đường liên kết đều có thể xem" (Viewer)** -> Sao chép liên kết.
   * Dán link đó vào file [`drive_link.txt`](drive_link.txt).
   * Chạy lệnh sau trên máy tính của bạn:
     ```bash
     python cap_nhat_mo_hinh.py
     ```
   * Script sẽ tự động kết nối vào Google Drive của bạn, tải `best_transformer.pt` về và ghi đè thẳng vào thư mục `models/` hoàn toàn tự động!
