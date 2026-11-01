# 🇻🇳🇬🇧 English - Vietnamese Neural Machine Translation (Pet Project)

Dự án cá nhân (Pet Project) tìm hiểu, xây dựng và thực nghiệm mô hình Dịch máy nơ-ron (Neural Machine Translation - NMT) từ tiếng Anh sang tiếng Việt. 

Mô hình được xây dựng dựa trên kiến trúc **Transformer Seq2Seq** kinh điển (*"Attention Is All You Need"*, Vaswani et al., 2017) và được lập trình hoàn toàn bằng **PyTorch thuần** từ đầu (không sử dụng các thư viện mô hình đóng gói sẵn).

---

## 📊 Dữ Liệu Huấn Luyện (Training Datasets)

Mô hình được huấn luyện và tinh chỉnh trên các tập ngữ liệu song ngữ Anh - Việt công khai:

1. **VinAI PhoMT** (~3.02 triệu cặp câu): Tập dữ liệu dịch máy Anh - Việt quy mô lớn của VinAI Research, bao quát nhiều thể loại văn bản đa dạng như tin tức, phụ đề phim, đối thoại và tài liệu học thuật.
2. **OPUS-100 (Helsinki-NLP)** (~1 triệu cặp câu): Tập dữ liệu song ngữ được trích xuất từ kho ngữ liệu mở OPUS, giúp bổ trợ thêm các mẫu câu giao tiếp đời sống.
3. **VietAI MTet**: Dữ liệu song ngữ được chuẩn hóa, hỗ trợ làm giàu ngữ cảnh và câu cú tiếng Việt tự nhiên.
4. **Từ điển song ngữ Anh - Việt & Core Vocabulary** (~104.000 mục từ): Cung cấp các cặp từ vựng đơn, thành ngữ và cụm động từ cơ bản để hỗ trợ mô hình nhận diện tốt hơn các từ đơn lẻ và thuật ngữ thông dụng.

---

## 💡 Một Số Ưu Thế & Đặc Điểm Kỹ Thuật Cơ Bản

Dự án tập trung vào việc hiện thực hóa các kỹ thuật cốt lõi một cách thực tế, không cường điệu hóa kết quả:

* **Tự xây dựng kiến trúc từ đầu (From Scratch)**: Toàn bộ các khối `MultiHeadAttention`, `PositionalEncoding`, `TransformerEncoder`, `TransformerDecoder` và cơ chế tạo mặt nạ (`Padding Mask`, `Causal Mask`) đều được viết trực tiếp bằng PyTorch, thuận tiện cho việc học tập và nghiên cứu cấu trúc bên trong của Transformer.
* **BPE Tokenization riêng biệt**: Sử dụng thuật toán Byte-Pair Encoding (thư viện `tokenizers`) được huấn luyện riêng cho từng ngôn ngữ (24.000 subwords cho tiếng Anh và 24.000 subwords cho tiếng Việt), giúp cân bằng giữa độ lớn từ vựng và chi phí tính toán.
* **Hỗ trợ xử lý danh từ riêng (Entity Handling)**: Kết hợp kỹ thuật gán nhãn từ loại (POS Tagging bằng NLTK) để nhận diện các danh từ riêng (tên người, địa danh, thương hiệu quốc tế và Việt Nam), tạm thời neo giữ trước khi qua mạng nơ-ron nhằm giảm thiểu hiện tượng BPE băm vụn hoặc dịch sai tên riêng.
* **Thuật toán giải mã Beam Search**: Hỗ trợ tìm kiếm chùm tia (Beam Search với độ rộng $k=4$) kết hợp chặn lặp n-gram (`no-repeat 3-gram`) giúp câu dịch tự nhiên và hạn chế đáng kể tình trạng lặp từ so với thuật toán Greedy Search.
* **Đóng gói sẵn sàng chạy (Production-ready Inference)**: Tách biệt rõ ràng giữa mã nguồn backend (`src/`) và giao diện người dùng; đi kèm giao diện Web Flask và công cụ dòng lệnh (CLI) phục vụ việc trải nghiệm trực tiếp.

---

## ⚡ Hướng Dẫn Cài Đặt & Chạy Thử (3 Bước)

Người dùng kéo (clone) repo về máy là có thể chạy thử nghiệm được ngay:

### Bước 1: Clone Repository & Cài đặt thư viện
```bash
git clone https://github.com/TX140907/en--vie-translator-pet-project-.git
cd en--vie-translator-pet-project-
pip install -r requirements.txt
```

### Bước 2: Tải Trọng số Mô hình (1-Click Tự Động)
Trọng số mô hình `best_transformer.pt` (>250MB) được lưu trữ trên Google Drive. Bạn chỉ cần chạy lệnh sau để tự động tải weights về thư mục `models/`:
```bash
python cap_nhat_mo_hinh.py
```
*(Link Google Drive chứa trọng số đã được cấu hình sẵn trong `drive_link.txt`: `https://drive.google.com/drive/folders/1S37n0rTYb1ANVz0hmQJNwXK37sSrr5jB?hl=vi`)*

### Bước 3: Trải nghiệm Dịch thuật

* **Cách 1: Mở Giao diện Web (Khuyên dùng)**
  ```bash
  python app.py
  ```
  Truy cập trình duyệt tại: **http://127.0.0.1:5000** để sử dụng giao diện dịch song ngữ thời gian thực.

* **Cách 2: Dịch nhanh trên dòng lệnh Terminal (CLI)**
  ```bash
  # Dịch 1 câu bất kỳ:
  python translate_cli.py "Artificial intelligence is changing the world."

  # Chế độ gõ dịch liên tục:
  python translate_cli.py
  ```

---

## 📁 Cấu Trúc Thư Mục

```text
├── app.py                      # 🌐 [Chạy trực tiếp] Giao diện Web Flask dịch thuật thời gian thực
├── translate_cli.py            # 💻 [Chạy trực tiếp] Công cụ dịch dòng lệnh Terminal tương tác
├── cap_nhat_mo_hinh.py         # 🚀 [Chạy trực tiếp] 1-Click tự động tải mô hình từ Google Drive
├── drive_link.txt              # 🔗 Link Google Drive chứa mô hình đã huấn luyện
├── requirements.txt            # 📦 Danh sách thư viện Python cần thiết
├── README.md                   # 📖 Tài liệu hướng dẫn sử dụng
│
├── src/                        # 🧠 [Mã nguồn backend nội bộ]
│   ├── __init__.py             # Package marker
│   ├── model.py                # Kiến trúc Transformer Seq2Seq (Encoder, Decoder, Attention)
│   ├── tokenizer.py            # Lớp bọc bộ mã hóa BPE Tokenizer
│   ├── dataset.py              # Xử lý Dynamic Padding & Attention Masks
│   ├── inference.py            # Bộ suy luận (POS Tagging, Beam Search, bảo tồn danh từ riêng)
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
