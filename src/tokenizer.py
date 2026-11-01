"""
tokenizer.py: Xây dựng và quản lý BPE Tokenizer (Byte-Pair Encoding) bằng Hugging Face `tokenizers`.
Hỗ trợ:
- Huấn luyện BPE Tokenizer riêng biệt cho tiếng Anh và tiếng Việt
- Xử lý các token đặc biệt: <pad>, <sos>, <eos>, <unk>
- Lưu và tải cấu hình Tokenizer định dạng JSON
- Mã hóa (encode) văn bản thành Tensor và Giải mã (decode) Token IDs thành chuỗi ký tự
"""

import os
from typing import List, Union
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import torch

PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3


class TranslationTokenizer:
    def __init__(self, tokenizer_path: str = None):
        """Khởi tạo Tokenizer từ file JSON đã lưu, hoặc tạo mới nếu chưa có."""
        if tokenizer_path and os.path.exists(tokenizer_path):
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
        else:
            self.tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
            self.tokenizer.pre_tokenizer = Whitespace()

    def train_from_iterator(self, text_iterator, vocab_size: int = 16000, min_frequency: int = 2):
        """Huấn luyện BPE tokenizer trực tiếp từ danh sách câu (iterator)."""
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS,
            show_progress=True
        )
        self.tokenizer.train_from_iterator(text_iterator, trainer=trainer)

    def train_from_files(self, file_paths: List[str], vocab_size: int = 16000, min_frequency: int = 2):
        """Huấn luyện BPE tokenizer từ danh sách các file văn bản."""
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS,
            show_progress=True
        )
        self.tokenizer.train(files=file_paths, trainer=trainer)

    def save(self, path: str):
        """Lưu cấu hình tokenizer ra file JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.tokenizer.save(path)

    @property
    def vocab_size(self) -> int:
        """Kích thước từ vựng."""
        return self.tokenizer.get_vocab_size()

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Mã hóa chuỗi văn bản thành danh sách token ID (kèm <sos> và <eos> nếu cần)."""
        encoding = self.tokenizer.encode(text)
        ids = encoding.ids
        if add_special_tokens:
            ids = [SOS_IDX] + ids + [EOS_IDX]
        return ids

    def decode(self, token_ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = True) -> str:
        """Giải mã danh sách token ID thành câu hoàn chỉnh."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().tolist()

        # Loại bỏ các token đặc biệt phía sau nếu có
        cleaned_ids = []
        for tid in token_ids:
            if tid == EOS_IDX:
                break
            if skip_special_tokens and tid in (PAD_IDX, SOS_IDX, UNK_IDX):
                continue
            cleaned_ids.append(tid)

        return self.tokenizer.decode(cleaned_ids)


def build_or_load_tokenizers(en_texts: List[str], vi_texts: List[str], 
                            save_dir: str = "models", vocab_size: int = 16000):
    """Tiện ích tạo hoặc tải 2 tokenizer tiếng Anh và tiếng Việt."""
    en_path = os.path.join(save_dir, "tokenizer_en.json")
    vi_path = os.path.join(save_dir, "tokenizer_vi.json")

    if os.path.exists(en_path) and os.path.exists(vi_path):
        tok_en = TranslationTokenizer(en_path)
        tok_vi = TranslationTokenizer(vi_path)
    else:
        print("[Tokenizer] Huấn luyện Tokenizer tiếng Anh...")
        tok_en = TranslationTokenizer()
        tok_en.train_from_iterator(en_texts, vocab_size=vocab_size)
        tok_en.save(en_path)

        print("[Tokenizer] Huấn luyện Tokenizer tiếng Việt...")
        tok_vi = TranslationTokenizer()
        tok_vi.train_from_iterator(vi_texts, vocab_size=vocab_size)
        tok_vi.save(vi_path)
        print(f"[Tokenizer] Đã lưu 2 bộ Tokenizer vào: {save_dir}")

    return tok_en, tok_vi
