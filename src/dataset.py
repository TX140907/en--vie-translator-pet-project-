"""
dataset.py: Xử lý dữ liệu song ngữ, tạo PyTorch Dataset, Collate Function và Masking.
Hỗ trợ:
- Tự động tạo Padding Mask (src_mask) để bỏ qua các token <pad>
- Tự động tạo Causal Mask (Look-ahead mask) cho tgt_mask để Decoder không nhìn trộm tương lai
- Dynamic Padding: Tự động đệm câu theo chiều dài lớn nhất của từng batch (tiết kiệm bộ nhớ)
"""

import torch
from torch.utils.data import Dataset
from typing import List, Tuple

try:
    from src.tokenizer import PAD_IDX, TranslationTokenizer
except ImportError:
    from tokenizer import PAD_IDX, TranslationTokenizer


def create_padding_mask(seq: torch.Tensor, pad_idx: int = PAD_IDX) -> torch.Tensor:
    """
    Tạo mask cho token <pad>.
    seq: (batch_size, seq_len)
    Output mask: (batch_size, 1, 1, seq_len) -> True ở vị trí từ thật, False ở vị trí <pad>.
    """
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


def create_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """
    Tạo ma trận tam giác dưới (Look-ahead / Causal Mask) kích thước (1, 1, seq_len, seq_len).
    Chỉ cho phép từ vị trí t chú ý đến các từ từ 0 đến t.
    """
    mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).bool()
    return mask.unsqueeze(0).unsqueeze(1)


def create_tgt_mask(tgt: torch.Tensor, pad_idx: int = PAD_IDX) -> torch.Tensor:
    """
    Kết hợp giữa Padding Mask và Causal Mask cho Target sequence của Decoder.
    tgt: (batch_size, tgt_len)
    Output mask: (batch_size, 1, tgt_len, tgt_len)
    """
    batch_size, tgt_len = tgt.size()
    pad_mask = create_padding_mask(tgt, pad_idx)       # (batch, 1, 1, tgt_len)
    causal_mask = create_causal_mask(tgt_len, tgt.device)  # (1, 1, tgt_len, tgt_len)
    return pad_mask & causal_mask


class TranslationDataset(Dataset):
    """Dataset chứa các cặp câu song ngữ đã được lọc sạch."""
    def __init__(self, en_sentences: List[str], vi_sentences: List[str], 
                 tok_en: TranslationTokenizer, tok_vi: TranslationTokenizer,
                 max_len: int = 128):
        assert len(en_sentences) == len(vi_sentences), "Số lượng câu En và Vi phải bằng nhau!"
        self.data = []

        # Lọc câu quá dài hoặc rỗng
        for en, vi in zip(en_sentences, vi_sentences):
            en_str = en.strip()
            vi_str = vi.strip()
            if not en_str or not vi_str:
                continue

            en_ids = tok_en.encode(en_str, add_special_tokens=True)
            vi_ids = tok_vi.encode(vi_str, add_special_tokens=True)

            if len(en_ids) <= max_len and len(vi_ids) <= max_len:
                self.data.append((en_ids, vi_ids))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch: List[Tuple[List[int], List[int]]]):
    """
    Gộp các mẫu thành batch và thực hiện Dynamic Padding.
    - Input của Decoder (tgt_input): bỏ token cuối <eos>
    - Nhãn dự đoán của Decoder (tgt_label): bỏ token đầu <sos>
    """
    src_list, tgt_list = zip(*batch)

    max_src_len = max(len(s) for s in src_list)
    max_tgt_len = max(len(t) for t in tgt_list)

    batch_size = len(batch)
    src_padded = torch.full((batch_size, max_src_len), PAD_IDX, dtype=torch.long)
    tgt_padded = torch.full((batch_size, max_tgt_len), PAD_IDX, dtype=torch.long)

    for i, (src_ids, tgt_ids) in enumerate(zip(src_list, tgt_list)):
        src_padded[i, :len(src_ids)] = torch.tensor(src_ids, dtype=torch.long)
        tgt_padded[i, :len(tgt_ids)] = torch.tensor(tgt_ids, dtype=torch.long)

    # Teacher Forcing:
    # tgt_input: [sos, w1, w2, ..., pad]
    # tgt_label: [w1, w2, ..., eos, pad]
    tgt_input = tgt_padded[:, :-1]
    tgt_label = tgt_padded[:, 1:]

    src_mask = create_padding_mask(src_padded)
    tgt_mask = create_tgt_mask(tgt_input)

    return src_padded, tgt_input, tgt_label, src_mask, tgt_mask
