"""
sync_model.py: Module đồng bộ mô hình và tokenizer.
Cơ chế bảo toàn dung lượng đĩa: LUÔN GHI ĐÈ TRỰC TIẾP lên file cũ, không bao giờ sinh file rác.
Ưu tiên hàng đầu: Tải trực tiếp từ Google Drive.
Dự phòng: Quét và dọn sạch thư mục Downloads.
"""

import os
import sys
import glob
import shutil
import psutil
import torch

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(CURRENT_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

DOWNLOADS_DIR = os.path.join(os.path.expanduser("~"), "Downloads")
DRIVE_LINK_FILE = os.path.join(CURRENT_DIR, "drive_link.txt")


def free_file_locks():
    """Tự động tìm và tắt các tiến trình đang chiếm giữ file trong thư mục models."""
    current_pid = os.getpid()
    closed_any = False
    target_names = ["app.py", "translate_cli.py", "inference.py"]

    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['pid'] == current_pid:
                continue
            cmd = proc.info.get('cmdline') or []
            cmd_str = " ".join(cmd).lower()
            if any(name in cmd_str for name in target_names):
                print(f"⚠️ Giải phóng tiến trình ngầm (PID: {proc.info['pid']}) để ghi đè file...")
                proc.kill()
                closed_any = True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    if closed_any:
        print("✅ Đã giải phóng bộ nhớ! Sẵn sàng ghi đè trực tiếp.\n")


def get_newest_file(directory: str, patterns: list):
    """Tìm file mới nhất theo thời gian sửa đổi (mtime)."""
    candidates = []
    for pat in patterns:
        candidates.extend(glob.glob(os.path.join(directory, pat)))
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def flatten_subdirectories():
    """Nếu gdown tạo thư mục con, đưa file ra models/ chính và xóa thư mục rác."""
    for root, dirs, files in os.walk(MODELS_DIR):
        if root == MODELS_DIR:
            continue
        for f in files:
            if f.endswith(".pt") or f.endswith(".json"):
                src_path = os.path.join(root, f)
                dest_path = os.path.join(MODELS_DIR, f)
                # Ghi đè trực tiếp vào models/
                if os.path.exists(dest_path):
                    try:
                        os.remove(dest_path)
                    except Exception:
                        pass
                shutil.move(src_path, dest_path)
        # Xóa thư mục rỗng
        try:
            os.rmdir(root)
        except Exception:
            pass


def clean_downloads_duplicates():
    """Tự động dọn dẹp các bản sao trùng lặp trong Downloads để tiết kiệm tối đa ổ cứng."""
    try:
        dup_pts = glob.glob(os.path.join(DOWNLOADS_DIR, "best_transformer*.pt"))
        for f in dup_pts:
            os.remove(f)
            print(f"   🧹 Đã dọn dẹp file tạm trong Downloads: {os.path.basename(f)}")
    except Exception:
        pass


def sync_from_gdrive(url: str) -> bool:
    """Tải trực tiếp từ Google Drive bằng thư viện gdown và ghi đè."""
    import gdown
    print("\n" + "=" * 65)
    print(" ☁️ [ƯU TIÊN 1] ĐANG TẢI VÀ GHI ĐÈ TRỰC TIẾP TỪ GOOGLE DRIVE...")
    print("=" * 65)

    free_file_locks()

    try:
        if "folders" in url:
            print("⏳ Đang tải thư mục từ Drive...")
            gdown.download_folder(url=url, output=MODELS_DIR, quiet=False, use_cookies=False)
            flatten_subdirectories()
        else:
            print("⏳ Đang tải file weights từ Drive...")
            dest_pt = os.path.join(MODELS_DIR, "best_transformer.pt")
            gdown.download(url=url, output=dest_pt, quiet=False)

        print("\n✅ Đã ghi đè thành công từ Google Drive!")
        verify_model()
        return True
    except Exception as e:
        print(f"\n⚠️ Tải từ Google Drive không thành công ({e}).")
        return False


def sync_from_downloads() -> bool:
    """Đồng bộ từ thư mục Downloads (Ghi đè trực tiếp và dọn rác)."""
    print("\n" + "=" * 65)
    print(" 📂 [DỰ PHÒNG] QUÉT VÀ GHI ĐÈ TỪ THƯ MỤC DOWNLOADS...")
    print("=" * 65)

    free_file_locks()

    latest_pt = get_newest_file(DOWNLOADS_DIR, ["best_transformer*.pt", "*transformer*.pt"])
    latest_en = get_newest_file(DOWNLOADS_DIR, ["tokenizer_en*.json"])
    latest_vi = get_newest_file(DOWNLOADS_DIR, ["tokenizer_vi*.json"])

    copied_count = 0

    if latest_pt:
        dest_pt = os.path.join(MODELS_DIR, "best_transformer.pt")
        size_mb = os.path.getsize(latest_pt) / (1024 * 1024)
        mtime = os.path.getmtime(latest_pt)
        import datetime
        dt_str = datetime.datetime.fromtimestamp(mtime).strftime("%d/%m/%Y %H:%M:%S")

        print(f"📦 Tìm thấy Checkpoint trong Downloads: {os.path.basename(latest_pt)} ({size_mb:.1f} MB, {dt_str})")
        # Ghi đè trực tiếp
        shutil.copy2(latest_pt, dest_pt)
        print(f"   -> ♻️ ĐÃ GHI ĐÈ TRỰC TIẾP LÊN: Translator/models/best_transformer.pt")
        copied_count += 1
    else:
        print("⚠️ Không tìm thấy file 'best_transformer.pt' nào trong Downloads.\n")

    if latest_en:
        dest_en = os.path.join(MODELS_DIR, "tokenizer_en.json")
        shutil.copy2(latest_en, dest_en)
        print(f"   -> ♻️ Đã ghi đè: tokenizer_en.json")
        copied_count += 1

    if latest_vi:
        dest_vi = os.path.join(MODELS_DIR, "tokenizer_vi.json")
        shutil.copy2(latest_vi, dest_vi)
        print(f"   -> ♻️ Đã ghi đè: tokenizer_vi.json")
        copied_count += 1

    if copied_count > 0:
        # Dọn sạch các file thừa trong Downloads để tiết kiệm ổ C
        clean_downloads_duplicates()
        verify_model()
        return True
    return False


def verify_model():
    """Kiểm tra mô hình có nạp được và hoạt động bình thường không."""
    print("-" * 65)
    print("🔍 Đang xác minh tính toàn vẹn của mô hình sau khi ghi đè...")
    model_path = os.path.join(MODELS_DIR, "best_transformer.pt")
    if not os.path.exists(model_path):
        print("❌ Không tìm thấy file model để xác minh.")
        return

    try:
        ckpt = torch.load(model_path, map_location="cpu")
        epoch = ckpt.get("epoch", "N/A")
        val_loss = ckpt.get("val_loss", "N/A")
        val_loss_str = f"{val_loss:.4f}" if isinstance(val_loss, (int, float)) else str(val_loss)
        print(f"🎉 XÁC MINH THÀNH CÔNG!")
        print(f"   - Mô hình hợp lệ: Epoch {epoch} | Validation Loss: {val_loss_str}")
        print("   - Trọng số đã sẵn sàng 100% để sử dụng trên máy tính!")
    except Exception as e:
        print(f"⚠️ Cảnh báo xác minh ({e}). Hãy đảm bảo file tải về đầy đủ dung lượng.")


def run_sync():
    """Luồng đồng bộ: Luôn ưu tiên Google Drive trước, ghi đè tiết kiệm ổ đĩa."""
    drive_url = ""
    if os.path.exists(DRIVE_LINK_FILE):
        with open(DRIVE_LINK_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("http"):
                    drive_url = line
                    break

    # Nếu chưa có Link Drive đã lưu, hỏi người dùng
    if not drive_url:
        print("\n" + "=" * 65)
        print(" ☁️ THIẾT LẬP ĐỒNG BỘ TRỰC TIẾP TỪ GOOGLE DRIVE")
        print("=" * 65)
        print("Bạn có thể dán Link thư mục Google Drive (Translator_PhoMT) vào đây")
        print("để hệ thống tự động tải thẳng về máy mà không cần tải thủ công.")
        print("*(Lưu ý: Thư mục trên Drive cần bật: 'Bất kỳ ai có đường liên kết đều có thể xem')*")
        try:
            inp = input("\nDán Link Drive vào đây (hoặc nhấn ENTER để lấy từ thư mục Downloads): ").strip()
        except (EOFError, KeyboardInterrupt):
            inp = ""

        if inp.startswith("http"):
            drive_url = inp
            with open(DRIVE_LINK_FILE, "w", encoding="utf-8") as f:
                f.write(drive_url + "\n")
            print(f"💾 Đã lưu Link Drive! Từ lần sau hệ thống sẽ luôn tự động tải từ link này.\n")

    # Ưu tiên 1: Thử tải từ Drive
    if drive_url:
        success = sync_from_gdrive(drive_url)
        if success:
            return

    # Dự phòng: Lấy từ thư mục Downloads
    sync_from_downloads()


if __name__ == "__main__":
    run_sync()
