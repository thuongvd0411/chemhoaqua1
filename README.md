# Hướng Dẫn Cài Đặt và Chạy Game Tương Tác Camera

Dự án này bao gồm hai trò chơi: "Chém Hoa Quả" (Fruit Ninja) và "Đập Chuột" (Whack-a-Mole), điều khiển bằng cử chỉ tay thông qua Webcam.

## Yêu Cầu Hệ Thống
- Python 3.8 trở lên.
- Webcam hoạt động tốt.

## Cài Đặt

Chúng tôi sử dụng `uv` để quản lý môi trường cho tốc độ nhanh nhất, nhưng bạn cũng có thể dùng `pip` truyền thống.

3. **Cách 3: Chạy tự động (Dễ nhất)**
   - Chỉ cần nhấp đúp vào file `run_game.bat`.
   - Sẽ tự động cài đặt mọi thứ và mở game (yêu cầu đã cài Python).

> [!IMPORTANT]
> **Lỗi thường gặp: "Python was not found..." hoặc không chạy được.**
> - Nguyên nhân: Windows có sẵn "shortcut" Python nhưng chưa cài thực sự.
> - Khắc phục:
>   1. Tải Python từ [python.org](https://www.python.org/downloads/).
>   2. Khi cài đặt, **BẮT BUỘC** tích chọn **"Add Python to PATH"**.
>   3. Tắt và mở lại ứng dụng/terminal này.

### Cách 1: Sử dụng `uv` (Khuyên dùng)
1. **Cài đặt uv** (nếu chưa có):
   ```bash
   pip install uv
   ```
   *hoặc trên Windows:*
   ```powershell
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. **Tạo môi trường ảo và cài thư viện**:
   Tại thư mục chứa file `app.py`:
   ```bash
   uv venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/Mac:
   source .venv/bin/activate
   
   uv pip install streamlit streamlit-webrtc opencv-python-headless mediapipe numpy av
   ```

### Cách 2: Sử dụng `pip` chuẩn
```bash
python -m venv .venv
# Kích hoạt môi trường (như trên)
pip install streamlit streamlit-webrtc opencv-python-headless mediapipe numpy av
```

## Chạy Ứng Dụng

Sau khi kích hoạt môi trường, chạy lệnh:

```bash
streamlit run app.py
```

Trình duyệt sẽ tự động mở địa chỉ (thường là `http://localhost:8501`).

## Hướng Dẫn Chơi

### 1. Chém Hoa Quả (Fruit Ninja)
- **Cách chơi**: Sử dụng ngón tay trỏ (Index Finger) đưa lên camera để điều khiển "lưỡi dao".
- **Mục tiêu**: Di chuyển ngón tay qua các trái cây bay lên để chém.
- **Tránh**: Bom (Bomb) và Gai (Spike).
- **Thắng**: Đạt 20 điểm.
- **Thua**: Hết 5 mạng.

### 2. Đập Chuột (Whack-a-Mole)
- **Cách chơi**: Sử dụng hai nắm tay (Wrists) để đập.
- **Mục tiêu**: Đưa nắm tay vào vị trí các lỗ chuột khi chuột xuất hiện.
- **Tránh**: Chuột đội mũ gai (Spike/Red).
- **Thắng**: Đạt 20 điểm.
- **Thua**: Hết 5 mạng.

## Lưu Ý
- Đảm bảo cấp quyền truy cập Camera cho trình duyệt.
- Nếu video bị lag hoặc không hiện, hãy thử tải lại trang (F5).
- Nên đứng cách camera khoảng 1-2 mét để MediaPipe nhận diện tốt nhất.
