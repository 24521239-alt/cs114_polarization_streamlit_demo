# CS114 Polarization Streamlit Demo

Đây là web demo cho đồ án môn Máy học CS114 về bài toán phát hiện phân cực quan điểm trong văn bản tiếng Anh.

Ứng dụng cho phép người dùng nhập một đoạn văn bản, chọn mô hình dự đoán và xem xác suất của từng nhãn. Web được xây dựng bằng Streamlit. Các mô hình sử dụng đặc trưng TF-IDF và đã được tối ưu siêu tham số bằng Bayesian Optimization.

## Chức năng chính

- Nhập một đoạn văn bản tiếng Anh để phân tích.
- Chọn một trong ba mô hình:
  - Logistic Regression + TF-IDF
  - XGBoost + TF-IDF
  - SVM + TF-IDF
- Hiển thị nhãn dự đoán:
  - Nhãn 0: Non-Polarized
  - Nhãn 1: Polarized
- Hiển thị xác suất dự đoán của từng nhãn.
- Hiển thị biểu đồ trực quan hóa xác suất.
- Hiển thị văn bản trước và sau bước tiền xử lý.
- Hiển thị các chỉ số thực nghiệm của mô hình đang được chọn.

## Cấu trúc thư mục

- `app.py`: File chính để chạy giao diện Streamlit.
- `requirements.txt`: Danh sách thư viện cần cài đặt.
- `train_models.py`: File dùng để huấn luyện lại mô hình khi cần.
- `data/`: Chứa dữ liệu dùng cho huấn luyện và kiểm thử.
- `models/`: Chứa các mô hình đã huấn luyện và file metadata.
- `src/`: Chứa các module xử lý chính như tiền xử lý văn bản, load mô hình và dự đoán.

## Cài đặt môi trường

Tạo môi trường ảo:

    python -m venv .venv

Kích hoạt môi trường ảo trên Windows:

    .venv\Scripts\activate

Cài đặt các thư viện cần thiết:

    pip install -r requirements.txt

## Chạy ứng dụng

Chạy lệnh sau trong thư mục project:

    streamlit run app.py

Sau đó mở đường dẫn localhost được Streamlit hiển thị trên terminal. Thông thường đường dẫn là:

    http://localhost:8501

## Cách sử dụng demo

Bước 1: Chọn mô hình ở thanh bên trái.

Bước 2: Nhập một đoạn văn bản tiếng Anh hoặc chọn câu mẫu có sẵn.

Bước 3: Nhấn nút phân tích.

Bước 4: Xem kết quả dự đoán, xác suất từng nhãn và phần văn bản sau tiền xử lý.

Khi tự nhập văn bản, nên nhập câu có đủ ngữ cảnh. Nếu câu quá ngắn hoặc chỉ gồm một vài từ riêng lẻ, mô hình có thể khó xác định rõ mức độ phân cực.

## Tiền xử lý dữ liệu

Văn bản đầu vào được xử lý trước khi đưa vào mô hình. Các bước chính gồm:

- Chuyển văn bản về chữ thường.
- Mở rộng một số dạng viết tắt trong tiếng Anh.
- Loại bỏ URL, mention, hashtag và ký tự không cần thiết.
- Loại bỏ một số stopwords nhưng giữ lại các từ phủ định quan trọng.
- Chuẩn hóa văn bản để phù hợp với đặc trưng TF-IDF.

## Huấn luyện lại mô hình

Nếu muốn huấn luyện lại mô hình, chạy lệnh:

    python train_models.py

Sau khi chạy xong, các file mô hình mới sẽ được lưu trong thư mục `models/`.

## Lưu ý

Demo hiện chỉ xử lý văn bản tiếng Anh theo phạm vi dữ liệu của đồ án. Kết quả dự đoán có thể chưa ổn định với các câu quá ngắn, thiếu ngữ cảnh hoặc có cách diễn đạt mỉa mai phức tạp.
