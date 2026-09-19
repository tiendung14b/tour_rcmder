# Tourism Recommender System 🌴

Dự án này xây dựng một hệ thống gợi ý địa điểm du lịch (Tourism Recommender System) sử dụng bộ dữ liệu thực tế: *"A tourism dataset from historical transaction for recommender systems"*. Hệ thống hỗ trợ dự đoán và phân loại đánh giá (rating) của người dùng đối với các điểm tham quan dựa trên bối cảnh chuyến đi (`Visiting Mode`) và đặc trưng địa điểm (`Item Features`).

## 🛠 Kiến trúc Hệ thống

Dự án thử nghiệm và so sánh hai phương pháp tiếp cận:

1. **MF Baseline (Hồi quy Matrix Factorization):**
   - Thuật toán tự xây dựng dựa trên `NumPy`.
   - Học phân rã ma trận sử dụng Stochastic Gradient Descent (SGD) với các tham số độ lệch (Global Bias, User Bias, Item Bias).
   - Đầu ra dự đoán điểm liên tục được giới hạn (clip) trong khoảng `[1.0, 5.0]`.

2. **Advanced Model (LightGBM Classifier):**
   - Chuyển đổi dữ liệu sang dạng bảng (Tabular Data) với tập hợp 17 Features.
   - Xử lý mất cân bằng dữ liệu bằng cách sử dụng `LGBMClassifier` với `class_weight='balanced'`.
   - Phân loại trực tiếp rating thành 3 vùng: **Low** (1-3 điểm), **Medium** (3-4 điểm), và **High** (4-5 điểm).

Cả hai mô hình đều được đánh giá trực tiếp dựa trên **Bucket Accuracy**, **Balanced Accuracy** và **Macro F1-Score**.

## 🚀 Hướng dẫn Cài đặt

### Yêu cầu môi trường
- Hệ điều hành: Windows, macOS hoặc Linux.
- Python `3.9+`

### Bước 1: Clone mã nguồn
Mở terminal hoặc command prompt và clone kho lưu trữ:

```bash
git clone https://github.com/tiendung14b/tour_rcmder.git
cd tour_rcmder
```

### Bước 2: Thiết lập thư mục dữ liệu
Đảm bảo bạn đã giải nén bộ dữ liệu du lịch gốc chứa các file Excel (`Transaction.xlsx`, `User.xlsx`, `Item.xlsx`, `City.xlsx`,...). 
> **Lưu ý:** Bạn cần cập nhật biến `data_dir` trong file `train.py` trỏ tới đúng đường dẫn chứa các file này trên máy tính của bạn trước khi chạy.
> ```python
> data_dir = r"Đường_dẫn_tới_thư_mục_chứa_data"
> ```

### Bước 3: Cài đặt thư viện phụ thuộc
Cài đặt các gói Python yêu cầu bằng `pip`:

```bash
pip install -r requirements.txt
```
*(Các thư viện chính bao gồm: pandas, numpy, scikit-learn, scipy, openpyxl, lightgbm).*

## 🏃 Hướng dẫn Sử dụng

Chạy file `train.py` để thực thi toàn bộ luồng xử lý: tải dữ liệu, tiền xử lý, huấn luyện cả 2 mô hình và in ra bảng đánh giá chi tiết trên tập kiểm thử (Test Set).

```bash
python train.py
```

### Kết quả đầu ra (Mẫu)
Script sẽ in ra bảng so sánh hiệu suất giữa 2 mô hình và 10 mẫu gợi ý ngẫu nhiên có chứa xác suất được dự đoán cho từng vùng (Prob_Low, Prob_Medium, Prob_High).

```text
Metrics             | MF Baseline | Advanced Classifier
--------------------|-------------|--------------------
Bucket Accuracy     |     47.56% |             43.79%
Balanced Accuracy   |     36.06% |             44.69%
Macro F1-Score      |    0.3172  |            0.4263

--- Top 10 Random Samples (LightGBM Classifier) ---
 User  Attraction  VisitMode True_Rating Pred_Rating True_Bucket Pred_Bucket Prob_Low Prob_Medium Prob_High    Result
79342         748          3        5.00        3.25        High         Low     0.36        0.34      0.29 Incorrect
...
```

## 📝 Giấy phép
Dự án được xây dựng cho mục đích nghiên cứu và học tập.
