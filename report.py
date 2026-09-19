import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ==========================================
# 1. KHỞI TẠO DỮ LIỆU TỪ KẾT QUẢ THỰC NGHIỆM
# ==========================================

labels = ["Low", "Medium", "High"]

# Confusion Matrix từ log thực tế
cm_baseline = np.array([[43, 639, 1044], [27, 668, 2335], [30, 673, 3596]])

cm_lightgbm = np.array([[899, 495, 332], [1008, 1055, 967], [988, 1267, 2044]])

# Chuẩn hóa ma trận theo hàng (Recall per cell - tính theo %)
cm_baseline_norm = cm_baseline.astype("float") / cm_baseline.sum(axis=1)[:, np.newaxis] * 100
cm_lightgbm_norm = cm_lightgbm.astype("float") / cm_lightgbm.sum(axis=1)[:, np.newaxis] * 100

# ==========================================
# 2. VẼ HEATMAP CONFUSION MATRIX SO SÁNH
# ==========================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
sns.set_theme(style="white", font_scale=1.1)

# Heatmap Baseline
sns.heatmap(
    cm_baseline_norm,
    annot=True,
    fmt=".1f",
    cmap="Blues",
    cbar=False,
    xticklabels=[f"Pred_{l}" for l in labels],
    yticklabels=[f"True_{l}" for l in labels],
    ax=axes[0],
    linewidths=1,
    linecolor="white",
)
axes[0].set_title(
    "Baseline: Matrix Factorization",
    fontsize=13,
    fontweight="bold",
    pad=12,
)
axes[0].set_xlabel("Vùng dự đoán", fontweight="bold")
axes[0].set_ylabel("Vùng thực tế", fontweight="bold")

# Thêm giá trị đếm thô (raw counts) vào heatmap
for i in range(3):
  for j in range(3):
    axes[0].text(
        j + 0.5,
        i + 0.75,
        f"(n={cm_baseline[i, j]})",
        ha="center",
        va="center",
        fontsize=9,
        color="gray",
    )

# Heatmap LightGBM
sns.heatmap(
    cm_lightgbm_norm,
    annot=True,
    fmt=".1f",
    cmap="Greens",
    cbar=False,
    xticklabels=[f"Pred_{l}" for l in labels],
    yticklabels=[f"True_{l}" for l in labels],
    ax=axes[1],
    linewidths=1,
    linecolor="white",
)
axes[1].set_title(
    "Advanced: LightGBM Classifier (Balanced)",
    fontsize=13,
    fontweight="bold",
    pad=12,
)
axes[1].set_xlabel("Vùng dự đoán", fontweight="bold")

for i in range(3):
  for j in range(3):
    axes[1].text(
        j + 0.5,
        i + 0.75,
        f"(n={cm_lightgbm[i, j]})",
        ha="center",
        va="center",
        fontsize=9,
        color="gray",
    )

plt.suptitle(
    "So sánh Ma trận nhầm lẫn chuẩn hóa (%) giữa Baseline và LightGBM",
    fontsize=15,
    fontweight="bold",
    y=1.03,
)
plt.tight_layout()
plt.savefig("confusion_matrix_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

# ==========================================
# 3. XUẤT BẢNG SO SÁNH METRICS CHO BÁO CÁO
# ==========================================

# Bảng 1: So sánh tổng quan (Overall Summary)
summary_df = pd.DataFrame(
    {
        "Chỉ số đánh giá": [
            "Độ chính xác thô (Bucket Accuracy)",
            "Độ chính xác cân bằng (Balanced Accuracy)",
            "Macro F1-Score",
            "Số đặc trưng đầu vào (Used Features)",
            "Recall vùng Low (1.0 - 3.0 sao)",
            "Recall vùng High (4.0 - 5.0 sao)",
        ],
        "Baseline (MF)": [
            "47.56%",
            "36.06%",
            "0.3172",
            "2 (User, Item)",
            "2.49%",
            "83.65%",
        ],
        "LightGBM Classifier": [
            "44.15%",
            "44.82%",
            "0.4283",
            "16 (Features + Aggregations)",
            "52.09%",
            "47.55%",
        ],
        "Mức độ cải thiện / Nhận xét": [
            "-3.41% (Giảm ảo do không còn đoán dồn High)",
            "+8.76% (Độ chính xác công bằng tăng mạnh)",
            "+35.03% (Chất lượng phân loại vượt trội)",
            "+14 đặc trưng thống kê & ngữ cảnh",
            "+49.60% (Cứu vãn thành công vùng đánh giá thấp)",
            "Giảm mức độ tự tin thái quá, hạn chế đoán bừa",
        ],
    }
)

# Bảng 2: So sánh chi tiết theo từng vùng (Per-bucket Metrics)
detailed_df = pd.DataFrame(
    {
        "Vùng điểm (Bucket)": [
            "Low [1.0 - 3.0]",
            "Low [1.0 - 3.0]",
            "Medium (3.0 - 4.0]",
            "Medium (3.0 - 4.0]",
            "High (4.0 - 5.0]",
            "High (4.0 - 5.0]",
        ],
        "Mô hình": [
            "MF Baseline",
            "LightGBM",
            "MF Baseline",
            "LightGBM",
            "MF Baseline",
            "LightGBM",
        ],
        "Precision": [0.43, 0.31, 0.34, 0.37, 0.52, 0.61],
        "Recall": [0.02, 0.52, 0.22, 0.35, 0.84, 0.48],
        "F1-Score": [0.05, 0.39, 0.27, 0.36, 0.64, 0.53],
        "Số lượng mẫu (Support)": [1726, 1726, 3030, 3030, 4299, 4299],
    }
)

# In kết quả ra màn hình console
print("\n" + "=" * 80)
print("BẢNG 1: TỔNG HỢP SO SÁNH HIỆU NĂNG TỔNG QUAN")
print("=" * 80)
print(summary_df.to_string(index=False))

print("\n" + "=" * 80)
print("BẢNG 2: CHI TIẾT TỪNG PHÂN VÙNG ĐÁNH GIÁ (PER-CLASS PERFORMANCE)")
print("=" * 80)
print(detailed_df.to_string(index=False))

# Xuất ra file CSV để import thẳng vào Word/Excel/LaTeX
summary_df.to_csv("summary_metrics_comparison.csv", index=False, encoding="utf-8-sig")
detailed_df.to_csv("detailed_bucket_metrics.csv", index=False, encoding="utf-8-sig")
print("\n--> Đã lưu biểu đồ vào: 'confusion_matrix_comparison.png'")
print("--> Đã xuất 2 file báo cáo vào: 'summary_metrics_comparison.csv' và 'detailed_bucket_metrics.csv'")