# PYTHON-AND-MACHINE-LEARNING-2025

## Giới thiệu ngắn

Khóa học **PYTHON-AND-MACHINE-LEARNING-2025** là một khóa học thực hành kết hợp nền tảng lập trình Python, toán học cần thiết và các phương pháp học máy & học sâu phổ biến. Mục tiêu giúp học viên xây dựng được pipeline từ tiền xử lý dữ liệu đến huấn luyện, đánh giá và triển khai các mô hình ML/DL cơ bản cho các bài toán phân loại, hồi quy, phân cụm, thị giác máy tính và sinh dữ liệu.

## Mục tiêu & Chuẩn đầu ra (Learning Outcomes)

Sau khi hoàn thành khóa học, học viên sẽ có thể:

1. **Sử dụng thành thạo Python và thư viện khoa học dữ liệu**: cài đặt môi trường, sử dụng Jupyter/Colab, làm việc với NumPy, Pandas, Matplotlib, Scikit-learn, TensorFlow/Keras.
2. **Áp dụng toán học nền tảng cho ML**: hiểu và vận dụng các khái niệm cơ bản trong đại số tuyến tính, giải tích, xác suất để phân tích mô hình ML (ví dụ PCA, đạo hàm, gradient).
3. **Triển khai quy trình ML hoàn chỉnh**: tiền xử lý dữ liệu, feature engineering, chia tập huấn luyện/kiểm tra, lựa chọn mô hình và đánh giá (accuracy, precision, recall, F1, R², RMSE, confusion matrix).
4. **Xây dựng và huấn luyện mô hình ML cổ điển**: Linear/Logistic Regression, KNN, K-Means, SVM, Decision Tree, Random Forest.
5. **Hiểu và triển khai các mô hình Deep Learning cơ bản**: mạng nơ-ron (MLP), CNN cho phân loại ảnh và object detection (ví dụ YOLO sử dụng mô hình pretrained), GANs cơ bản.
6. **Làm quen với Reinforcement Learning cơ bản**: MDP, Q-Learning, áp dụng cho game đơn giản (ví dụ qua PyGame).
7. **Đọc hiểu và trình bày kết quả thực nghiệm**: trực quan hoá, viết báo cáo ngắn, chuẩn bị demo/ứng dụng nhỏ.

## Tổng quan syllabus

Syllabus chi tiết đã được upload kèm theo repo và sẽ là kim chỉ nam cho các buổi học (biểu đồ thời lượng, nội dung từng buổi, người phụ trách). Nội dung chính bao gồm ba phần lớn:

* **Phần 1: Nền tảng Python và Toán học**

  * Cài môi trường, Python cơ bản, cấu trúc dữ liệu, NumPy, Pandas, các thư viện thường dùng.
* **Phần 2: Machine Learning**

  * Tiền xử lý dữ liệu, Linear/Logistic Regression, K-Means, KNN, SVM, Decision Tree & Random Forest, PCA, Reinforcement Learning cơ bản.
* **Phần 3: Deep Learning**

  * Giới thiệu Deep Learning, mạng nơ-ron cơ bản, CNN cho phân loại ảnh và object detection, Generative Adversarial Networks.

> Syllabus đầy đủ và chi tiết đã được dùng để soạn README này. (Xem file syllabus đã upload trong repository.)

## Lịch & Cấu trúc buổi học (tóm tắt)

* **Buổi 0 (Ở nhà)**: Cài đặt môi trường (Anaconda, Jupyter, Colab, VSCode/PyCharm).
* **Buổi 1–4**: Python cơ bản, toán nền tảng, cấu trúc dữ liệu, NumPy/Pandas.
* **Buổi 5–12**: Các thuật toán ML cơ bản, tiền xử lý, feature engineering, PCA, RL.
* **Buổi 13–16**: Deep Learning — MLP, CNN, Object Detection, GANs.

> Chi tiết từng buổi, thời lượng và người phụ trách được ghi trong syllabus dưới đây:

### Syllabus — PYTHON-AND-MACHINE-LEARNING-2025 (Linked)

> Phiên bản: mỗi ô **Chủ đề chính** là một link đến thư mục/buổi học tương ứng. Bạn có thể đổi đường dẫn (relative path) tuỳ cấu trúc repo.

| Buổi | Thời lượng | Chủ đề chính (click để mở folder buổi)                                          | Nội dung giảng dạy chi tiết                                                   | Hoạt động trên lớp & Thực hành                                |      TA đảm nhận     | Kiến thức bổ sung |
| ---: | :--------: | :------------------------------------------------------------------------------ | :---------------------------------------------------------------------------- | :------------------------------------------------------------ | :------------------: | :---------------- |
|    0 |    Ở nhà   | [Cài môi trường Python](lessons/00-cai-moi-truong/)                             | Cài đặt Python & môi trường (Anaconda, Jupyter); dùng Colab, VSCode, PyCharm. | Hướng dẫn cài; video hướng dẫn; kiểm tra môi trường.          |                      |                   |
|    1 |   2–2.5h   | [Giới thiệu Python & Lập trình cơ bản](lessons/01-gioi-thieu-python/)           | Biến, kiểu dữ liệu; điều kiện; vòng lặp; input/output.                        | Viết chương trình đơn giản; bài tập biến & vòng lặp.          |   M. Huy, V. Thịnh   |                   |
|    2 |   2–2.5h   | [Cơ sở Toán cho ML](lessons/02-toan-cho-ml/)                                    | Giải tích, Đại số tuyến tính, Xác suất ứng dụng vào ML.                       | Bài tập nhỏ về ma trận, đạo hàm, xác suất.                    |   T. Thịnh, M. Huy   |                   |
|    3 |   2–2.5h   | [Cấu trúc dữ liệu & Hàm trong Python](lessons/03-struct-data/)                  | List, Tuple, Dict, Set; hàm, tham số, phạm vi biến.                           | Viết hàm xử lý dữ liệu nhỏ.                                   |   V. Thịnh, Ng. Tín  |                   |
|    4 |   2–2.5h   | [Thư viện thường dùng trong Python](lessons/04-thu-vien/)                       | NumPy, Pandas, Matplotlib, Sklearn, OpenCV, librosa, tokenizer.               | Thực hành NumPy & Pandas với dataset mẫu.                     |   T. Thịnh, Ng. Tín  |                   |
|    5 |   2–2.5h   | [Giới thiệu ML & Data Preprocessing](lessons/05-ml-preprocessing/)              | Định nghĩa ML; chuẩn hoá, imputation, encoding, feature engineering.          | Làm sạch dataset; so sánh hiệu quả tiền xử lý.                |   V. Thịnh, D. Tân   |                   |
|    6 |   2–2.5h   | [Linear Regression](lessons/06-linear-regression/)                              | Hồi quy tuyến tính; MSE; Gradient Descent; R², RMSE.                          | Xây dựng mô hình với Scikit-learn; trực quan hoá.             |   T. Thịnh, M. Huy   |                   |
|    7 |   2–2.5h   | [Logistic Regression](lessons/07-logistic-regression/)                          | Hàm sigmoid; threshold; accuracy, precision, recall, F1.                      | Huấn luyện mô hình phân loại; đánh giá bằng confusion matrix. |    M. Huy, D. Tân    |                   |
|    8 |   2–2.5h   | [K-Means & KNN](lessons/08-kmeans-knn/)                                         | K-Means, Elbow Method; KNN cơ bản.                                            | Phân cụm & phân loại với ví dụ thực tế.                       |  T. Thịnh, V. Thịnh  |                   |
|    9 |   2–2.5h   | [Support Vector Machine (SVM)](lessons/09-svm/)                                 | Hyperplane, margin, kernel trick (RBF, poly).                                 | So sánh kernel; huấn luyện SVM.                               | D. Tân, Trung & Hiếu |                   |
|   10 |   2–2.5h   | [Decision Tree & Random Forest](lessons/10-tree-rf/)                            | Gini, Entropy, Overfitting; Ensemble.                                         | Xây dựng & so sánh Decision Tree và Random Forest.            |   T. Thịnh, Ng. Tín  |                   |
|   11 |   2–2.5h   | [Principal Component Analysis (PCA)](lessons/11-pca/)                           | Lý thuyết PCA, giảm chiều, ứng dụng.                                          | Áp dụng PCA lên dataset thực tế.                              |   V. Thịnh, D. Tân   |                   |
|   12 |   2–2.5h   | [Reinforcement Learning (RL)](lessons/12-rl/)                                   | MDP, Value Iteration, Q-Learning, PyGame.                                     | Áp dụng RL cho game đơn giản.                                 |   D. Tân, T. Thịnh   |                   |
|   13 |   2–2.5h   | [Giới thiệu Deep Learning & Mạng Nơ-ron](lessons/13-dl-intro/)                  | NN, forward/backprop, loss, optimizer; TensorFlow/Keras.                      | Xây mạng cơ bản bằng Keras; training & evaluation.            |   M. Huy, V. Thịnh   |                   |
|   14 |   2–2.5h   | [CNNs cho phân loại ảnh](lessons/14-cnn-classification/)                        | Conv layer, pooling, filters, stride, padding.                                | Xây & huấn luyện CNN đơn giản.                                |  V. Thịnh, T. Thịnh  |                   |
|   15 |   2–2.5h   | [CNNs cho Phát hiện đối tượng (Object Detection)](lessons/15-object-detection/) | Object Detection; dùng model pretrained (YOLO).                               | Triển khai YOLO demo; fine-tune.                              |    Duy Tân, M. Huy   |                   |
|   16 |   2–2.5h   | [Generative Adversarial Network (GANs)](lessons/16-gans/)                       | Generator vs Discriminator; ứng dụng: tạo ảnh, phục chế, style transfer.      | Xây GAN cơ bản; demo tạo ảnh.                                 |   M. Huy, V. Thịnh   |                   |

---

**Ghi chú:**

* Các đường link trên là **relative paths** giả định: `lessons/<slug>/` — bạn có thể đổi thành `notebooks/`, `slides/` hoặc đường dẫn trên GitHub.
* Tài liệu gốc (syllabus) đã được dùng để tạo phiên bản này.

## Hình thức đánh giá

* **Bài tập về nhà & thực hành nhỏ**: Mỗi buổi có bài tập thực hành (code notebooks).
* **Mini project cuối khóa**: Một dự án nhỏ (phân loại/nhận dạng/phát hiện/thu thập & tiền xử lý dữ liệu) trình bày demo và báo cáo kỹ thuật.
* **Tham gia & trình bày trên lớp**: Thảo luận, code review, demo.

## Yêu cầu trước khi tham gia

* Biết cơ bản về lập trình (không bắt buộc Python trước đó nhưng sẽ thuận lợi).
* Máy có thể cài Python/Anaconda; kết nối Internet để dùng Colab nếu cần.

## Tài nguyên & Tài liệu tham khảo

* Giới thiệu & cài đặt: video hướng dẫn PyCharm/Anaconda (link được ghi trong syllabus).
* Thư viện: NumPy, Pandas, Scikit-learn, TensorFlow/Keras, Matplotlib, Seaborn, OpenCV, PyGame.
* Tài liệu học thêm: sách, blog, và tài liệu chính thức của các thư viện.

## Cấu trúc đề xuất cho repository

```
README.md
lessons/
  homework/
  slide-xx-<name-lesson>.pdf
  record-name-lesson (link youtube)
  name-lesson.md
final-project/          
  name-project/
    name-project.md
    slide-name-project.pdf
    video-name-project (link youtube)
```

## Hướng dẫn nhanh cho giảng viên và trợ giảng

* Sử dụng notebooks cho phần thực hành; chuẩn bị dữ liệu mẫu cho mỗi buổi.
* Chuẩn bị rubrics đánh giá cho mini-project (tính đúng đắn, chất lượng code, báo cáo, demo).

## Liên hệ

* Các thông tin liên hệ, phân công giảng dạy và trợ giảng được ghi trong file syllabus và trang quản lý khóa học.

---

> *Profile, Porfolio, ... của các TA tại đây.*