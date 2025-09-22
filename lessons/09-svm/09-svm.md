# Lesson 09 — Support Vector Machines (SVM)

## Mục tiêu bài học

* Hiểu ý tưởng cơ bản của SVM: tìm siêu phẳng phân tách tốt nhất (max margin).
* Nắm được khác biệt giữa hard-margin và soft-margin; khái niệm slack variable.
* Biết cơ sở bài toán đối ngẫu (dual), KKT và ý nghĩa support vectors.
* Hiểu kernel trick và cách áp dụng SVM cho dữ liệu không tách tuyến tính.
* Thực hành triển khai SVM bằng scikit-learn và tune hyperparameters.

## Yêu cầu trước buổi học

* Làm quen Python, NumPy, Pandas và scikit-learn cơ bản.
* Chuẩn hoá dữ liệu (StandardScaler) trước khi dùng SVM.

## Tóm tắt nội dung (ngắn gọn)

* **Hyperplane:**

  $w^T x + b = 0$

* **Khoảng cách từ điểm $x_0$ đến hyperplane:**

  $\dfrac{|w^T x_0 + b|}{\lVert w \rVert}$

* **Hard-margin (tuyến tính tách được):**

  Minimize $\tfrac{1}{2}\lVert w\rVert^2$ subject to $y_n(w^T x_n + b)\ge 1$.

* **Soft-margin (cho phép sai):**

  Minimize $\tfrac{1}{2}\lVert w\rVert^2 + C\sum_n \xi_n$ subject to $y_n(w^T x_n + b)\ge 1-\xi_n,\ \xi_n\ge0$.

* **Hinge loss (tương đương biến thể không ràng buộc):**

  $L(y,f(x))=\max(0,\,1 - y\,f(x))$

* **Đối ngẫu & Kernel (tóm tắt):**

  Dual tối ưu hóa theo các hệ số $\lambda_n$:

  $\max_{\lambda}\, \sum_n \lambda_n - \tfrac{1}{2}\sum_{n,m}\lambda_n\lambda_m y_n y_m K(x_n,x_m)$

  với ràng buộc $0\le\lambda_n\le C$ và $\sum_n \lambda_n y_n = 0$.

  Kernel: $K(x,x')=\Phi(x)^T \Phi(x')$ (không cần xây $\Phi$ thực tế).

* **KKT (tóm tắt):** stationarity, primal & dual feasibility, complementary slackness — quyết định điểm hỗ trợ (support vectors).

## Lưu ý thực hành (scikit-learn)

* Thư viện: `from sklearn.svm import SVC` (đa số dùng SVC cho phân loại nhị phân/đa lớp).
* Thường cần `StandardScaler` trước khi huấn luyện.
* Kernel phổ biến: `linear`, `rbf`, `poly` — với `rbf` tune `C` và `gamma`.
* Dùng `GridSearchCV`/`RandomizedSearchCV` để chọn `C`, `gamma`, `degree`.
* Support vectors: `clf.support_vectors_` để xem các vectơ hỗ trợ.

## Mẹo giảng dạy & demo

* Bắt đầu bằng trực quan 2D (đồ thị điểm, siêu phẳng, margin, support vectors).
* So sánh linear SVM và RBF SVM trên dữ liệu XOR/hình vòng để minh hoạ kernel.
* Cho sinh viên thử thay đổi `C` để thấy trade-off giữa margin và số lỗi.
* Thảo luận nhanh: vì sao dùng đối ngẫu? (kernel + giảm biến đổi dimension).

## Bài tập / Homework

* **Áp dụng SVM cho bài toán phân loại nấm (mushroom classification)** — chuẩn hoá, train/test split, thử ít nhất 2 kernel, báo cáo accuracy, precision, recall và số support vectors.

## Tài nguyên tham khảo

* Slide buổi học (bản gốc) và tài liệu tham khảo kèm theo trong slide.

