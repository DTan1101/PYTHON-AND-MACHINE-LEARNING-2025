# Lesson 15 — Object Detection (OD)

## Mục tiêu bài học

* Hiểu bài toán Object Detection: trả lời câu hỏi “What?” (đó là đối tượng nào) và “Where?” (nó ở đâu).
* Nắm được khái niệm **bounding box**, **anchor box**, phát hiện đa tỉ lệ (multiscale detection).
* Nắm các thuật toán chính trong OD:

  * **Viola–Jones** (cascade classifiers, Haar features).
  * **R-CNN series** (R-CNN, Fast R-CNN, Faster R-CNN, Mask R-CNN).
  * **YOLO series** (YOLOv1 → YOLOv12, các cải tiến kiến trúc).
  * **DETR (Detection Transformer)** – phát hiện dựa trên transformer.
* Hiểu ưu/nhược điểm của từng phương pháp (two-stage vs one-stage, anchor-based vs anchor-free).
* Thực hành cài đặt/thử nghiệm: từ bounding box cơ bản, tới inference YOLOv8/YOLOv10.

## Yêu cầu trước buổi học

* Nắm vững CNN cơ bản (convolution, pooling, feature map).
* Hiểu cơ bản về bài toán image classification.
* Kiến thức Python, PyTorch/TF/Keras cơ bản để chạy model inference.

## Tóm tắt nội dung (ngắn gọn)

* **Bounding Box Representation**:
  \$(x\_1,y\_1,x\_2,y\_2)\$ ↔ \$(c\_x,c\_y,w,h)\$, dùng để mô tả vị trí đối tượng.

* **Anchor Boxes & Multiscale Detection**:

  * Anchor box: hộp tham chiếu được sinh ra tại mỗi pixel của feature map.
  * Multiscale: sử dụng nhiều feature map (FPN, SSD) để phát hiện đối tượng ở nhiều kích thước.

* **Các phương pháp chính**:

  * **Viola–Jones**: Cascade of weak classifiers, Haar-like features, dùng nhiều cho face detection.
  * **R-CNN series**:

    * R-CNN: selective search + CNN + SVM.
    * Fast R-CNN: end-to-end training, ROI pooling.
    * Faster R-CNN: Region Proposal Network (RPN) + detector.
    * Mask R-CNN: thêm nhánh segmentation mask.
  * **YOLO series**: one-stage detection, real-time, phát triển đến YOLOv12 với nhiều cải tiến backbone, head, và attention.
  * **DETR / RT-DETR**: dùng transformer encoder-decoder, không cần anchor box, truy vấn đối tượng (object queries) → bounding box + label.

* **Loss Function & Training**:

  * Classification loss + bounding box regression loss (IoU, GIoU, CIoU).
  * Với DETR: Hungarian matching giữa prediction và ground truth.

## Lưu ý thực hành (tools & môi trường)

* Frameworks: **PyTorch**, **Ultralytics YOLO**, **Transformers** (cho DETR).
* Dataset minh họa: COCO (80 classes), Pascal VOC, custom dataset nhỏ (label bằng Roboflow/LabelImg).
* Thực hành inference: chạy YOLOv8/YOLOv10 với ảnh hoặc webcam.
* Visualization: vẽ bounding box bằng `matplotlib` hoặc `cv2`.
* Lưu ý khi fine-tune: chọn lr phù hợp, augment hợp lý (flip, mosaic), batch size vừa phải.

## Mẹo giảng dạy & demo

* Bắt đầu với ví dụ đơn giản: vẽ bounding box thủ công, giải thích (x1,y1,x2,y2) → (cx,cy,w,h).
* Demo Viola–Jones bằng OpenCV face detection.
* So sánh inference tốc độ: Faster R-CNN vs YOLOv8 trên cùng ảnh.
* Cho học viên thử chỉnh confidence threshold, NMS IoU threshold để thấy ảnh hưởng.
* Giới thiệu DETR như một cách nhìn mới: detection = set prediction problem.

## Bài tập / Homework

1. **Bounding Box Coding**

   * Cài đặt hàm chuyển đổi corner-to-center và center-to-corner (giống trong slide).
   * Tạo một ảnh nhỏ, vẽ nhiều box khác nhau.

2. **YOLOv8 Inference**

   * Chạy YOLOv8 pretrained trên 5 ảnh tuỳ chọn, hiển thị kết quả.
   * So sánh output khi thay confidence threshold (0.25 → 0.5 → 0.75).

3. **So sánh mô hình**

   * Viết ngắn gọn ưu/nhược điểm của Faster R-CNN vs YOLOv8 vs DETR.

4. **Nâng cao (tuỳ chọn)**

   * Fine-tune YOLOv8 trên một dataset nhỏ (ví dụ: detect 2-3 class custom).
   * Thử inference với RT-DETR (PyTorch) và so sánh tốc độ.

## Tài liệu tham khảo & Links

* **Slides buổi học (gốc):** `slides/15-object-detection/slide-15-object-detection.pdf`
* **Dive into Deep Learning – SSD & R-CNN:**

  * [SSD Chapter](https://d2l.ai/chapter_computer-vision/ssd.html)
  * [R-CNN Chapter](https://d2l.ai/chapter_computer-vision/rcnn.html)
* **YOLO Guide:** [Roboflow — Guide to YOLO Models](https://blog.roboflow.com/guide-to-yolo-models/)
* **Faster R-CNN paper:** [arXiv:1506.01497](https://arxiv.org/abs/1506.01497)
* **DETR paper:** [DETRs Beat YOLOs on Real-time Object Detection (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhao_DETRs_Beat_YOLOs_on_Real-time_Object_Detection_CVPR_2024_paper.pdf)
