# Real-time ASL Word Recognition to Text

Đây là một dự án nhận dạng ngôn ngữ ký hiệu Mỹ (ASL) theo thời gian thực, tập trung vào nhận dạng **word-level gloss** từ video webcam. Project được xây dựng trên kiến trúc **I3D (Inflated 3D ConvNet)** và có khả năng chuyển các gloss đã nhận dạng thành câu tiếng Anh tự nhiên nhờ **Google Gemini Flash API**.

## 🚀 Các tính năng chính

* **Nhận dạng ASL theo thời gian thực** từ webcam.
* **Hiển thị gloss trực tiếp** ngay trên video.
* **Chuyển gloss thành câu tiếng Anh hoàn chỉnh** với Gemini Flash.
* **Tuỳ chỉnh tham số**: CLIP\_LEN, STRIDE, voting bag size, threshold,...
* **Hỗ trợ tuỳ chọn background segmentation** bằng MediaPipe.

## 🔗 Chi tiết

🔗 **Repo GitHub:** [ASL-Interpreter](https://github.com/phatle0106/ASL-Interpreter)
