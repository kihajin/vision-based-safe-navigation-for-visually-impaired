# Hệ thống hỗ trợ điều hướng an toàn cho người khiếm thị

> Khóa luận: **Hệ thống hỗ trợ điều hướng an toàn cho người khiếm thị dựa trên thị giác máy tính**  
> Tác giả: Trần Kiều Hạnh  

## Giới thiệu

Dự án nghiên cứu hệ thống phân tích video RGB ngoài trời để hỗ trợ người khiếm thị nhận biết vật cản và lựa chọn hướng di chuyển an toàn. Pipeline kết hợp YOLOv11 để phát hiện đối tượng, SegFormer-B5 để phân đoạn ngữ nghĩa và Depth Anything V2 để ước lượng độ sâu tương đối. Từ các tín hiệu này, hệ thống đánh giá mức nguy hiểm, xác định vùng có thể đi và sinh cảnh báo điều hướng bằng tiếng Việt; notebook cũng tạo video kết quả có hình minh họa và âm thanh.

Ba cấu hình được so sánh là `YOLO-only`, `YOLO + Depth` và `YOLO + Depth + Segmentation`. Việc đánh giá tập trung vào chất lượng cảnh báo, false warning, độ ổn định theo thời gian và độ chính xác hướng điều hướng.

## Cấu trúc dự án

```text
KLTN/
├── Code_final/
│   ├── foundation.ipynb       # Khảo sát mask SANPO/ADE20K và dữ liệu depth
│   ├── mydata.ipynb           # Sinh segmentation mask và depth map
│   └── test_program.ipynb     # Pipeline phát hiện, cảnh báo, video và đánh giá
├── ground_truth*.json         # Nhãn hướng điều hướng dùng cho đánh giá
└── labels_coco_vi.json        # Ánh xạ tên lớp COCO sang tiếng Việt
```

Các thư mục `dataset/`, `my_data_ver2/` và `output/` không được lưu trong repo GitHub. Tải chúng từ [Google Drive](https://drive.google.com/drive/folders/1TZ_Z8pymearis42j1NoIqxCJXs4xaC6P?usp=sharing). Nếu không mở được, cần được cấp quyền xem thư mục.

### Các thư mục trên Drive

- `dataset/`: các tập SANPO, từ `data1/` đến `data10/`. Trong mỗi tập, `left/` có thể chứa `video_frames/` (ảnh RGB), `depth_maps/` (depth nén `.float16.gz`), `segmentation_masks/` (mask PNG) và `zed_depth_maps/` (depth từ cảm biến ZED, nếu có). Một số tập không có đủ mọi loại dữ liệu. Ở dự án này, mình chỉ dùng data1, data2, data5.
- `my_data_ver2/`: dữ liệu video tự thu thập đã tiền xử lý, gồm frame RGB, depth map và segmentation mask đồng bộ. Đây là đầu vào mặc định của pipeline trong `test_program.ipynb`.
- `output/`: kết quả chạy mẫu và nơi notebook ghi kết quả mới. Chứa video `final_*.mp4`, dự đoán `predictions*.json`, cảnh báo `messages*.json` và dữ liệu phát hiện YOLO thô.

Các tập con trong `dataset/` có thể thiếu `segmentation_masks` hoặc một số loại depth map. Kiểm tra dữ liệu sau khi tải trước khi chạy pipeline.

## Môi trường và phụ thuộc

Các notebook hiện được viết cho **Google Colab**: có lệnh `!pip`, magic command `%cd`, mount Google Drive và đường dẫn `/content/drive/MyDrive/KLTN`. Chúng chưa được đóng gói thành chương trình chạy trực tiếp trên máy local và repo hiện chưa có `requirements.txt`.

Các thư viện chính được dùng gồm:

- `ultralytics` (YOLOv11), `torch`, `torchvision`
- `transformers` (SegFormer), `huggingface_hub`
- `opencv-python`, `numpy`, `tqdm`, `matplotlib`, `Pillow`
- `gTTS`, `pydub`
- `ffmpeg` để ghép audio với video

Trong Google Colab, có thể cài các gói notebook cần bằng cell sau:

```python
%pip install ultralytics transformers gTTS pydub
!apt-get update -qq && apt-get install -y ffmpeg
```

Colab thường đã có `torch`, `torchvision`, `numpy` và OpenCV. Nên bật GPU để tăng tốc suy luận. Việc tải model từ Hugging Face, tải YOLO weights và dùng gTTS cần kết nối Internet. `mydata.ipynb` còn clone mã nguồn Depth Anything V2 và tải checkpoint `depth_anything_v2_vitl.pth`.

## Chuẩn bị dữ liệu

1. Tải mã nguồn từ GitHub hoặc sao chép repo vào Google Drive tại `MyDrive/KLTN/`.
2. Tải `dataset/`, `my_data_ver2/` và `output/` từ [Google Drive](https://drive.google.com/drive/folders/1TZ_Z8pymearis42j1NoIqxCJXs4xaC6P?usp=sharing). Nếu Drive chứa file ZIP, giải nén chúng vào `MyDrive/KLTN/` để các thư mục nằm cạnh `Code_final/`.
3. Kết nối Google Drive trong Colab. Nếu đặt dự án ở vị trí khác, cập nhật các đường dẫn tuyệt đối trong notebook.

Dữ liệu đầu vào của `test_program.ipynb` cần có cấu trúc sau:

```text
my_data_ver2/
├── video_frames/           # Ảnh RGB theo frame
├── depth_maps/             # File .float16.gz tương ứng từng frame
└── segmentation_masks/     # Mask PNG tương ứng từng frame
```

Các notebook tiền xử lý trên SANPO cũng dùng dữ liệu dạng:

```text
dataset/data1/left/
├── video_frames/
├── depth_maps/
└── segmentation_masks/
```

- File depth phải theo định dạng notebook đọc: dữ liệu `float16` nén gzip, trong đó hai giá trị đầu là chiều cao và chiều rộng, phần còn lại là ma trận depth.
- Đảm bảo tên file giữa frame, depth map và segmentation mask khớp nhau. Mask cần dùng đúng bộ nhãn khai báo trong `ThesisConfig.DATASET_TYPE` và `SEGMENTATION_ID2LABEL`.

## Chạy notebook

Mở `Code_final` trong Google Colab hoặc mở từng notebook bằng Jupyter, sau khi đã đưa dữ liệu vào Drive và cập nhật đường dẫn.

1. **`foundation.ipynb`**: notebook khảo sát/kiểm tra mask SANPO, ánh xạ nhãn ADE20K sang SANPO, đọc depth và trực quan hóa vùng đi được. Chủ yếu phục vụ phát triển và kiểm tra dữ liệu.
2. **`mydata.ipynb`**: notebook tiền xử lý. Có cell sinh mask bằng SegFormer-B5 và sinh depth map bằng Depth Anything V2. Các cell mẫu hiện trỏ tới `dataset/data1/left`; cập nhật `INPUT_DIR` và `OUTPUT_DIR` nếu chạy trên tập khác.
3. **`test_program.ipynb`**: notebook chính. Chạy các cell từ trên xuống để cấu hình dữ liệu, thực hiện phân tích, tạo cảnh báo, dựng video/audio và chạy phần đánh giá. Cấu hình trong notebook hiện chọn dữ liệu `my_data_ver2` và xuất kết quả vào `output`.

Notebook chính dùng model YOLO `yolo11n.pt`; Ultralytics có thể tự tải weights khi chạy lần đầu. Các mô hình SegFormer và Depth Anything V2 cũng cần được tải/cấu hình ở các cell tiền xử lý tương ứng. Không có checkpoint YOLO riêng được lưu trong repo.

## Kết quả

Sau khi chạy notebook chính, kết quả được ghi vào `output/` trong Google Drive, gồm:

- `final_*.mp4`: video đầu ra; một số bản có cảnh báo âm thanh.
- `predictions*.json`: dự đoán hướng điều hướng theo frame.
- `messages*.json`: cảnh báo và thông tin tương ứng theo đoạn/frame.
- `raw_yolo_detections.json`: kết quả phát hiện đối tượng thô.

Các file `ground_truth*.json` lưu hướng điều hướng tham chiếu; `ground_truth_my_data_ver2.json` là nhãn bổ sung được tạo từ annotation và được lưu trong repo. `test_program.ipynb` có các cell đánh giá dự đoán so với ground truth và thống kê warning.

## Phạm vi và hạn chế

- Đây là bộ notebook nghiên cứu/thực nghiệm, chưa có giao diện ứng dụng hoặc lệnh CLI độc lập.
- Dữ liệu RGB, depth map và segmentation mask phải được chuẩn bị đồng bộ; một vài thư mục trong repo không có đủ các loại dữ liệu này.
- Các đường dẫn hiện phụ thuộc Google Drive và cần sửa khi dùng vị trí hoặc dataset khác.
- Kết quả depth là độ sâu tương đối phục vụ phân tích trong pipeline, không nên diễn giải như khoảng cách vật lý đã hiệu chuẩn.
- Đánh giá cảnh báo sai trong khóa luận có phần kiểm tra thủ công trên các warning segment; kết quả không thay thế đánh giá an toàn ngoài thực địa.