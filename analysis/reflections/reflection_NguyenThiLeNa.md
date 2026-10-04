# Individual Reflection — Lab 18: Production RAG

**Học viên:** Nguyen Thi Le Na
**Khóa:** K4 - Track 3A
**Ngày hoàn thành:** 04/10/2026

---

## Phần 1: Mapping bài giảng (Lecture Mapping)

| Lecture Concept | Module | Hàm/lớp trong code | Quan sát |
|---|---|---|---|
| Semantic, hierarchical và structure-aware chunking | M1 | `chunk_semantic()`, `chunk_hierarchical()`, `chunk_structure_aware()` trong `src/m1_chunking.py` | Semantic gom câu dựa trên độ tương đồng; hierarchical giữ liên kết cha-con qua `parent_id`; structure-aware giữ nội dung dưới tiêu đề Markdown. Cấu hình parent/child hiện là 2048/256 ký tự. |
| Vietnamese BM25, dense retrieval và rank fusion | M2 | `segment_vietnamese()`, `BM25Search`, `DenseSearch`, `reciprocal_rank_fusion()`, `HybridSearch.search()` trong `src/m2_search.py` | BM25 bắt từ khóa chính xác, dense search tìm tương đồng ngữ nghĩa; RRF hợp nhất thứ hạng mà không cần chuẩn hóa thang điểm hai bộ tìm kiếm. |
| Cross-Encoder reranking | M3 | `CrossEncoderReranker.rerank()` trong `src/m3_rerank.py` | Mô hình chấm trực tiếp từng cặp query-document để sắp xếp lại các ứng viên trước khi đưa context vào LLM. Cross-Encoder đã nạp thành công từ cache trong lần chạy tích hợp. |
| RAGAS và chẩn đoán lỗi | M4 | `evaluate_ragas()`, `failure_analysis()`, `save_report()` trong `src/m4_eval.py` | Lần đánh giá chi tiết đạt Faithfulness 0.6292, Answer Relevancy 0.4975, Context Precision 0.8917 và Context Recall 0.8333. Production tăng Relevancy so với baseline nhưng giảm ba metric còn lại; 5 ca thấp nhất được đối chiếu với answer/context/ground truth trong failure analysis. |
| Contextual prepend, HyQA và metadata enrichment | M5 | `_enrich_single_call()`, `contextual_prepend()`, `generate_hypothesis_questions()`, `extract_metadata()` trong `src/m5_enrichment.py` | Chế độ combined gộp bốn tác vụ vào một yêu cầu cho mỗi chunk; khi thiếu key hoặc request lỗi, các helper cục bộ giữ cho bước tiền xử lý tiếp tục được. `tests/test_m5.py` đã passed 10/10. |

## Phần 2: Khó khăn và cách xử lý (Challenges & Debugging)

- **Docker không kết nối được Linux engine:** Docker CLI ban đầu báo không tìm thấy `npipe:////./pipe/dockerDesktopLinuxEngine`. Code M2 chuyển sang Qdrant in-memory khi không kết nối được server, nên pipeline vẫn index và chạy được nhưng dữ liệu Qdrant không tồn tại sau khi tiến trình kết thúc. Khi triển khai lâu dài cần khởi động Docker Desktop và xác nhận Qdrant hoạt động.
- **Chạy nhầm Python hệ thống:** `python main.py` dừng với `ModuleNotFoundError: No module named 'pypdf'`. Các dependency đã có trong virtual environment; dùng `.venv\Scripts\python.exe` thì đọc tài liệu và chạy được pipeline.
- **Model và quyền mạng:** Lần đầu Hugging Face trả `WinError 10013` khi tải `BAAI/bge-m3`; cache còn shard dở. Sau khi mạng được cấp cho lần chạy đã được người dùng cho phép, model tải/nạp thành công. Cross-Encoder `BAAI/bge-reranker-v2-m3` cũng nạp thành công từ cache.
- **Quyền gửi dữ liệu ra ngoài:** Auto-review ban đầu từ chối lần chạy vì M5, sinh đáp án và RAGAS gửi nội dung chunk/context/câu hỏi/đáp án tới OpenRouter. Sau khi xác nhận rõ việc gửi dữ liệu, toàn pipeline mới được chạy.
- **PDF scan không có text layer:** `BCTC.pdf` và Nghị định 13 bị bỏ qua; muốn truy vấn nội dung hai tài liệu này cần OCR trước khi indexing.
- **Kiểm thử từng phần:** `pytest tests/test_m5.py -v` chạy thành công với 10/10 test. Kết quả này xác nhận helper và cấu trúc chunk M5; nó không xác nhận tải model hoặc chạy toàn pipeline.
- **Thiếu dữ liệu chi tiết trong report:** Bản `save_report()` ban đầu chỉ lưu aggregate và danh sách failure, không lưu answer/context từng câu. Đã sửa để thêm `per_question`, sau đó chạy lại Production nhằm lưu evidence thật và phân tích Bottom-5.

## Phần 3: Kế hoạch áp dụng cho project cá nhân (Action Plan)

### Project đề xuất: Trợ lý hỏi đáp quy định và sổ tay nhân sự

#### 1. Hiện trạng và rủi ro cần giải quyết

Tài liệu quy định thường có tiêu đề, bảng, danh sách và nhiều phiên bản theo năm. Tìm kiếm theo embedding đơn thuần có thể bỏ sót mã/số liệu, còn lấy nhầm phiên bản cũ có thể dẫn đến câu trả lời sai. Tài liệu nội bộ cũng cần được bảo vệ khi chọn mô hình và dịch vụ đánh giá.

#### 2. Kế hoạch cải tiến

1. **Chunking:** Dùng structure-aware để giữ tiêu đề, bảng và danh sách; dùng hierarchical để tìm trên child chunk nhưng trả parent context. Gắn `source`, `section`, `effective_date` và `version` làm metadata.
2. **Retrieval:** Kết hợp BM25 với dense retrieval bằng RRF để hỗ trợ cả mã văn bản/số liệu chính xác và câu hỏi diễn đạt tự nhiên.
3. **Reranking:** Chấm lại top 20 ứng viên bằng Cross-Encoder, sau đó chỉ đưa một số context liên quan nhất vào LLM.
4. **Evaluation:** Tạo bộ câu hỏi chuẩn có câu hỏi theo phiên bản và số liệu; theo dõi đủ bốn metric RAGAS, đồng thời duyệt thủ công các lỗi nghiêm trọng trước mỗi lần phát hành.
5. **Enrichment:** Thử summary, HyQA và contextual prepend trên bản sao dữ liệu đã được phép xử lý. Với tài liệu nhạy cảm, ưu tiên model chạy nội bộ hoặc chỉ dùng dịch vụ ngoài sau khi có phê duyệt về dữ liệu.

#### 3. Timeline triển khai

- **Tuần 1:** Kiểm kê tài liệu, xác định quyền truy cập và metadata phiên bản/ngày hiệu lực; tạo tập câu hỏi cùng đáp án chuẩn.
- **Tuần 2:** Dựng ingestion với chunking phân cấp, Hybrid Search và lọc theo phiên bản.
- **Tuần 3:** Thêm reranker, chạy benchmark retrieval và rà soát thủ công các câu trả lời sai.
- **Tuần 4:** So sánh trước/sau bằng RAGAS, kiểm tra quyền riêng tư và triển khai pilot có logging, phản hồi người dùng.
