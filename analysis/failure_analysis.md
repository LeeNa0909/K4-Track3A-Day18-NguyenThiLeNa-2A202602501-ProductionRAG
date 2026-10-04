# Failure Analysis — Lab 18: Production RAG

**Học viên:** Nguyen Thi Le Na
**Khóa:** K4 - Track 3A

---

## Kết quả tích hợp

`main.py` đã chạy xong với Python trong `.venv`. Baseline và Production đều đánh giá 20 câu. Hai PDF scan (`BCTC.pdf` và Nghị định 13) bị bỏ qua vì không có text layer/OCR; pipeline dùng 26 tài liệu còn lại để tạo 100 chunk Production.

Bảng dưới đây dùng baseline hiện có và Production mới nhất trong `reports/ragas_report.json`:

| Metric | Naive Baseline | Production | Δ |
|---|---:|---:|---:|
| Faithfulness | 0.7583 | 0.6292 | -0.1292 |
| Answer Relevancy | 0.4182 | 0.4975 | +0.0793 |
| Context Precision | 0.9250 | 0.8917 | -0.0333 |
| Context Recall | 0.9250 | 0.8333 | -0.0917 |

Production cải thiện Answer Relevancy, nhưng ba metric còn lại giảm; Faithfulness và Answer Relevancy vẫn dưới 0.70. Lần chạy `main.py` đầu tiên in Faithfulness 0.5875, Answer Relevancy 0.5095, Context Precision 0.8917, Context Recall 0.8250. Sau khi sửa report để lưu per-question evidence, mình chạy lại riêng Production; vì vậy số trong JSON là kết quả Production của lần chạy chi tiết sau. Đầu ra LLM/RAGAS có thể dao động giữa các lần chạy.

`reports/ragas_report.json` hiện có 20 phần tử `per_question`, gồm `question`, `answer`, `contexts`, `ground_truth` và bốn metric; trường `failures` xếp hạng 10 câu thấp nhất. Lần chạy hoàn chỉnh có 100 request enrichment M5, 20 lần sinh đáp án Production và 80 phép chấm RAGAS.

## Bottom-5 Failures

### #1 — Phân loại thông tin lương

- **Question:** Thông tin lương thuộc cấp độ phân loại dữ liệu nào?
- **Expected:** Thông tin lương là dữ liệu Bí mật (cấp 3); phải mã hóa khi truyền và hạn chế truy cập theo need-to-know.
- **Got:** “Không tìm thấy.”
- **Điểm:** trung bình 0.2708; worst metric `faithfulness` = 0.0. `answer_relevancy` = 0.0, `context_precision` = 0.5833, `context_recall` = 0.5.
- **Câu trả lời có đúng không?** Không đầy đủ so với đáp án chuẩn. Từ chối trả lời tránh bịa nhưng bỏ sót quy định có trong tài liệu.
- **Context có đáp án không?** Context đưa ra các cấp “Bí mật” và cách xử lý, nhưng không truy xuất đoạn `ky_luong.md` nối trực tiếp “thông tin lương” với cấp Bí mật. Context chứa khái niệm chung, chưa có bằng chứng trực tiếp cho thực thể được hỏi.
- **Cần viết lại query không?** Không; câu hỏi rõ. Có thể bổ sung query expansion nội bộ cho các từ “lương”, “phiếu lương”, “dữ liệu Bí mật”.
- **Module cần sửa:** M2 trước tiên; bổ trợ M1/M5. Tăng khả năng ưu tiên chunk từ `ky_luong.md`, giữ quan hệ thuật ngữ–phân loại trong chunk, và gắn metadata loại dữ liệu.

### #2 — Phép năm và khung lương theo phiên bản

- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** Theo chính sách nghỉ phép 2024: 15 ngày cơ bản + 3 ngày thâm niên = 18 ngày. Khung lương Senior là 20–35 triệu VNĐ/tháng.
- **Got:** “Không tìm thấy.”
- **Điểm:** trung bình 0.3333; worst metric `faithfulness` = 0.0. `answer_relevancy` = 0.0, `context_precision` = 0.8333, `context_recall` = 0.5.
- **Câu trả lời có đúng không?** Sai/thiếu: context đầu đã nói rõ 18 ngày, còn phần lương không có trong context được đưa vào.
- **Context có đáp án không?** Có đáp án phép năm 18 ngày. Context còn lẫn bản 2023 (12 ngày và quy tắc cộng phép khác), nhưng không có bảng lương 2024.
- **Cần viết lại query không?** Câu hỏi tự nó rõ, nhưng gồm hai phép tra cứu. Có thể tách retrieval thành câu hỏi về phép năm 2024 và câu hỏi về khung lương Senior.
- **Module cần sửa:** M2/M3 để lấy đủ hai tài liệu và ưu tiên phiên bản hiệu lực mới; M1 để giữ nguyên parent context/bảng; M4 để trả lời phần đã có bằng chứng và nêu rõ phần còn thiếu thay vì từ chối toàn bộ.

### #3 — Phê duyệt mua laptop 30 triệu

- **Question:** Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?
- **Expected:** Giám đốc phòng ban (Director) phê duyệt vì đơn hàng thuộc khoảng 5–50 triệu; cần phòng CNTT xác nhận cấu hình kỹ thuật; đính kèm ít nhất 3 báo giá vì giá trị trên 10 triệu.
- **Got:** Cần xác nhận cấu hình từ CNTT; người duyệt thuộc cấp tương ứng, “có thể là Giám đốc hoặc cấp cao hơn”, nhưng câu trả lời nói tài liệu không nêu cụ thể.
- **Điểm:** trung bình 0.3958; worst metric `answer_relevancy` = 0.0. `faithfulness` = 0.25, `context_precision` ≈ 1.0, `context_recall` = 0.3333.
- **Câu trả lời có đúng không?** Đúng một phần: nêu đúng yêu cầu CNTT nhưng không chốt Director và bỏ sót 3 báo giá.
- **Context có đáp án không?** Có xác nhận cấu hình CNTT; các context được gửi không chứa ngưỡng 5–50 triệu hoặc yêu cầu 3 báo giá, dù cả hai có trong `mua_sam.md`.
- **Cần viết lại query không?** Không; query cụ thể. Có thể phân rã truy vấn thành mức phê duyệt, giấy tờ cần nộp và điều kiện kỹ thuật.
- **Module cần sửa:** M1/M2 chính. Bảng thẩm quyền và danh sách bước bị tách khỏi chunk có xác nhận CNTT; pipeline hiện chỉ index child chunks và chưa mở rộng child hit thành parent context. Tăng recall cho các dòng bảng và truy xuất đủ các điều khoản trên 10 triệu.

### #4 — Ứng phó malware

- **Question:** Khi phát hiện malware trên máy, nhân viên có nên tự xử lý không?
- **Expected:** Không. Không tự xử lý; báo sự cố cho CNTT trong vòng 1 giờ qua helpdesk hoặc hotline.
- **Got:** “Không tìm thấy.”
- **Điểm:** trung bình 0.4167; worst metric `faithfulness` = 0.0. `answer_relevancy` = 0.0, `context_precision` ≈ 1.0, `context_recall` = 0.6667.
- **Câu trả lời có đúng không?** Không. Context trả lời trực tiếp rằng tuyệt đối không tự ý xử lý malware.
- **Context có đáp án không?** Có bằng chứng cho câu hỏi yes/no và hotline; đoạn trích không nhất thiết chứa đầy đủ thời hạn 1 giờ, nhưng thời hạn không làm thay đổi đáp án chính.
- **Cần viết lại query không?** Không; câu hỏi ngắn và rõ.
- **Module cần sửa:** M4/`run_query()` trong `src/pipeline.py`. Retriever đã tìm được bằng chứng nhưng LLM vẫn từ chối. Sửa prompt để buộc trả lời trực tiếp từ câu có bằng chứng, đặt `temperature=0`, và thêm kiểm tra đầu ra “Không tìm thấy” khi context thực sự có câu trả lời.

### #5 — Lương thử việc Junior

- **Question:** Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu?
- **Expected:** Mức Junior cao nhất 20 triệu; lương thử việc bằng 85%, tức 17 triệu VNĐ/tháng.
- **Got:** “Không tìm thấy.”
- **Điểm:** trung bình 0.5000; worst metric `faithfulness` = 0.0. `answer_relevancy` = 0.0, `context_precision` ≈ 1.0, `context_recall` = 1.0.
- **Câu trả lời có đúng không?** Không; không trả lời phép tính dù các dữ kiện đều có.
- **Context có đáp án không?** Có: `bang_luong_2024.md` có khung Junior 12–20 triệu và `thu_viec.md` quy định thử việc nhận 85%. Mô hình cần tính 0.85 × 20 triệu = 17 triệu.
- **Cần viết lại query không?** Không; câu hỏi đủ rõ.
- **Module cần sửa:** M4/`run_query()` trong `src/pipeline.py`. Lỗi ở tổng hợp và tính toán câu trả lời, không phải retrieval. Prompt nên yêu cầu tính toán từng bước ngắn gọn và nêu phép tính dựa trên context.

## Case Study: Câu hỏi lương thử việc Junior

1. **Output đúng?** Không; trả lời “Không tìm thấy” trong khi đáp án suy ra được là 17 triệu.
2. **Context đúng?** Có; retrieved contexts chứa mức lương Junior tối đa 20 triệu và tỷ lệ thử việc 85%.
3. **Query rewrite?** Không cần. Thất bại nằm ở bước suy luận/tổng hợp.
4. **Fix ở đâu?** M4, prompt sinh đáp án trong `run_query()`: yêu cầu dùng đủ dữ kiện, thực hiện phép nhân, trả lời trực tiếp và chỉ abstain khi thiếu dữ kiện thật.

**Nếu có thêm 1 giờ, sẽ ưu tiên:** sửa prompt sinh đáp án và đặt temperature về 0; sau đó bổ sung parent-context expansion cho hierarchical chunks, lọc metadata theo ngày hiệu lực để tránh trộn bản 2023/2024, và OCR hai PDF scan.
