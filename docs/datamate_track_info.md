# Zalo AI Challenge 2026 — Track DataMate
Nguồn: https://challenge.zalo.ai/portal/datamate (+ trang chủ https://challenge.zalo.ai/), đọc ngày 27/10/2026.

> Tab **Data** cần đăng nhập nên nội dung ở mục 8 do người dùng dán vào; các tab khác đọc công khai.

## 1. Đề bài
**DataMate – Talking To Your Data Through Agentic AI.** Xây AI agent trả lời câu hỏi tiếng Việt đời thường trực tiếp trên dữ liệu bảng (CSV, Excel, SQLite) của tiểu thương/doanh nghiệp nhỏ: Excel nhiều sheet, tên sản phẩm không nhất quán, cách hỏi văn nói, logic nghiệp vụ (giá vốn, công nợ quá hạn…). Agent phải trả lời đúng **hoặc từ chối khi dữ liệu không đủ để trả lời**. Đúng được điểm, sai bị trừ, im lặng không mất điểm.

Ví dụ câu hỏi: "Tháng này bán chạy nhất mặt hàng gì?", "Tuần rồi bán được bao nhiêu ly trà sữa?", "Còn ai nợ tiền chưa trả?", "Thuốc nào sắp hết hạn trong kho?"

### Input
- Data package của 1 doanh nghiệp: 1+ file (.csv, .xlsx, .sqlite) + `README.md` mô tả doanh nghiệp.
- Câu hỏi tiếng Việt đời thường.
- `as_of_date`: "hôm nay" để quy đổi thời gian tương đối ("tháng này", "tuần rồi", "cùng kỳ năm ngoái").
- Rulebook + lịch âm–dương dùng chung cho mọi business của split (xem tab Data).

### Output
Một answer thuộc: value (number, percent, string, date, boolean, list, table), `unanswerable`, hoặc `abstain`.

Ví dụ `questions.jsonl`:
```json
{"id": "q_06de33fa21cc", "business_id": "biz_118", "question": "sp bán dc nhiều nhất tuần rồi là gì", "as_of_date": "2026-03-04"}
```
`submission.jsonl`:
```json
{"id": "q_06de33fa21cc", "type": "string", "answer": "Xi măng Hà Tiên PCB40 bao 50kg"}
```

## 2. Đánh giá (Evaluation)
Mỗi câu: **correct / wrong / abstain**. `abstain` (hoặc thiếu answer) = abstain. Câu khác chỉ đúng nếu **type trùng reference** và value khớp; không có điểm từng phần.

| Type | Nội dung | Ví dụ | Quy tắc khớp |
|---|---|---|---|
| number | số (VND, đếm, số lượng) | 48250000 | Khớp chính xác. Reference là số nguyên hoặc làm tròn 2 chữ số thập phân |
| percent | phần trăm | -12.45 | Khớp sau làm tròn 2 chữ số; đúng dấu |
| string | tên (sản phẩm, khách, chi nhánh, thứ…) | "Sting dâu 330ml" | Chuẩn hoá cả hai phía rồi so bằng: (1) Unicode NFC; (2) lowercase; (3) ký tự không phải chữ/số/khoảng trắng → thay bằng khoảng trắng; (4) trim và gộp khoảng trắng. **Giữ dấu tiếng Việt.** Reference liệt kê mọi dạng chấp nhận: mọi cách viết tên xuất hiện trong dữ liệu của business đó + mã (vd `sting dâu 330ml`, `sp0042`) |
| date | ngày ISO | "2026-08-15", "2025-09" | Khớp chính xác; yyyy-mm-dd, hỏi theo tháng thì yyyy-mm |
| boolean | true/false | true | Khớp chính xác |
| list | danh sách có thứ tự tên/số | ["Cà phê sữa","Bạc xỉu","Trà đào"] | Khớp theo vị trí; thứ tự theo câu hỏi hoặc mặc định của rulebook. Phần tử khớp theo rule string/number |
| table | bảng nhỏ | {"columns":[...],"rows":[[...],...]} | Cột khớp theo tên đã chuẩn hoá (nếu tên lệch thì theo vị trí); hàng khớp như tập; ô tên theo rule string, ô số theo rule number |
| unanswerable | dữ liệu không đủ để trả lời | — | Đúng khi và chỉ khi reference là unanswerable |
| abstain | agent chủ động không trả lời | — | Luôn tính abstain bất kể reference |

**Ranking:** dùng tiêu chí trong câu hỏi, nếu không có thì mặc định của rulebook; giảm dần chỉ trên các đơn hợp lệ; bằng nhau thì xếp theo code tăng dần. Câu có tie ở ranh giới top-k (hoặc rank 1 với "… nhất") bị loại khỏi test set. Ít hơn k ứng viên thì đáp án là tất cả.

**Score – Penalized Accuracy (PA):**
PA = (1/N) · Σ sᵢ, với sᵢ = +1 nếu đúng, 0 nếu abstain, −1 nếu sai. Mọi câu trọng số bằng nhau. **Trả lời bất kỳ đáp án nào cho câu unanswerable đều tính sai.** Hoà thì xét tổng thời gian inference khi organizers re-run, rồi đến nộp sớm hơn.

## 3. Rule (ràng buộc kỹ thuật)
Hai vòng: Round 1 Public test (nộp `submission.jsonl`, điểm chỉ tham khảo) → nộp final solution dạng Docker image → Round 2 Private test (chạy solution đã đóng băng, nộp `submission.jsonl`). **Xếp hạng cuối = kết quả organizers re-run Docker image.**

1. Đội/điều kiện theo regulations chung của Zalo AI Challenge 2026.
2. **Model:** bất kỳ model có weights công bố **trước 1/10/2026** với licence cho phép; được fine-tune. Tổng tham số của mọi model dùng khi inference (LLM, embedding, reranker, classifier; kể cả không load cùng lúc) **≤ 15B**, MoE tính toàn bộ tham số. Cho phép mọi quantization. Phải khai báo model trong README của solution.
3. **Không dịch vụ ngoài:** inference chạy hoàn toàn offline trong Docker, không gọi API LLM thương mại, không download, **network tắt**.
4. **Không human-in-the-loop.** Cấm hard-code đáp án hoặc logic theo question/business id.
5. Agent được sinh và chạy code trong container; thư mục data mount **read-only**, file tạm ghi chỗ khác.
6. Data: được dùng training data cung cấp, data tự synthesize, dataset/model công khai. **Cấm dùng public/private test để train**, hoặc thu thập đáp án test bằng bất kỳ cách nào.
7. **Máy chấm:** 2 × RTX 5090 (32GB mỗi card, 64GB tổng), 512GB RAM. Xử lý toàn bộ private test trong **6 giờ wall-clock** gồm cả load model. Blackwell cần CUDA ≥ 12.8 và PyTorch/vLLM tương thích; Docker guideline chỉ định base image.
8. **Reproducibility:** |PA(nộp) − PA(re-run)| ≤ 0.03. Nếu lệch hơn, organizers xem log trong image, có thể chạy lại, có thể hỏi thêm; không giải thích được thì **loại**. Nếu re-run quá giờ/dừng sớm, chỉ tính các câu đã ghi vào `/result/submission.jsonl`, phần còn lại là abstain.

Giao thức chấm: organizers mount test set vào container tại `/data` và lấy `/result/submission.jsonl`.

## 4. Submission Guidance & lịch
**I. Public Test Round (26/10 – 16/11)**
- Nộp file JSONL qua mục Submission; **tối đa 5 lần/ngày**, reset cuối ngày. Điểm hiện trên public leaderboard.
- **16/11:** nộp full solution (training code, inference code, data) dạng **Docker package** qua form "Solution Submission". Vì file lớn: kèm checksum + link Google Drive public. Tối đa **2 lần nộp**, chỉ chấm lần cuối. Hạn 23:59 (GMT+7).

**II. Private Test Round (20/11 – 23/11)**
- 9:00 sáng 20/11: release private test set. Đội chạy solution đã nộp, nộp JSONL, hạn 23:59 (GMT+7) 23/11. Leaderboard private xếp theo thời gian nộp.
- Organizers kiểm chứng kết quả, review source, phát hiện gian lận. Top 2 công bố 2/12.

### Timeline (trang chủ)
| Ngày | Mốc |
|---|---|
| 26/10 | Bắt đầu; phát dev kit, data (train & public test), evaluation software |
| 16/11 | Nộp solution (code train/test, model), 23:59 GMT+7 |
| 20/11 | Release private test, 9:00 GMT+7 |
| 23/11 | Nộp kết quả cuối |
| 2/12 | Công bố finalists (top 2 mỗi track) |
| 12/12 | Công bố xếp hạng cuối |

## 5. Giải thưởng & đội
- Tổng giải **USD 12,000** tiền mặt + gói quà tặng.
- Nhất **USD 3,500**, Nhì **USD 2,500**; Top 5 mỗi track nhận ZAIC gift package.
- Đội tối đa **5 thành viên**, mỗi người chỉ 1 đội; chốt thành viên trước ~1 tuần so với hạn nộp; **từ 17/11 khoá thêm/bớt thành viên**.
- Nhận thưởng: nộp hồ sơ trong 2 tuần kể từ khi được báo, thưởng trao trong 30 ngày sau khi đủ hồ sơ.
- Hỗ trợ kỹ thuật: aichallenge@zalo.me
- Mở cho người Việt yêu AI; từng tham gia các năm trước vẫn được.
- Advisors: GS Nguyễn Lê Minh (JAIST), PGS.TS Quản Thành Thơ (HCMUT), Nguyễn Minh Tứ (CTO Zalo), TS Nguyễn Trường Sơn, Nguyễn Cảnh Hùng, Trần Văn Hòa (Zalo AI).

## 6. Public leaderboard (chụp 27/10/2026, top 10)
1. SakiTenma 1.000 · 2. Fuho 1.000 · 3. Cơm Tấm Sài Gòn 1.000 · 4. <BK/> 0.998 · 5. NNC 0.988 · 6. FatSheeDuk 0.984 · 7. Ánh Dương 0.980 · 8. BN_VN3S 0.980 · 9. WWW 0.975 · 10. Odylytics 0.972. Hạng 20 ≈ 0.804. Lưu ý: leaderboard public chỉ để tham khảo, điểm thật theo private test.

## 7. Track còn lại
LifeMate – Agentic AI biến giọng nói thành hành động (smart home, ngân hàng, gọi xe, email…), chấm theo trạng thái hệ thống cuối. Không thuộc phạm vi file này.

## 8. Data (tab cần đăng nhập — nội dung do người dùng dán)

**Loại business:** tạp hoá, quán cà phê/trà sữa, nhà hàng, nhà thuốc, VLXD, tiệm sửa xe máy, cửa hàng mẹ & bé… **Private test có loại business không xuất hiện ở train/public test.** Mỗi business 12–30 tháng hoạt động, có mùa vụ (Tết, Trung thu, lễ, ngày lương, cuối tuần), từ vài trăm đến ~300,000 dòng.

**Cấu trúc gói:**
```
<split>/
├── data/<business_id>/  *.csv | *.xlsx | *.sqlite  + README.md
├── rulebook.md          # luật mặc định + từ vựng (MỖI split có bản riêng, agent phải đọc bản của split đang trả lời)
├── calendar.csv         # lịch âm–dương (Tết)
├── questions.jsonl      # {"id","business_id","question","as_of_date"}
└── answers.jsonl        # chỉ có ở train: {"id","type","answer","alias":[...]}; câu unanswerable có "answer": null
```
`alias` = các cách viết khác của tên trong dữ liệu + mã sản phẩm (không cần nộp). Chất lượng README mỗi business khác nhau: có khi đủ data dictionary, có khi gần như trống.

**Luật mặc định điển hình (rulebook):** doanh thu = tổng thành tiền dòng sau giảm giá dòng (không trừ voucher cấp đơn), không VAT, không phí giao hàng, loại đơn huỷ và đơn có hàng trả; "bán chạy" = theo số lượng; tuần = thứ Hai–Chủ nhật; "tháng này" = tháng chứa as-of date, tới hết ngày đó; giá vốn mặc định = bình quân gia quyền theo lô nhập; cửa sổ Tết = 23 tháng Chạp → mùng 10 (âm lịch).

**Format `submission.jsonl`:**
```json
{"id":"q_01234","type":"string","answer":"Sting dâu 330ml"}
{"id":"q_01235","type":"number","answer":48250000}
{"id":"q_01236","type":"percent","answer":-12.45}
{"id":"q_01237","type":"date","answer":"2026-08-15"}
{"id":"q_01238","type":"boolean","answer":true}
{"id":"q_01239","type":"list","answer":["Cà phê sữa","Bạc xỉu","Trà đào"]}
{"id":"q_01240","type":"table","answer":{"columns":["khach_hang","so_tien_no"],"rows":[["Chị Lan",3200000],["Anh Tú",1500000]]}}
{"id":"q_01241","type":"unanswerable","reason":"(tuỳ chọn)"}
{"id":"q_01242","type":"abstain"}
```
Câu thiếu = abstain.

**Docker image:** chứa source (train + inference), model weights, `predict.sh` đọc test từ `/data` và ghi `/result/submission.jsonl`, kèm README. Nên ghi log từng câu vào `/result/logs/` (tối thiểu: question id, final answer, elapsed time; tốt nhất thêm code sinh ra + output thực thi) để organizers đối chiếu khi re-run (Rule 8).

### Dữ liệu đã tải (D:\AI vietnam\zalo-ai\data)
| Split | Business | Câu hỏi | Có answers | Dung lượng |
|---|---|---|---|---|
| train | 30 | 3,000 | có (3,000) | 186 MB |
| public_test | 14 | 1,000 | không | 105 MB |
