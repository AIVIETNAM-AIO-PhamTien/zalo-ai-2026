# Kiến trúc tối ưu PA — bản hợp nhất (30/10/2026)

Bản này thay hướng "DataMate 2.0 thuần" trong `docs/v2/DATAMATE_2_PLAN.md`. Ý tưởng chính: dữ liệu, câu hỏi và đáp án
gần như chắc chắn được sinh từ một schema chuẩn và rulebook, rồi làm bẩn. Vì vậy hệ thống **đảo ngược bộ sinh**:
dữ liệu bẩn → schema chuẩn, câu hỏi → chương trình, rồi tính lại đúng như bộ sinh. Mục tiêu duy nhất là PA trên
business chưa gặp.

Cột **Nguồn** ghi rõ: `legacy` = code trước `724a6e4`, `v2` = code v2 đã có trong repo, `plan` = ý trong
`docs/v2/DATAMATE_2_PLAN.md` chưa có code, `mới` = chưa có ở đâu.

## 1. Nguyên tắc (sửa từ plan v2 §2)

1. Tối đa kỳ vọng điểm: trả lời khi P(đúng) đã hiệu chỉnh > 0.5 + biên. Kiểm chứng là **đặc trưng** cho bộ
   chọn, không phải cổng chặn bắt buộc.
2. Không hardcode theo id câu hỏi / id business; không giả định mọi shop có bảng bán hàng.
3. Không mất dữ liệu gốc; NULL ≠ 0; không UNION/dedup ngầm.
4. LLM **được** sinh code chạy trong sandbox (luật 5 BTC cho phép), nhưng code luôn dùng chung semantic layer.
5. Public: được đọc rulebook public và phân tích log lỗi; **không** fine-tune, few-shot, hiệu chỉnh hay chọn
   model bằng đáp án/điểm public.
6. ≤ 15B tham số tổng mọi model; offline; tái lập |ΔPA| ≤ 0.03.
7. Chỉ giữ thay đổi có số đo trên validation theo loại cửa hàng.

## 2. Luồng

```
Data shop ──► [1] Nền dữ liệu lossless ──► [2] Chuẩn hoá schema (top-k ánh xạ) ─┐
Rulebook  ──► [3] Biên dịch luật → policy có cấu trúc ──────────────────────────┤
Câu hỏi   ──► [4] Hiểu câu: chuẩn hoá chữ + truy hồi schema/giá trị + parser DSL ┤
                                                                                 ▼
                              [5] Executor tất định  ──(ngoài DSL)──► [6] Agent sinh code
                                         │                                 │
                                         └──────────► [7] Kiểm chứng → đặc trưng
                                                            │
                                                    [8] Bộ chọn đã hiệu chỉnh
                                                            │
                                                    [9] Định dạng đầu ra
                                                            ▼
                                       đáp án / unanswerable / abstain
```

Chuẩn bị theo business (làm 1 lần, cache theo hash nguồn + rulebook + phiên bản code): [1] [2] [3].
Mỗi câu: [4] → [5]/[6] → [7] → [8] → [9].

## 3. Các tầng

### [1] Nền dữ liệu lossless

**Đường nộp bài thực tế (31/10):** tầng [1] chạy bằng `datamate/loader.py` + `datamate/clean.py`, không phải
`foundation.py` (v2 chỉ dùng ở nhánh v2; không thay vì số đo chưa cho thấy lợi PA). "Lossless" của đường nộp nghĩa là:
mỗi dòng giữ có provenance (`meta['src_part']`, `meta['src_row']`); mỗi dòng bỏ (tiêu đề lặp, ghi chú, tổng phụ) nằm
trong `meta['dropped_rows']` kèm lý do và giá trị; tổng phụ giữ nguyên ở `meta['subtotal_rows']` và được kiểm với chi
tiết; ô số / ngày không đọc được giữ chữ gốc ở `meta['unparsed_cells']`. `scripts/data_quality.py` kiểm bảo toàn: dòng
nguồn = dòng giữ + dòng bỏ có lý do (31/10: ~9,5 triệu dòng, 0 mất). Bảng dưới là thiết kế đích; dòng tổng phụ hiện vẫn
bỏ khỏi bảng (chưa thành cờ trên dòng). Tầng gold (`datamate/gold.py`) dựng trên `OrderView` sau tầng [2].

| Việc | Nguồn |
|---|---|
| Snapshot byte, ô gốc, dòng vật lý, sheet, encoding, merged cell, Excel serial date | v2 `foundation.py` |
| Header nhiều hàng là ứng viên, không xoá dòng | v2 `foundation.py` |
| Ép kiểu số/ngày có bằng chứng, không đoán locale khi mơ hồ; ngày tiếng Việt `1 thg 6 2025` | v2 `projections.py`, `temporal.py`; legacy `parsing.py` |
| Bỏ dòng tổng phụ / ghi chú như **cờ** trên dòng, không xoá | legacy `loader.py` (đổi thành gắn cờ) |
| Profile: null/distinct/khoá/khoảng thời gian/đơn vị | v2 `catalog.py` |
| Quan hệ giữa bảng với cardinality (1-1, 1-n, n-n) và độ phủ giá trị | v2 `catalog.py`; plan §6.3 |
| Partition theo tháng/ca/chi nhánh: kiểm schema, trùng dòng, trùng file trước khi gộp | v2 `catalog.py`; plan §6.4 |
| Provenance từ giá trị tính ra tới ô nguồn (dùng để debug và làm đặc trưng) | v2; plan §6.1 |

### [2] Chuẩn hoá schema về ontology chuẩn

Ontology: đơn, dòng hàng, thanh toán, trả hàng, sản phẩm, khách, nhân viên, nhà cung cấp, lô nhập/tồn,
công nợ, dịch vụ/sửa chữa, ca, giao hàng; cộng vai trò "khác" cho ngành lạ (plan §7.1, một bảng có thể nhiều vai trò).

| Việc | Nguồn |
|---|---|
| LLM đọc README + hồ sơ cột → gán vai trò | legacy `schema_card.py` |
| Ánh xạ từ README không cần LLM | legacy `semantic.py` |
| Sổ chuyên biệt: công nợ, lô/hạn dùng, nhập, dịch vụ, sổ dòng không mã đơn, phiếu nhập | legacy `debt.py`, `lots.py`, `imports.py`, `service_ledger.py`, `line_ledger.py`, `receipt_sem.py` |
| **Giữ top-k ánh xạ**, không chốt sớm | plan §7.4 → mới |
| Chấm ánh xạ bằng bất biến số: SL×giá≈thành tiền, tổng đơn không đổi trong đơn, khoá duy nhất, ngày trong kỳ hoạt động, join không nhân bản | legacy (một phần trong `schema_card.py`); plan §12 → mới (bộ chấm chung) |
| Semantic contract cho từng metric: nguồn, grain, join path, cột thời gian, đơn vị, luật áp dụng, điều kiện thiếu | plan §7.2; v2 `semantics.py` (khung) |
| Contract ghi rõ **nguồn đối trừ**: trạng thái trả trên đơn hay bảng trả riêng, giảm giá / voucher / điểm cấp đơn, thu máy cũ, phí ship; metric tiền thiếu nguồn đối trừ mà shop có bảng loại đó → không dựng view | phản biện 30/10 (R2); legacy `order_adjustment_total`, `returned_count` |
| Số lượng quy về đơn vị cơ bản (`base_qty`) chỉ theo bảng quy đổi / README (rulebook: hệ số luôn đọc ở đó, không đoán từ tên hàng); mặt hàng có đơn vị lạ không có hệ số → metric số lượng của shop không dựng | legacy `line_ledger.py` (`bqty`), `qty_sold`; phản biện 30/10 (R3) |
| View chuẩn ảo (`v_fact_sales`, `v_dim_products`…) chỉ dựng khi contract đủ điều kiện; không đủ thì giữ bảng thô | plan §7.3 → mới |
| Phân giải thực thể: cùng tên khác mã, nhiều cách viết, tên ↔ mã | v2 `entities.py`; legacy `helpers.py` (khớp mờ) |

### [3] Biên dịch rulebook → policy

| Việc | Nguồn |
|---|---|
| Bảng khái niệm × **phương án đã liệt kê**: doanh thu (dòng/đơn, trừ giảm giá cấp nào, VAT, phí ship, trả hàng toàn phần/một phần), đơn hợp lệ, kỳ thời gian, "bán chạy", tie-break, làm tròn, giá vốn, khách trung thành/sộp/đổi ý, giờ cao điểm, cửa sổ Tết… | mới; danh mục khái niệm lấy từ `docs/legacy/RULEBOOK_COVERAGE.md`, `docs/legacy/PUBLIC_RULEBOOK.md` |
| LLM đọc rulebook của split → chọn phương án, kèm trích dẫn câu luật | mới |
| Luật không khớp phương án nào → LLM viết biểu thức DSL; câu nào dùng luật đó thì giảm độ tin cậy | mới |
| Phát hiện luật đổi so với bản đã biết (chỉ để báo, không tự tắt) | legacy `rule_guard.py`, `known_rules.json` |
| Thứ tự ưu tiên thời gian: `as_of_date` → rulebook split → `calendar.csv` → câu hỏi; không lấy MAX(ngày) làm hôm nay; khoảng nửa mở | plan §9; v2 `temporal.py`; legacy `helpers.py` |
| Kiểm: thay policy vào train/strange, đáp án phải giữ nguyên | mới |
| Kiểm bất biến của policy trên split đang chạy (không cần đáp án): doanh thu ≥ 0, tổng theo đơn = tổng theo dòng khi policy nói vậy, đơn huỷ ⊂ đơn, số lượng cơ bản ≥ số lượng ghi; vi phạm → hạ tin cậy mọi câu dùng policy đó | phản biện 30/10 → mới |

Bỏ: so khớp nguyên văn câu luật (`temporal_rules.py`, `metric_rules.py`).

### [4] Hiểu câu hỏi

| Việc | Nguồn |
|---|---|
| Sửa Telex, teencode, viết tắt; giữ câu gốc và vết thay thế | legacy `telex.py`, `vi_vocab.json`; v2 `question_text.py` |
| Truy hồi giá trị (tên sản phẩm/khách trong câu → ô dữ liệu) và schema liên quan, giữ đường join | v2 `value_context.py`, `schema_context.py` |
| Đoán trước kiểu đáp án | legacy `answer_type.py` |
| **DSL tổ hợp**: metric × lọc × kỳ × nhóm × xếp hạng/top-k × so sánh/tỉ lệ × kiểu đầu ra; nhiều bước; anti-join; distinct | legacy `plan.py` (≈40 phép làm phép nguyên tử) + v2 `ir.py` (khung tổ hợp) |
| Phân biệt bắt buộc trong DSL: doanh thu ≠ tiền thu; huỷ ≠ trả; khách mới ≠ mua lần đầu; số mặt hàng ≠ số lượng bán; số đơn ≠ số dòng | plan §8 → thành kiểu riêng trong DSL + cặp ví dụ đối nghịch khi fine-tune |
| **Gán nhãn chương trình bằng tìm kiếm theo đáp án** trên train + synth: liệt kê chương trình, giữ cái chạy ra gold, ưu tiên đơn giản và nhất quán theo mẫu câu | mới |
| Fine-tune LoRA parser (merge vào model chính), sinh k chương trình | mới; legacy `synth_sft.py` (khung dữ liệu SFT) |
| Few-shot động từ kho lời giải đúng (train/synth) | legacy `shots.py` |
| Sửa có mục tiêu: lỗi cột không tồn tại / sai kiểu / sai grain trả về đúng chỗ sai, tối đa N vòng | v2 `planner.py`; plan §10 Strategy C |
| Mọi phép xếp hạng / top-k của DSL sort theo (số đo giảm, **mã tăng**; không có mã thì tên) ngay trong executor, không để chương trình tự chọn (đúng rulebook "Bằng nhau") | rulebook train; agent prompt rule 10; phản biện 30/10 (R5) |
| Gộp trước khi join: bảng con 1-n (nhân viên, tiền công, phụ tùng của một phiếu) tính tổng theo khoá phiếu rồi mới join; executor kiểm số dòng sau join ≤ số dòng bảng chính, vượt → lỗi grain | phản biện 30/10 (R4); legacy `service_ledger.py` |

### [5] Executor tất định

| Việc | Nguồn |
|---|---|
| Phép nguyên tử đã kiểm trên train (0 sai). **Vẫn còn sai trên public**: đáp án tất định là một ứng viên qua [7]–[8], parser DSL chạy cả trên câu tất định đã trả lời để đối chiếu | legacy `sem_api.py`, `semantic.py`, `*_sem.py`, `business_query.py` |
| Tổ hợp mới: IR → DuckDB có tham số, allowlist cột, chỉ đọc, timeout, giới hạn bộ nhớ/đầu ra | v2 `engine.py`; plan §11 |
| Decimal cho tiền; không làm tròn sớm; báo giá trị không biểu diễn chính xác | v2 `engine.py`, `verifier.py`; plan §12 |
| Phân biệt lỗi cú pháp / schema / nghiệp vụ / thiếu dữ liệu; 0 dòng ≠ unanswerable | plan §11 → mới (mã lỗi chuẩn) |

### [6] Agent sinh code (lưới an toàn)

| Việc | Nguồn |
|---|---|
| LLM viết pandas → chạy process con, timeout, lặp tới khi có đáp án | legacy `agent.py`, `executor.py` |
| Công cụ dùng chung semantic layer (gọi lại phép tất định, policy từ [3]) | legacy `agent_tools.py`, `rule_recall.py`, `helpers.py` |
| Bắt lỗi code thường gặp | legacy `lint.py` |
| Vết thực thi (bảng/cột nào đã đọc) làm đặc trưng | legacy `agent_evidence.py` |
| Brief shop cho prompt | legacy `brief.py` → thay bằng ngữ cảnh từ [2] + [4] |

### [7] Kiểm chứng → đặc trưng

Các mức kiểm của plan §13 được giữ, nhưng đầu ra là **đặc trưng số** cho [8], không phải cổng chặn:

| Mức | Đặc trưng |
|---|---|
| Cấu trúc | bảng/cột tồn tại, kiểu, grain đầu ra đúng kiểu dự kiến |
| Ngữ nghĩa | chương trình dùng đúng contract/policy; luật dùng có chắc không |
| Thực thi | lỗi ép kiểu, rỗng, NULL bị bỏ, join nhân bản dòng |
| Tính độc lập | đường tính thứ hai (Decimal evaluator của v2 hoặc phép legacy) có khớp không. Đặc trưng ghi **mức chung gốc**: hai đường dùng chung view / contract / policy thì "khớp" chỉ chứng minh phép tính, không chứng minh nghĩa (phản biện 30/10, R1) |
| Guard trên đáp án cuối | `datamate/arbiter.py` chạy trên đáp án đã chọn **kể cả khi mọi đường đồng ý**; guard veto (`in_stock`) chặn thẳng, guard còn lại là đặc trưng |
| Liên ứng viên | các ánh xạ top-k / chương trình top-k / agent có cùng đáp án không; nếu khác thì khác ở nguồn, grain, lọc, kỳ hay luật (plan §13 L5) |

Nguồn: v2 `verifier.py`, `semantic_obligations.py` (đổi thành đặc trưng); legacy `agent_evidence.py`.

### [8] Bộ chọn đã hiệu chỉnh

| Việc | Nguồn |
|---|---|
| Logistic/GBM nhỏ dự đoán P(đúng), out-of-fold theo **loại cửa hàng** | legacy `calibrate_abstain.py` (mở rộng sang mọi ứng viên) |
| Đặc trưng: [7] + logprob parser + độ phức tạp chương trình + độ chắc ánh xạ/policy | legacy `calibrate_logprob.py`; plan §15 |
| Trả lời khi P > 0.5 + biên; biên chọn trên validation | plan §15 |
| Biên theo **mức lạ** của shop (loại shop chưa gặp, độ chắc ánh xạ schema thấp, luật đổi): chọn trên fold leave-one-business-type-out + strange2 + synth, không đặt cứng 0.65–0.70 | phản biện 30/10 (R6) → mới |
| Đo riêng tỉ lệ "mọi đường đồng ý nhưng sai" trên train/strange; nếu đáng kể thì không cho đặc trưng đồng thuận gánh một mình | phản biện 30/10 (R1) → mới |
| **Unanswerable là lớp riêng**: khái niệm cần không có vai trò nào trong ánh xạ, kỳ hỏi ngoài dữ liệu, thông tin không tồn tại | legacy `unanswerable.py` + mới |
| Phân bổ tính toán thích nghi: câu dễ (ứng viên đồng ý, phép đã kiểm) → 1 lượt; câu không chắc → thêm chương trình/ánh xạ/agent tới ngân sách | plan §14 → mới |

### [9] Định dạng đầu ra

| Việc | Nguồn |
|---|---|
| Đúng 9 kiểu chấm; làm tròn 2 số; ngày `yyyy-mm-dd` / `yyyy-mm`; list có thứ tự; table | legacy `answer.py`, `scoring.py` |
| Hỏi tên thì trả tên (có ánh xạ rõ), hỏi mã thì trả mã | plan §16 |
| Table: cột theo đúng thứ tự câu hỏi nêu, tên cột snake_case không dấu theo ontology. Bộ chấm khớp cột theo tên đã chuẩn hoá, lệch tên thì theo vị trí, nên **thứ tự cột** là phần phải đúng | `docs/datamate_track_info.md`; phản biện 30/10 (R5) |
| Phân biệt 0 hợp lệ / tập rỗng / thiếu nguồn / không đủ thông tin | plan §16 |
| Chỉ lọc `tồn > 0` khi câu hỏi thật sự hỏi hàng đang tồn | plan §16 |

## 4. Runtime và nộp bài

| Việc | Nguồn |
|---|---|
| Chuẩn bị theo business 1 lần, cache theo hash nguồn + rulebook + code | v2 `pipeline.py`; plan §7.5, §17 |
| Ghi checkpoint atomic, chạy lại tiếp được, một câu lỗi không làm sập cả lượt | v2 `pipeline.py`; legacy `run.py` |
| Một model ≤ 15B (Qwen3-14B + LoRA đã merge), pin revision + SHA; không embedding model | v2 `model.py`; plan §18 |
| vLLM offline trên 2×5090; xử lý timeout, OOM, retry có giới hạn; còn thời gian bao nhiêu ghi bấy nhiêu | legacy `predict.sh`, `Dockerfile`, `scripts/vast/`; plan §19 |
| Tái lập: temperature 0 / seed cố định cho bước quyết định; ghi log quyết định trong image | mới |
| Dry-run Docker trên vast.ai với `--network none` trước khi nộp | mới |

## 5. Đánh giá và quản lý thí nghiệm

| Việc | Nguồn |
|---|---|
| Thước đo chính: **leave-one-business-type-out** trên train | v2 `evaluation.py` (mở rộng) |
| Shop lạ hoá từ train + shop tổng hợp thuộc ngành mới (có oracle) | legacy `make_strange_shops.py`, `datamate/synth/` |
| Báo cáo mỗi lượt: đúng/sai/abstain, PA, precision, coverage, runtime, số lần gọi LLM/token, theo loại shop và kiểu đáp án, đúng→sai và sai→đúng | plan §22; v2 `report_v2_run.py`; legacy `compare_runs.py` |
| Ablation theo tầng (tắt từng tầng [2] top-k, [3], parser fine-tune, [6], [8]) thay cho chuỗi A0–A6 | plan §22 (rút gọn) |
| Ghi commit, hash config, model revision, hash split/rulebook, seed; không ghi đè lượt cũ | plan §26; v2 `predict_v2.py` |
| Thứ tự tìm nguyên nhân khi sai: ingest → header → quan hệ/grain → nghĩa metric → thời gian → chương trình → thực thi → ép số → định dạng → chọn | plan §23 |
| Bộ test hồi quy theo danh sách ca bắt buộc (grain khác nhau cùng mã đơn, header nhiều hàng, merged cell, số kiểu VN, partition trùng, trả một phần, phí cấp đơn/dòng, tên trùng khác mã, n-n join, teencode, 0 hợp lệ, top-k có hoà, list có thứ tự, làm tròn %, JSON hỏng/timeout/OOM) | plan §24 |
| Public: chỉ phân tích log lỗi, sửa phải tổng quát và kiểm trên train/strange | quyết định của người dùng 28–29/10 |

## 6. Phản biện 30/10: nhận / không nhận

| Điểm | Kết luận | Lý do |
|---|---|---|
| R1 đồng thuận ảo | **Nhận** | Đúng: hai đường cùng view thừa kế cùng lỗi nghĩa. Thêm mức chung gốc vào đặc trưng, guard chạy cả khi đồng ý, đo tỉ lệ đồng ý mà sai |
| R2 giao dịch đối trừ | Nhận một phần | Policy [3] đã liệt kê trả toàn phần/một phần, giảm giá cấp nào, phí ship; thiếu là contract chưa ghi **nguồn** đối trừ → thêm |
| R3 quy đổi đơn vị | Đã có, ghi rõ hơn | Legacy đã quy về đơn vị cơ bản (`bqty`, `qty_sold`). Không nhận "tự trích hệ số đóng gói": rulebook bắt đọc hệ số ở bảng quy đổi / README |
| R4 nhân bản dòng khi join | Nhận | [2] và [7] mới có kiểm, chưa có quy tắc gộp trước khi join trong DSL / executor |
| R5 header table, hoà top-N | Nhận một phần | Bộ chấm khớp tên đã chuẩn hoá rồi theo vị trí → thứ tự cột quan trọng hơn chuẩn hoá tên. Hoà theo **mã tăng dần** (rulebook), không theo chữ cái tên như phản biện đề xuất |
| R6 lệch hiệu chỉnh trên ngành lạ | Nhận ý, không nhận con số | Biên theo mức lạ, chọn trên fold theo loại shop. Không ép giảm trọng số logprob bằng tay: để hồi quy học, kiểm bằng ablation |
| Tứ phân đầu ra (giá trị / rỗng / 0 / unanswerable) | Đã có | [9] và [5] (0 dòng ≠ unanswerable) |

## 7. Ý của plan v2 không lấy

| Ý trong plan | Lý do |
|---|---|
| Cấm LLM sinh code (nguyên tắc 4) | BTC cho phép; bỏ thì mất lưới an toàn linh hoạt nhất |
| Cấm mọi thứ của public (nguyên tắc 10) | Chặt hơn luật BTC và quyết định đã chốt (được đọc rulebook public, phân tích log) |
| Verifier là cổng chặn, đòi adapter viết tay chứng minh | Làm LLM không thể tăng số câu trả lời được; thay bằng đặc trưng + hiệu chỉnh |
| So nguyên văn câu luật | Vỡ khi rulebook private viết khác |
| Gate khảo sát paper trước mọi phần | Tốn thời gian, không làm tăng PA; chỉ khảo sát khi chọn kỹ thuật mới lớn |
| Chuỗi ablation A0–A6 đầy đủ | Thay bằng ablation theo tầng |
