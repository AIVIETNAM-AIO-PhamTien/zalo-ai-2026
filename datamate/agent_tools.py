"""Công cụ dữ liệu chung cho agent, không chứa luật ngành hay lời giải câu hỏi."""
import pandas as pd


def checked_merge(left, right, on=None, *, left_on=None, right_on=None, how="left",
                  validate="many_to_one", require_match=False, suffixes=("_x", "_y")):
    """Nối có kiểm quan hệ; không tự xoá dòng trùng/điền số 0 để che mất khoá.

    validate theo pandas. require_match chỉ dành cho left join, kiểm mọi dòng
    trái có bản ghi phải. Thông tin số dòng ở result.attrs['join_audit'].
    """
    if require_match and how != "left":
        raise ValueError("require_match cần how='left' để kiểm cả những dòng không nối được")
    if validate not in {"one_to_one", "many_to_one", "one_to_many", "many_to_many"}:
        raise ValueError("validate phải khai báo quan hệ one_to_one/many_to_one/one_to_many/many_to_many")
    if on is None and (left_on is None or right_on is None):
        raise ValueError("Chỉ định rõ on hoặc cả left_on/right_on; không tự nối mọi cột cùng tên")
    lk, rk = (on, on) if on is not None else (left_on, right_on)
    lk = [lk] if isinstance(lk, str) else list(lk)
    rk = [rk] if isinstance(rk, str) else list(rk)
    if require_match and (left[lk].isna().any().any() or right[rk].isna().any().any()):
        raise ValueError("Khoá nối có ô trống; không coi null-null là khớp danh mục")
    indicator = "__join_status__"
    while indicator in left.columns or indicator in right.columns:
        indicator += "_"
    try:
        result = pd.merge(left, right, on=on, left_on=left_on, right_on=right_on,
                          how=how, validate=validate, suffixes=suffixes, indicator=indicator)
    except pd.errors.MergeError as exc:
        left_duplicates = left.loc[left.duplicated(lk, keep=False), lk].drop_duplicates().head(5).to_dict("records")
        right_duplicates = right.loc[right.duplicated(rk, keep=False), rk].drop_duplicates().head(5).to_dict("records")
        raise pd.errors.MergeError(f"{exc}; khoá trái lặp: {left_duplicates}; khoá phải lặp: {right_duplicates}. "
                                   "Kiểm độ hạt bảng và khoá ghép; không tự drop_duplicates để ép nối.") from exc
    counts = {str(k): int(v) for k, v in result[indicator].value_counts().items()}
    if require_match and counts.get("left_only", 0):
        display_keys = [key if key in result else f"{key}{suffixes[0]}" for key in lk]
        missing = result.loc[result[indicator] == "left_only", display_keys].drop_duplicates().head(5).to_dict("records")
        raise ValueError(f"Có {counts['left_only']} dòng trái thiếu khoá danh mục; ví dụ: {missing}")
    result = result.drop(columns=indicator)
    result.attrs["join_audit"] = {"left_rows": len(left), "right_rows": len(right), "result_rows": len(result),
                                 "validate": validate, "matched_rows": counts.get("both", 0),
                                 "left_only": counts.get("left_only", 0), "right_only": counts.get("right_only", 0)}
    return result
