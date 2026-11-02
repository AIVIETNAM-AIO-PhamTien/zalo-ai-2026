import pandas as pd
import pytest

from datamate.agent_tools import checked_merge


def test_many_to_one_preserves_lines_and_records_audit():
    left = pd.DataFrame({"id": ["A", "A", "B"], "amount": [10, 20, 30]})
    right = pd.DataFrame({"id": ["A", "B"], "name": ["An", "Bình"]})
    result = checked_merge(left, right, on="id", require_match=True)
    assert result.amount.sum() == 60 and len(result) == 3
    assert result.attrs["join_audit"]["matched_rows"] == 3
    assert list(left.columns) == ["id", "amount"]  # không sửa đầu vào


def test_duplicate_dimension_errors_instead_of_multiplying_money():
    with pytest.raises(pd.errors.MergeError, match="khoá phải lặp"):
        checked_merge(pd.DataFrame({"id": ["A"], "amount": [100]}),
                      pd.DataFrame({"id": ["A", "A"]}), on="id")


def test_missing_keys_and_nulls_do_not_silently_become_zero():
    left, right = pd.DataFrame({"id": ["A", "B"]}), pd.DataFrame({"id": ["A"]})
    with pytest.raises(ValueError, match="thiếu khoá"):
        checked_merge(left, right, on="id", require_match=True)
    assert checked_merge(left, right, on="id").attrs["join_audit"]["left_only"] == 1
    with pytest.raises(ValueError, match="ô trống"):
        checked_merge(pd.DataFrame({"id": [None]}), pd.DataFrame({"id": [None]}), on="id", require_match=True)


def test_different_key_names_suffixes_and_indicator_collisions():
    left = pd.DataFrame({"id": ["B"], "__join_status__": [1]})
    right = pd.DataFrame({"code": ["A"], "id": [10]})
    with pytest.raises(ValueError, match="thiếu khoá"):
        checked_merge(left, right, left_on="id", right_on="code", require_match=True)
    result = checked_merge(left, right, left_on="id", right_on="code")
    assert result["__join_status__"].tolist() == [1]
    with pytest.raises(ValueError, match="Chỉ định rõ"):
        checked_merge(left, right)


def test_join_guidance_is_opt_in(tmp_path):
    import datetime as dt
    from dataclasses import replace
    from datamate.agent import ShopContext, Task, build_messages
    ctx = ShopContext(tmp_path / "unused.pkl", None, "", "", "")
    task = Task("q", "s", "Doanh thu tháng 1/2025?", dt.date(2025, 2, 1))
    assert "checked_merge" not in str(build_messages(task, ctx))
    assert "checked_merge" in str(build_messages(task, replace(ctx, checked_joins=True)))
