#!/usr/bin/env python3
"""
백테스팅 결과 기반 스코어링 가중치 자동 업데이트.
Usage: python apply_weights.py [--analysis PATH] [--dry-run]
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
WEIGHTS_PATH = ROOT / "scoring_weights.json"
BACKTESTING_DIR = ROOT / "results-backtesting"

# 팩터별 최소/최대 배점 제약 (너무 급격한 변경 방지)
CONSTRAINTS = {
    "technical": {
        "52w_low": (5, 20),
        "rsi":     (8, 20),
        "macd":    (8, 18),
        "bollinger": (5, 15),
        "volume":  (5, 12),
    },
    "valuation": {
        "pe":        (8, 15),
        "fcf_yield": (3, 12),
        "peg":       (3, 10),
    },
}

# 가중치 조정 매핑: backtest JSON factor_key → (category, weight_key)
FACTOR_MAP = {
    "rsi":             ("technical", "rsi"),
    "macd_golden_cross": ("technical", "macd"),
    "bollinger_pct_b": ("technical", "bollinger"),
    "52w_pct_from_low": ("technical", "52w_low"),
    "fcf_yield":       ("valuation", "fcf_yield"),
    "peg":             ("valuation", "peg"),
}

# 팩터별 "유효" 조건 함수
CONDITIONS = {
    "rsi":               lambda v: 30 <= v <= 45,
    "macd_golden_cross": lambda v: v is True,
    "bollinger_pct_b":   lambda v: v is not None and v < 0.3,
    "52w_pct_from_low":  lambda v: 5 <= v <= 20,
    "fcf_yield":         lambda v: v is not None and v > 5,
    "peg":               lambda v: v is not None and v < 1.0,
}


def _avg(lst):
    return sum(lst) / len(lst) if lst else None


def compute_factor_differentials(picks: list) -> dict:
    """각 팩터별 2주 수익률 차이(해당 - 비해당) 계산."""
    with_2w = [p for p in picks if p["returns"]["2w"] is not None]
    diffs = {}
    for fkey, cond in CONDITIONS.items():
        t = [p["returns"]["2w"] for p in with_2w
             if p["factors"].get(fkey) is not None and cond(p["factors"][fkey])]
        f = [p["returns"]["2w"] for p in with_2w
             if p["factors"].get(fkey) is not None and not cond(p["factors"][fkey])]
        at, af = _avg(t), _avg(f)
        n = len(t)
        if at is not None and af is not None:
            diffs[fkey] = {"diff": round(at - af, 2), "n": n, "avg_true": at, "avg_false": af}
    return diffs


def recommend_weights(current: dict, diffs: dict) -> dict:
    """
    수익률 차이 기반 가중치 조정.
    - diff > +1.5%: 현행 대비 +2점
    - diff > +0.5%: 현행 대비 +1점
    - diff < -1.0%: 현행 대비 -3점
    - diff < 0%:    현행 대비 -1점
    - N < 20: 조정 없음 (샘플 부족)
    """
    new_weights = {
        "technical": dict(current.get("technical", {})),
        "valuation": dict(current.get("valuation", {})),
    }

    changes = []
    for fkey, stat in diffs.items():
        if fkey not in FACTOR_MAP:
            continue
        cat, wkey = FACTOR_MAP[fkey]
        diff = stat["diff"]
        n = stat["n"]
        cur_val = new_weights[cat].get(wkey, 0)
        mn, mx = CONSTRAINTS[cat][wkey]

        if n < 20:
            changes.append(f"  {wkey}: 변경 없음 (N={n} 샘플 부족)")
            continue

        if diff > 1.5:
            delta = 2
        elif diff > 0.5:
            delta = 1
        elif diff < -1.0:
            delta = -3
        elif diff < 0:
            delta = -1
        else:
            delta = 0

        new_val = max(mn, min(mx, cur_val + delta))
        if new_val != cur_val:
            new_weights[cat][wkey] = new_val
            changes.append(f"  {wkey}: {cur_val} → {new_val} (diff={diff:+.2f}%, N={n})")
        else:
            changes.append(f"  {wkey}: {cur_val} (변경 없음, diff={diff:+.2f}%)")

    return new_weights, changes


def load_latest_analysis() -> dict | None:
    """results-backtesting/에서 가장 최신 analysis-*.json 로딩."""
    files = sorted(BACKTESTING_DIR.glob("analysis-*.json"))
    if not files:
        return None
    return json.loads(files[-1].read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description="백테스팅 결과로 scoring_weights.json 업데이트")
    parser.add_argument("--analysis", type=Path, help="분석 JSON 파일 경로 (기본: 최신)")
    parser.add_argument("--dry-run", action="store_true", help="변경 내용만 출력, 파일 저장 안 함")
    args = parser.parse_args()

    # 분석 데이터 로딩
    if args.analysis:
        data = json.loads(args.analysis.read_text(encoding="utf-8"))
    else:
        data = load_latest_analysis()
        if data is None:
            print("오류: results-backtesting/analysis-*.json 파일 없음", file=sys.stderr)
            sys.exit(1)

    picks = data.get("picks", [])
    with_2w = [p for p in picks if p["returns"]["2w"] is not None]
    print(f"분석 데이터: 전체 {len(picks)}픽 / 2주 수익률 확인 {len(with_2w)}픽")

    if len(with_2w) < 20:
        print("샘플 부족 (N<20). 가중치 업데이트를 건너뜁니다.")
        sys.exit(0)

    # 현재 가중치 로딩
    try:
        current = json.loads(WEIGHTS_PATH.read_text(encoding="utf-8"))
    except Exception:
        current = {"technical": {}, "valuation": {}}

    # 팩터 차이 계산
    diffs = compute_factor_differentials(picks)
    print("\n팩터별 2주 수익률 차이 (해당 - 비해당):")
    for fkey, stat in diffs.items():
        print(f"  {fkey}: {stat['diff']:+.2f}% (N={stat['n']})")

    # 권장 가중치 계산
    new_weights, changes = recommend_weights(current, diffs)
    print("\n가중치 변경 내역:")
    for c in changes:
        print(c)

    if args.dry_run:
        print("\n[dry-run] 파일 저장 안 함")
        return

    # 저장
    new_weights["_comment"] = current.get("_comment", "")
    new_weights["_updated"] = datetime.now().strftime("%Y-%m-%d")
    WEIGHTS_PATH.write_text(json.dumps(new_weights, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장됨: {WEIGHTS_PATH}")


if __name__ == "__main__":
    main()
