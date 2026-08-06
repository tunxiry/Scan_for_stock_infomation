#!/usr/bin/env python3
"""
미장 개장 30분 전 트레이딩 후보 스캐너.
yfinance로 가격을 받아 지지/저항을 계산하고, 손익비 기준을 통과한 종목만 텔레그램으로 보냅니다.

필요 환경변수:
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
설치:
  pip install yfinance pandas numpy requests
"""
import json
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ── 설정 ────────────────────────────────────────────────
ICKERS = ["SPY", "QQQ", "SOXX", "IGV", "META", "AMZN", "GOOGL", "NVDA", "TSLA", "AAPL", "MSFT", "MSTR"]

MIN_RR = 2.0          # 최소 손익비 (익절폭 / 손절폭)
MAX_SUPPORT_GAP = 3.0 # 현재가가 지지선 위로 이 % 이내일 것
SWING_WINDOW = 5      # 스윙 고/저점 판정 창 (좌우 N봉)
ATR_MULT = 0.5        # 손절가를 지지선 아래로 ATR의 몇 배 내릴지
LOOKBACK = "1y"
STATE_FILE = "state.json"  # 같은 종목 중복 알림 방지용

# 피벗 레벨 표기명
PIVOT_LABEL = {"S1": "1차 지지", "S2": "2차 지지",
               "R1": "1차 저항", "R2": "2차 저항"}

TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TG_CHAT = os.environ.get("TELEGRAM_CHAT_ID")


# ── 지표 ────────────────────────────────────────────────
def atr(df: pd.DataFrame, n: int = 14) -> float:
    """Wilder ATR. 손절 여유폭(변동성 버퍼) 산출용."""
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return float(tr.ewm(alpha=1 / n, adjust=False).mean().iloc[-1])


def swing_levels(df: pd.DataFrame, w: int = SWING_WINDOW):
    """프랙탈 방식 스윙 저점/고점 추출."""
    lows, highs = [], []
    l, h = df["Low"].values, df["High"].values
    for i in range(w, len(df) - w):
        if l[i] == l[i - w:i + w + 1].min():
            lows.append(float(l[i]))
        if h[i] == h[i - w:i + w + 1].max():
            highs.append(float(h[i]))
    return lows, highs


def cluster(levels, tol=0.015):
    """가까운 레벨끼리 묶어 대표값(평균)과 터치 횟수를 반환. 터치가 많을수록 강한 레벨."""
    out = []
    for lv in sorted(levels):
        if out and abs(lv - out[-1][0]) / out[-1][0] < tol:
            vals = out[-1][1] + [lv]
            out[-1] = (float(np.mean(vals)), vals)
        else:
            out.append((lv, [lv]))
    return [(price, len(vals)) for price, vals in out]


def pivots(df: pd.DataFrame):
    """전일 고가/저가/종가 기준 클래식 피벗 (S1/S2, R1/R2)."""
    h, l, c = df["High"].iloc[-1], df["Low"].iloc[-1], df["Close"].iloc[-1]
    p = (h + l + c) / 3
    return {
        "S1": 2 * p - h, "S2": p - (h - l),
        "R1": 2 * p - l, "R2": p + (h - l),
    }


# ── 종목 분석 ───────────────────────────────────────────
def analyze(ticker: str):
    tk = yf.Ticker(ticker)
    df = tk.history(period=LOOKBACK, interval="1d", auto_adjust=False)
    if len(df) < 60:
        return None

    # 프리마켓 포함 최신가. 실패하면 전일 종가로 폴백.
    price, src = float(df["Close"].iloc[-1]), "전일종가"
    try:
        pm = tk.history(period="1d", interval="1m", prepost=True, auto_adjust=False)
        if not pm.empty:
            price, src = float(pm["Close"].iloc[-1]), "프리마켓"
    except Exception:
        pass

    a = atr(df)
    lows, highs = swing_levels(df)
    piv = pivots(df)
    sma = {n: float(df["Close"].rolling(n).mean().iloc[-1])
           for n in (20, 50, 200) if len(df) >= n}

    # 지지 후보: 스윙 저점 + 피벗 S1/S2 + 이평선 (현재가 아래)
    sup_raw = [lv for lv in lows if lv < price]
    sup_raw += [v for k, v in piv.items() if k.startswith("S") and v < price]
    sup_raw += [v for v in sma.values() if v < price]
    # 저항 후보: 스윙 고점 + 피벗 R1/R2 + 이평선 (현재가 위)
    res_raw = [lv for lv in highs if lv > price]
    res_raw += [v for k, v in piv.items() if k.startswith("R") and v > price]
    res_raw += [v for v in sma.values() if v > price]

    if not sup_raw or not res_raw:
        return None

    support = max(cluster(sup_raw), key=lambda x: x[0])       # 가장 가까운 지지

    def touches(levels, target, tol=0.015):
        """해당 가격대를 실제로 몇 번 되돌림했는지. 스윙 고/저점만 셈."""
        return sum(1 for lv in levels if abs(lv - target) / target < tol)

    # 저항: 현재가에서 최소 1.5% 이상 떨어진 레벨 중, 실제로 2번 이상 막힌 곳을 우선.
    # (바로 위 잡음 레벨을 익절가로 잡으면 손익비가 무의미해짐)
    res_c = [c[0] for c in cluster(res_raw) if (c[0] - price) / price > 0.015]
    if not res_c:
        return None
    strong = [lv for lv in res_c if touches(highs, lv) >= 2]
    resistance = min(strong or res_c)

    stop = support[0] - ATR_MULT * a
    target = resistance
    risk, reward = price - stop, target - price
    if risk <= 0 or reward <= 0:
        return None

    gap = (price - support[0]) / price * 100
    rr = reward / risk

    reasons = []
    for n, v in sma.items():
        if abs(support[0] - v) / price < 0.01:
            reasons.append(f"{n}일선 지지")
    # 피벗 지지: S1 우선, 하나만 표기 (전일 변동폭이 좁으면 S1·S2가 겹침)
    for k in ("S1", "S2"):
        if abs(support[0] - piv[k]) / price < 0.01:
            reasons.append(f"피벗 {PIVOT_LABEL[k]}")
            break
    n_touch = touches(lows, support[0])
    if n_touch >= 2:
        reasons.append(f"지지 {n_touch}회 반등")
    elif n_touch == 1:
        reasons.append("스윙 저점")
    # 익절가가 피벗 저항과 겹치면 함께 표기 (R1 우선, 하나만)
    for k in ("R1", "R2"):
        if abs(target - piv[k]) / price < 0.01:
            reasons.append(f"목표 = 피벗 {PIVOT_LABEL[k]}")
            break

    return {
        "ticker": ticker, "price": price, "src": src,
        "stop": stop, "target": target, "rr": rr, "gap": gap,
        "reason": " + ".join(reasons) or "스윙 저점 지지",
    }


# ── 출력 ────────────────────────────────────────────────
def build_message(rows):
    kst = datetime.now(ZoneInfo("Asia/Seoul"))
    header = f"📈 <b>오늘의 주식 ({kst.month}/{kst.day})</b>"

    if not rows:
        return (f"{header}\n\n"
                f"오늘은 손익비 {MIN_RR}:1 이상 + 지지선 {MAX_SUPPORT_GAP}% 이내 조건을\n"
                f"만족하는 종목이 없습니다.")

    blocks = [
        "\n".join([
            f"🔹 <b>{r['ticker']}</b> 현재가 : {r['price']:.2f}",
            f"손절가 : {r['stop']:.2f} | 익절가 : {r['target']:.2f}",
            f"손익비 : {r['rr']:.1f} : 1 (지지선까지 -{r['gap']:.1f}%)",
            f"<b>[분석]</b> {r['reason']}",
        ])
        for r in rows
    ]
    return header + "\n\n" + "\n\n".join(blocks)


def send(text: str):
    if not TG_TOKEN or not TG_CHAT:
        print("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID 환경변수가 없습니다.", file=sys.stderr)
        print(text)
        return
    r = requests.post(
        f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
        json={"chat_id": TG_CHAT, "text": text,
              "parse_mode": "HTML", "disable_web_page_preview": True},
        timeout=30,
    )
    r.raise_for_status()


def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def main():
    """15분마다 실행돼도 같은 종목을 하루 한 번만 알립니다.
    DAILY_SUMMARY=1 이면 후보가 없어도 '없음' 메시지를 보냅니다(개장 30분 전 1회용)."""
    today = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    state = load_state()
    daily = os.environ.get("DAILY_SUMMARY") == "1"

    rows = []
    for t in TICKERS:
        try:
            r = analyze(t)
            if r and r["rr"] >= MIN_RR and r["gap"] <= MAX_SUPPORT_GAP:
                rows.append(r)
        except Exception as e:
            print(f"{t} 실패: {e}", file=sys.stderr)

    rows.sort(key=lambda x: x["rr"], reverse=True)
    fresh = [r for r in rows if state.get(r["ticker"]) != today][:5]

    if not fresh and not daily:
        print("새 신호 없음 — 발송 생략")
    else:
        send(build_message(fresh))
        for r in fresh:
            state[r["ticker"]] = today

    # 발송 여부와 무관하게 항상 기록 (없으면 워크플로의 git add가 실패)
    state = {k: v for k, v in state.items() if v == today}
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
