import json
import time
import random
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, Any, Tuple

import streamlit as st

# Optional (AI人格生成)
OPENAI_AVAILABLE = False
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False


# -----------------------------
# Core
# -----------------------------
class Vote(str, Enum):
    YES = "YES"
    NO = "NO"
    HOLD = "HOLD"


@dataclass
class PersonaConfig:
    name: str
    w_cost: float
    w_risk: float
    w_urgency: float
    w_public_impact: float
    yes_threshold: float
    no_threshold: float
    veto_risk_at: int
    hold_risk_at: int
    hold_public_impact_at: int
    style_note: str = ""


def default_personas() -> Dict[str, PersonaConfig]:
    return {
        "CASPER": PersonaConfig(
            name="CASPER",
            w_cost=-0.3, w_risk=-1.2, w_urgency=+0.6, w_public_impact=-0.8,
            yes_threshold=15.0, no_threshold=-15.0,
            veto_risk_at=85, hold_risk_at=60, hold_public_impact_at=75,
            style_note="安全・倫理・コンプラ重視。危ないものは止める。"
        ),
        "MELCHIOR": PersonaConfig(
            name="MELCHIOR",
            w_cost=-1.0, w_risk=-0.6, w_urgency=+0.8, w_public_impact=-0.2,
            yes_threshold=10.0, no_threshold=-10.0,
            veto_risk_at=95, hold_risk_at=70, hold_public_impact_at=90,
            style_note="合理性・費用対効果重視。非効率は嫌う。"
        ),
        "BALTHASAR": PersonaConfig(
            name="BALTHASAR",
            w_cost=-0.2, w_risk=-0.6, w_urgency=+0.3, w_public_impact=-1.1,
            yes_threshold=8.0, no_threshold=-8.0,
            veto_risk_at=98, hold_risk_at=75, hold_public_impact_at=60,
            style_note="人間的・納得感・評判重視。炎上と不信を嫌う。"
        ),
    }


def council_decide(votes: Dict[str, Vote], hold_priority: bool = True) -> Vote:
    yes = sum(1 for v in votes.values() if v == Vote.YES)
    no = sum(1 for v in votes.values() if v == Vote.NO)
    hold = sum(1 for v in votes.values() if v == Vote.HOLD)

    if hold_priority and hold >= 1 and no == 0:
        return Vote.HOLD
    if no >= 2:
        return Vote.NO
    if yes >= 2:
        return Vote.YES
    return Vote.HOLD


# -----------------------------
# Debate (AI discussion)
# -----------------------------
def _persona_tone(name: str) -> str:
    n = (name or "").upper()
    if "CASPER" in n:
        return "堅め・安全第一"
    if "MELCHIOR" in n:
        return "合理的・効率重視"
    if "BALTHASAR" in n:
        return "人間味・評判重視"
    return "中立"

def _can_use_openai() -> bool:
    return OPENAI_AVAILABLE and bool(st.secrets.get("OPENAI_API_KEY", None))

def _llm(text_system: str, text_user: str) -> str:
    api_key = st.secrets.get("OPENAI_API_KEY", None)
    client = OpenAI(api_key=api_key)
    model = st.secrets.get("OPENAI_MODEL", "gpt-5-mini")

    max_retries = 5
    base_sleep = 1.0

    for attempt in range(max_retries):
        try:
            resp = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": text_system},
                    {"role": "user", "content": text_user},
                ],
            )
            return getattr(resp, "output_text", None) or str(resp)

        except Exception as e:
            if "RateLimit" in e.__class__.__name__:
                # 1s,2s,4s... + 少しランダムで待って再試行
                time.sleep(base_sleep * (2 ** attempt) + random.uniform(0, 0.5))
                continue
            raise

    return "（混雑のため議論生成に失敗。少し待って再実行してください）"


import re

def _extract_vote_from_text(text: str) -> str | None:
    """
    LLM出力から YES/NO/HOLD を拾う。
    想定：文末に「最終投票：YES」などが入る。
    """
    if not text:
        return None

    # よくあるパターンを拾う（日本語/英語混在に耐える）
    patterns = [
        r"最終投票\s*[:：]\s*(YES|NO|HOLD)",
        r"Final\s*Vote\s*[:：]\s*(YES|NO|HOLD)",
        r"\b(Vote|投票)\b.*\b(YES|NO|HOLD)\b",
        r"\b(YES|NO|HOLD)\b",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            v = m.group(m.lastindex)  # 最後のキャプチャ
            v = v.upper()
            if v in ("YES", "NO", "HOLD"):
                return v
    return None


def build_debate_log(personas: Dict[str, PersonaConfig], proposal: dict, details: Dict[str, Any], rounds: int = 2):
    """
    returns:
      log: [{speaker, content}, ...]
      votes_after: {persona_name: "YES|NO|HOLD"}
      chair_summary: str | None
    """
    title = proposal.get("title", "")
    desc = proposal.get("description", "")
    numbers = f"cost={proposal['cost']}, risk={proposal['risk']}, urgency={proposal['urgency']}, public_impact={proposal['public_impact']}"

    log = []
    log.append({"speaker": "SYSTEM", "content": f"議題：{title}\n概要：{desc}\n指標：{numbers}\n---\nRound1：各人格の主張"})

    # 議論前の投票（初期値）
    votes_after = {name: details[name]["vote"] for name in personas.keys()}

    # Round1
    for name, cfg in personas.items():
        d = details[name]
        vote = d["vote"]
        reason = d["reason"]
        breakdown = d["breakdown"]

        if _can_use_openai():
            sys = f"あなたは意思決定人格『{name}』。口調は「{_persona_tone(name)}」。短く、箇条書きを混ぜ、賛否・根拠・懸念・条件を述べて。最後に「最終投票：YES/NO/HOLD」はまだ書かない。"
            usr = f"提案:{title}\n{desc}\n数値:{numbers}\n暫定投票:{vote}\n根拠:{reason}\n内訳:{json.dumps(breakdown, ensure_ascii=False)}"
            content = _llm(sys, usr)
        else:
            content = f"（{_persona_tone(name)}）結論は {vote}。\n理由：{reason}"

        log.append({"speaker": name, "content": content})

    if rounds < 2:
        return log, votes_after, None

    # Round2（ここで“最終投票”を書かせる）
    log.append({"speaker": "SYSTEM", "content": "---\nRound2：相互反論と妥協案 → 最終投票（再投票）"})
    others_summary = {k: {"vote": details[k]["vote"], "reason": details[k]["reason"][:120]} for k in personas.keys()}

    for name, cfg in personas.items():
        others = {k: v for k, v in others_summary.items() if k != name}

        if _can_use_openai():
            sys = (
                f"あなたは意思決定人格『{name}』。口調は「{_persona_tone(name)}」。"
                "他者の意見に1つ反論し、妥協案（条件）を提示。"
                "最後に必ず 1行で「最終投票：YES/NO/HOLD」を出力する。"
            )
            usr = (
                f"提案:{title}\n数値:{numbers}\n"
                f"他者意見:{json.dumps(others, ensure_ascii=False)}\n"
                f"あなたの暫定投票:{details[name]['vote']}\n根拠:{details[name]['reason']}\n"
                "注意：最終投票は議論を踏まえて変更してよい。"
            )
            content = _llm(sys, usr)
        else:
            # AIなしの場合：暫定のまま
            content = (
                f"（{_persona_tone(name)}）他者の意見を踏まえ、条件付きで進める余地はある。\n"
                f"最終投票：{details[name]['vote']}"
            )

        # ここで最終投票を抽出して votes_after を更新
        extracted = _extract_vote_from_text(content)
        if extracted is None:
            extracted = details[name]["vote"]  # 取れなければ暫定のまま
        votes_after[name] = extracted

        log.append({"speaker": name, "content": content})

    # 議長サマリー（任意：OpenAIが使える時だけ）
    chair_summary = None
    if _can_use_openai():
        sys = "あなたは議長（SYSTEM）。議論ログを読み、結論・条件・次のToDoを3点で簡潔に要約。"
        usr = "議論ログ:\n" + "\n\n".join([f"[{x['speaker']}]\n{x['content']}" for x in log])
        chair_summary = _llm(sys, usr)

    return log, votes_after, chair_summary

def score_vote(cfg: PersonaConfig, cost: int, risk: int, urgency: int, public_impact: int) -> Tuple[Vote, str, float, Dict[str, Any]]:
    # 強制ルール
    if risk >= cfg.veto_risk_at:
        return (
            Vote.NO,
            f"強制NO：risk({risk}) >= veto_risk_at({cfg.veto_risk_at})",
            -999.0,
            {"rule": "veto", "risk": risk, "veto_risk_at": cfg.veto_risk_at},
        )
    if risk >= cfg.hold_risk_at:
        return (
            Vote.HOLD,
            f"HOLD：risk({risk}) >= hold_risk_at({cfg.hold_risk_at}) → 対策/追加情報が必要",
            0.0,
            {"rule": "hold_risk", "risk": risk, "hold_risk_at": cfg.hold_risk_at},
        )
    if public_impact >= cfg.hold_public_impact_at:
        return (
            Vote.HOLD,
            f"HOLD：public_impact({public_impact}) >= hold_public_impact_at({cfg.hold_public_impact_at}) → 説明/合意形成が必要",
            0.0,
            {"rule": "hold_public", "public_impact": public_impact, "hold_public_impact_at": cfg.hold_public_impact_at},
        )

    contrib = {
        "cost": cfg.w_cost * cost,
        "risk": cfg.w_risk * risk,
        "urgency": cfg.w_urgency * urgency,
        "public_impact": cfg.w_public_impact * public_impact,
    }
    score = sum(contrib.values())

    ranked = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)
    top_parts = []
    for k, v in ranked[:3]:
        direction = "押し上げ" if v > 0 else "押し下げ"
        top_parts.append(f"{k}が{direction}（寄与 {v:+.1f}）")
    reason_core = " / ".join(top_parts)

    if score >= cfg.yes_threshold:
        return Vote.YES, f"スコア={score:.1f} >= YES閾値({cfg.yes_threshold})。主因：{reason_core}", score, contrib
    if score <= cfg.no_threshold:
        return Vote.NO, f"スコア={score:.1f} <= NO閾値({cfg.no_threshold})。主因：{reason_core}", score, contrib

    return Vote.HOLD, f"スコア={score:.1f} が中間 → 条件調整/追加情報待ち。主因：{reason_core}", score, contrib


def clamp_int_0_100(x: int) -> int:
    return int(max(0, min(100, x)))


def unique_key(personas: Dict[str, PersonaConfig], base: str) -> str:
    base = (base or "").strip() or "NEW_PERSONA"
    if base not in personas:
        return base
    i = 2
    while f"{base}_{i}" in personas:
        i += 1
    return f"{base}_{i}"


def manual_generate_persona(payload: dict) -> PersonaConfig:
    # 0-100 -> [-2,2]っぽい重みに雑マッピング（遊び用途）
    def map_weight(val, sign=1):
        return sign * ((val - 50) / 25.0)

    priority = payload["priority"]
    risk_tolerance = payload["risk_tolerance"]
    cost_sensitivity = payload["cost_sensitivity"]
    deadline_focus = payload["deadline_focus"]
    reputation_focus = payload["reputation_focus"]

    base = {"w_cost": -0.4, "w_risk": -0.4, "w_urg": 0.4, "w_pub": -0.4}
    if priority == "safety":
        base["w_risk"] -= 0.6
    elif priority == "cost":
        base["w_cost"] -= 0.6
    elif priority == "speed":
        base["w_urg"] += 0.6
    elif priority == "reputation":
        base["w_pub"] -= 0.6

    w_cost = base["w_cost"] + map_weight(cost_sensitivity, sign=-1)
    w_risk = base["w_risk"] + map_weight(100 - risk_tolerance, sign=-1)
    w_urg = base["w_urg"] + map_weight(deadline_focus, sign=+1)
    w_pub = base["w_pub"] + map_weight(reputation_focus, sign=-1)

    def clamp(v):
        return float(max(-2.0, min(2.0, v)))

    veto = clamp_int_0_100(max(50, 100 - risk_tolerance))
    hold_r = clamp_int_0_100(max(40, 90 - risk_tolerance))
    hold_p = clamp_int_0_100(max(40, 90 - reputation_focus))

    return PersonaConfig(
        name=payload["key"],
        w_cost=clamp(w_cost),
        w_risk=clamp(w_risk),
        w_urgency=clamp(w_urg),
        w_public_impact=clamp(w_pub),
        yes_threshold=10.0,
        no_threshold=-10.0,
        veto_risk_at=veto,
        hold_risk_at=hold_r,
        hold_public_impact_at=hold_p,
        style_note=payload["style_note"],
    )

# ===== 履歴（C）ユーティリティ =====
def _push_history(entry: dict, limit: int = 30):
    # 新しい順に保存
    st.session_state.history.insert(0, entry)
    # 上限超えたら古いのを落とす
    if len(st.session_state.history) > limit:
        st.session_state.history = st.session_state.history[:limit]


def ai_generate_persona(payload: dict) -> Tuple[PersonaConfig, dict]:
    """
    AIがJSONを返せなかったときは手動生成にフォールバックする安全設計。
    """
    key = payload["key"]
    prompt = f"""
あなたは意思決定人格の設計者です。以下の回答から人格パラメータを決めてください。
出力は必ず JSON のみ（余計な文章なし）で、キーは次の通りにしてください：
name,w_cost,w_risk,w_urgency,w_public_impact,yes_threshold,no_threshold,veto_risk_at,hold_risk_at,hold_public_impact_at,style_note

【人格キー】{key}
【表示名】{payload["display_name"]}
【最優先】{payload["priority"]}
【リスク許容(0-100)】{payload["risk_tolerance"]}
【コスト厳しさ(0-100)】{payload["cost_sensitivity"]}
【納期重視(0-100)】{payload["deadline_focus"]}
【評判重視(0-100)】{payload["reputation_focus"]}
【拒否権】{payload["veto_rule"]}
【HOLD条件】{payload["hold_rule"]}
【口調】{payload["tone"]}
【方針メモ】{payload["style_note"]}
【質問例】{payload["example_questions"]}
""".strip()

    # Secrets からAPIキーが入っていれば動く
    api_key = st.secrets.get("OPENAI_API_KEY", None)
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in Streamlit Secrets")

    client = OpenAI(api_key=api_key)

    # model名は必要に応じて変えてください（まずは軽量でOK）
    resp = client.responses.create(
        model=st.secrets.get("OPENAI_MODEL", "gpt-5-mini"),
        input=[{"role": "user", "content": prompt}],
    )
    raw = getattr(resp, "output_text", None) or str(resp)

    # JSONとしてパースできない場合はフォールバック
    data = json.loads(raw)
    cfg = PersonaConfig(**data)
    return cfg, data


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="MAGI風 合議システム", layout="wide")
st.title("MAGI風 合議システム（Web / Streamlit Cloud）")

if "personas" not in st.session_state:
    st.session_state.personas = default_personas()

personas: Dict[str, PersonaConfig] = st.session_state.personas

tab1, tab2 = st.tabs(["合議（投票）", "人格編集・生成"])
if "personas" not in st.session_state:
    st.session_state.personas = default_personas()

# ===== 履歴（C）初期化 =====
if "history" not in st.session_state:
    st.session_state.history = []  # 新しい順に貯める

# ---- Tab1: Evaluate ----
with tab1:
    colL, colR = st.columns([1, 1], gap="large")

    with colL:
        st.subheader("提案入力")
        title = st.text_input("タイトル", value="例：社内AIチャット導入")
        desc = st.text_area("説明", value="例：問い合わせ対応を自動化して工数削減する", height=90)

        c1, c2 = st.columns(2)
        with c1:
            cost = st.slider("cost（コスト）", 0, 100, 60)
            risk = st.slider("risk（リスク）", 0, 100, 50)
        with c2:
            urgency = st.slider("urgency（緊急性）", 0, 100, 55)
            public_impact = st.slider("public_impact（評判/社会影響）", 0, 100, 40)

        hold_priority = st.toggle("HOLD優先モード", value=True)

        # ボタン類（←この行頭のスペースが重要：with colL の中に置く）
        run = st.button("判定（投票）", type="primary")

        # --- クールダウン（連打防止）---
        COOLDOWN_SEC = 8  # 5〜15くらいで好み

        if "last_debate_ts" not in st.session_state:
            st.session_state.last_debate_ts = 0.0

        now = time.time()
        remain = COOLDOWN_SEC - (now - st.session_state.last_debate_ts)
        cooling = remain > 0

        debate = st.button("議論する（AI同士）", disabled=cooling)
        if cooling:
            st.info(f"連続実行を防ぐため、あと {remain:.0f} 秒待ってください。")

        rounds = st.slider("議論ラウンド数", 1, 2, 2)

        if debate:
            st.session_state.last_debate_ts = time.time()

    

    with colR:
        st.subheader("結果")

        # run / debate のどちらか押されたら共通の材料を作る
        if run or debate:
            votes_only: Dict[str, Vote] = {}
            details: Dict[str, Any] = {}

            for key, cfg in personas.items():
                v, reason, score, breakdown = score_vote(cfg, cost, risk, urgency, public_impact)
                votes_only[key] = v
                details[key] = {
                    "vote": v.value,
                    "reason": reason,
                    "score": score,
                    "breakdown": breakdown,
                    "style_note": cfg.style_note,
                }

                # UI：投票結果 / 議論ログ をタブで分ける
        result_tab, debate_tab = st.tabs(["投票結果", "議論ログ"])


        # ===== 履歴表示（右側の下）=====
        st.divider()
        st.subheader("履歴（最新）")
        
        hist = st.session_state.history
        if not hist:
            st.info("まだ履歴がありません。『判定（投票）』または『議論する（AI同士）』を実行すると貯まります。")
        else:
            # 一覧（最新10件）
            shown = hist[:10]
            labels = []
            for i, h in enumerate(shown):
                kind = "投票" if h["type"] == "vote" else "議論"
                labels.append(f"{i+1}. [{kind}] {h['title']}")
        
            idx = st.selectbox("履歴を選択", options=list(range(len(shown))), format_func=lambda i: labels[i])
            sel = shown[idx]
        
            kind = "投票" if sel["type"] == "vote" else "議論"
            st.markdown(f"**種類**：{kind}")
            st.markdown(f"**タイトル**：{sel['title']}")
            st.write(sel["description"])
            st.write("入力値:", sel["inputs"])
        
            if sel["type"] == "vote":
                st.success(f"FINAL: {sel['final']}")
                for key, r in sel["details"].items():
                    with st.container(border=True):
                        st.markdown(f"**{key}** → **{r['vote']}**")
                        st.write(r["reason"])
            else:
                st.success(f"FINAL（議論後）: {sel['final_after']}")
                st.write("最終投票:", sel["votes_after"])
                if sel.get("chair_summary"):
                    st.markdown("### 議長サマリー")
                    st.write(sel["chair_summary"])
                with st.expander("議論ログを表示"):
                    for item in sel["debate_log"]:
                        st.markdown(f"**[{item['speaker']}]**")
                        st.write(item["content"])
        
            st.download_button(
                "この履歴をJSONでダウンロード",
                data=json.dumps(sel, ensure_ascii=False, indent=2),
                file_name="magi_history_item.json",
                mime="application/json",
            )
        
            if st.button("履歴を全てクリア"):
                st.session_state.history = []
                st.success("履歴をクリアしました。")
                st.rerun()

        with result_tab:
            if run:
                final = council_decide(votes_only, hold_priority=hold_priority)
                if final == Vote.YES:
                    st.success(f"FINAL: {final.value}")
                elif final == Vote.NO:
                    st.error(f"FINAL: {final.value}")
                else:
                    st.warning(f"FINAL: {final.value}")

                st.markdown("### 各人格の投票（カード表示）")
                for key, r in details.items():
                    with st.container(border=True):
                        st.markdown(f"**{key}**　→　**{r['vote']}**")

                        short_reason = r["reason"]
                        if len(short_reason) > 180:
                            short_reason = short_reason[:180] + "…"
                        st.write(short_reason)

                        with st.expander("詳細（内訳・方針・スコア）"):
                            st.write(f"方針: {r['style_note']}")
                            st.write(f"スコア: {r['score']:.1f}")
                            st.json(r["breakdown"])

                result_obj = {"final": final.value, "details": details}
                st.download_button(
                    "結果をJSONでダウンロード（必要なときだけ）",
                    data=json.dumps(result_obj, ensure_ascii=False, indent=2),
                    file_name="magi_result.json",
                    mime="application/json",
                )
                # ===== 履歴に保存（投票）=====
                _push_history({
                    "type": "vote",
                    "ts": time.time(),
                    "title": title,
                    "description": desc,
                    "inputs": {"cost": cost, "risk": risk, "urgency": urgency, "public_impact": public_impact},
                    "final": final.value,
                    "details": details,
                })
            else:
                st.info("左で入力して『判定（投票）』を押すと結果が出ます。")

        with debate_tab:
            if debate:
                proposal_obj = {
                    "title": title,
                    "description": desc,
                    "cost": cost,
                    "risk": risk,
                    "urgency": urgency,
                    "public_impact": public_impact,
                }

                debate_log, votes_after, chair_summary = build_debate_log(
                    personas, proposal_obj, details, rounds=rounds
                )

                st.markdown("### 議論ログ（MAGI風）")
                for item in debate_log:
                    speaker = item["speaker"]
                    content = item["content"]
                    if speaker == "SYSTEM":
                        st.info(content)
                    else:
                        with st.chat_message("assistant", avatar="🤖"):
                            st.markdown(f"**{speaker}**")
                            st.write(content)

                final_after = council_decide(
                    {k: Vote(v) for k, v in votes_after.items()},
                    hold_priority=hold_priority
                )

                st.markdown("### 議論後の合議（参考）")
                if final_after == Vote.YES:
                    st.success(f"FINAL: {final_after.value}")
                elif final_after == Vote.NO:
                    st.error(f"FINAL: {final_after.value}")
                else:
                    st.warning(f"FINAL: {final_after.value}")

                if chair_summary:
                    st.markdown("### 議長サマリー（要点）")
                    st.write(chair_summary)

                # ===== 履歴に保存（議論）=====
                _push_history({
                    "type": "debate",
                    "ts": time.time(),
                    "title": title,
                    "description": desc,
                    "inputs": {"cost": cost, "risk": risk, "urgency": urgency, "public_impact": public_impact, "rounds": rounds},
                    "final_after": final_after.value,
                    "votes_after": votes_after,
                    "chair_summary": chair_summary,
                    "debate_log": debate_log,
                })

            else:
                st.info("左で入力して『議論する（AI同士）』を押すと議論ログが出ます。")

    
# ---- Tab2: Persona edit + generator ----
with tab2:
    st.subheader("人格編集")
    colA, colB = st.columns([1, 1], gap="large")

    with colA:
        keys = list(personas.keys())
        selected = st.selectbox("編集する人格", keys)
        cfg = personas[selected]

        cfg.style_note = st.text_input("方針メモ", value=cfg.style_note)

        st.markdown("**重み（w）**：正→押し上げ、負→押し下げ")
        w1, w2 = st.columns(2)
        with w1:
            cfg.w_cost = st.slider("w_cost", -2.0, 2.0, float(cfg.w_cost), 0.1)
            cfg.w_risk = st.slider("w_risk", -2.0, 2.0, float(cfg.w_risk), 0.1)
        with w2:
            cfg.w_urgency = st.slider("w_urgency", -2.0, 2.0, float(cfg.w_urgency), 0.1)
            cfg.w_public_impact = st.slider("w_public_impact", -2.0, 2.0, float(cfg.w_public_impact), 0.1)

        th1, th2 = st.columns(2)
        with th1:
            cfg.yes_threshold = st.number_input("YES閾値", value=float(cfg.yes_threshold), step=1.0)
        with th2:
            cfg.no_threshold = st.number_input("NO閾値", value=float(cfg.no_threshold), step=1.0)

        st.markdown("**強制ルール**")
        r1, r2, r3 = st.columns(3)
        with r1:
            cfg.veto_risk_at = st.slider("veto_risk_at（risk>=で即NO）", 0, 100, int(cfg.veto_risk_at))
        with r2:
            cfg.hold_risk_at = st.slider("hold_risk_at（risk>=でHOLD）", 0, 100, int(cfg.hold_risk_at))
        with r3:
            cfg.hold_public_impact_at = st.slider("hold_public_impact_at（影響>=でHOLD）", 0, 100, int(cfg.hold_public_impact_at))

        personas[selected] = cfg
        st.session_state.personas = personas

        cbtn1, cbtn2, cbtn3 = st.columns(3)
        with cbtn1:
            if st.button("新規人格を追加"):
                new_key = unique_key(personas, "NEW_PERSONA")
                personas[new_key] = PersonaConfig(
                    name=new_key,
                    w_cost=-0.5, w_risk=-0.5, w_urgency=0.5, w_public_impact=-0.5,
                    yes_threshold=10.0, no_threshold=-10.0,
                    veto_risk_at=95, hold_risk_at=70, hold_public_impact_at=70,
                    style_note="ここに方針を書いてください。",
                )
                st.session_state.personas = personas
                st.rerun()
        with cbtn2:
            if st.button("この人格を削除"):
                if len(personas) <= 1:
                    st.error("最後の1人格は削除できません。")
                else:
                    personas.pop(selected, None)
                    st.session_state.personas = personas
                    st.rerun()
        with cbtn3:
            st.download_button(
                "人格設定をJSONでダウンロード",
                data=json.dumps({k: asdict(v) for k, v in personas.items()}, ensure_ascii=False, indent=2),
                file_name="personas.json",
                mime="application/json",
            )

        st.write("JSONを貼り付けて復元（上書き）")
        json_text = st.text_area("personas.json の中身", value="", height=140, placeholder="ここにJSONを貼る")
        if st.button("JSONを読み込んで上書き"):
            try:
                loaded = json.loads(json_text)
                new_personas = {k: PersonaConfig(**v) for k, v in loaded.items()}
                st.session_state.personas = new_personas
                st.success("読み込み完了")
                st.rerun()
            except Exception as e:
                st.error(f"読み込み失敗: {e}")

    with colB:
        st.subheader("人格生成フォーム（質問→追加）")
        st.caption("OpenAI連携は任意です。SecretsにAPIキーが無ければ手動生成に自動フォールバックします。")

        gen_key = st.text_input("キー（例：CASPER2）", value="CASPER2")
        gen_display_name = st.text_input("表示名", value="CASPER2")

        gen_priority = st.selectbox(
            "最優先",
            options=["safety", "cost", "speed", "reputation", "balance"],
            format_func=lambda x: {
                "safety":"安全/倫理", "cost":"コスト/効率", "speed":"スピード/納期",
                "reputation":"評判/納得感", "balance":"バランス"
            }[x]
        )

        g1, g2 = st.columns(2)
        with g1:
            gen_risk_tol = st.slider("リスク許容(0-100)", 0, 100, 30)
            gen_cost_sen = st.slider("コスト厳しさ(0-100)", 0, 100, 60)
        with g2:
            gen_deadline = st.slider("納期重視(0-100)", 0, 100, 50)
            gen_reput = st.slider("評判重視(0-100)", 0, 100, 50)

        gen_veto = st.text_input("拒否権（即NO条件）", value="例：法令違反や個人情報の可能性があるなら即NO。")
        gen_hold = st.text_input("HOLD条件（追加情報要求）", value="例：情報不足ならHOLD。説明不足ならHOLD。")
        gen_tone = st.selectbox("口調", options=["formal", "neutral", "emotional"], format_func=lambda x: {"formal":"硬め","neutral":"普通","emotional":"感情的"}[x])
        gen_style = st.text_area("方針メモ", value="例：安全第一。曖昧なら追加情報を求める。", height=90)
        gen_q = st.text_area("質問例（改行区切り）", value="- 想定ユーザーと影響範囲は？\n- リスク対策は？", height=90)

        if st.button("生成→追加", type="primary"):
            payload = {
                "key": unique_key(personas, gen_key),
                "display_name": gen_display_name,
                "priority": gen_priority,
                "risk_tolerance": gen_risk_tol,
                "cost_sensitivity": gen_cost_sen,
                "deadline_focus": gen_deadline,
                "reputation_focus": gen_reput,
                "veto_rule": gen_veto,
                "hold_rule": gen_hold,
                "tone": gen_tone,
                "style_note": gen_style,
                "example_questions": gen_q,
            }

            # AI→失敗なら手動フォールバック
            used = "manual"
            raw_ai = None
            try:
                if OPENAI_AVAILABLE and st.secrets.get("OPENAI_API_KEY", None):
                    cfg_ai, raw_ai = ai_generate_persona(payload)
                    # 衝突回避（AIがnameを変に返してもキーを優先）
                    cfg_ai.name = payload["key"]
                    personas[cfg_ai.name] = cfg_ai
                    used = "ai"
                else:
                    cfg = manual_generate_persona(payload)
                    personas[cfg.name] = cfg
                    used = "manual"
            except Exception:
                cfg = manual_generate_persona(payload)
                personas[cfg.name] = cfg
                used = "manual"

            st.session_state.personas = personas
            st.success(f"人格を追加しました（mode={used}）: {payload['key']}")
            if raw_ai:
                st.markdown("AIが作ったJSON（参考）")
                st.code(json.dumps(raw_ai, ensure_ascii=False, indent=2), language="json")
            st.rerun()
