"""Deterministic copilot router: maps an operator's question to an answer grounded in the
pipeline's detected events. No external LLM needed; unknown questions get a polite redirect."""
import re

SUGGESTIONS = ["Summarise all high severity anomalies", "Why was WTG_07 flagged?",
               "What should I check on the solar inverter?", "What are the likely causes on wind turbine WTG_07?",
               "How does the detection work?", "How accurate is it?"]

_GREET = re.compile(r"^\s*(hi+|hello+|hey+|hola|namaste|yo|good\s+(morning|afternoon|evening))\b[\s,!.]*", re.I)
_SOCIAL_REST = re.compile(r"^(there|copilot|team|everyone)?[\s,!.?]*(how are you( doing)?|how'?s it going|what'?s up|how do you do)?[\s,!.?]*$", re.I)
_HOWAREYOU = re.compile(r"^\s*(how are you( doing)?|how'?s it going|what'?s up)[\s!.?]*$", re.I)
_THANKS = re.compile(r"^\s*(thanks|thank you|thx|ty|cheers|great|awesome|nice)\b", re.I)
_BYE = re.compile(r"^\s*(bye|goodbye|see you|cya)\b", re.I)
_ABOUT_ME = re.compile(r"who are you|what are you|your name|what can you do|what do you do|capabilit|^\s*help\b", re.I)
_METHOD = re.compile(r"how (does|do|did) (it|this|the|you)\b|how .*work|(which|what) (model|method|algorithm)|isolation ?forest|approach|methodolog|architecture", re.I)
_EVAL = re.compile(r"accura|metric|evaluat|precision|recall|\bf1\b|how good|reliab|limitation|weakness|fail", re.I)
_ANOM = re.compile(r"anomal|fault|flag|event|issue|problem|abnormal|alert|alarm|\bwhy\b|explain|summar|status|health|check|inspect|next|action|cause|reason|severity|vibrat|temperat|underperform|power|loss|lost|worst|critical|urgent|priority|what happened|happened|wrong|\bok(ay)?\b|normal|fine|solar|wind|inverter|turbine", re.I)


def _short(a, n=3):
    return a[:n]


def _fmt_checks(sel):
    out = ["Recommended next checks:"]
    for e in sel:
        x = e["explanation"]
        out.append(f"\n{e['asset_id']} ({x['when']}):")
        out += [f"  • {c}" for c in _short(x["next_checks"], 4)]
        out.append(f"  ⚠ {x['safety_note']}")
    return "\n".join(out)


def _fmt_causes(sel):
    out = ["Likely causes (ranked by playbook match):"]
    for e in sel:
        x = e["explanation"]
        out.append(f"\n{e['asset_id']} ({x['when']}):")
        out += [f"  • {c}" for c in _short(x["likely_causes"], 4)]
    out.append("\nThese are hypotheses to verify on site, not confirmed diagnoses.")
    return "\n".join(out)


def _fmt_why(sel):
    out = []
    for e in sel:
        x = e["explanation"]
        out.append(f"{x['summary']}\nFlagged because: " + "; ".join(x["why_flagged"]) +
                   f".\nMost likely: {'; '.join(_short(x['likely_causes'], 2))}.")
    return "\n\n".join(out)


def _method():
    return ("How detection works:\n"
            "1. Each asset's power is compared with what its weather should produce (irradiance for solar, wind speed for wind); "
            "temperature is compared with that asset's normal for the hour of day.\n"
            "2. Robust z-score rules flag high vibration, over-temperature, underperformance, night generation, flatlined sensors and bad data.\n"
            "3. An IsolationForest model per asset catches unusual combinations of readings.\n"
            "4. Flags are merged into events, given a severity, matched to troubleshooting playbooks, and explained in plain language.")


def _eval(evaluation):
    lines = ["Accuracy (synthetic fault-injection test, not field data):"]
    try:
        for lv, d in evaluation["injection_benchmark"].items():
            e = d["ensemble"]
            lines.append(f"  • {lv} faults: precision {e['precision']}, recall {e['recall']}, F1 {e['f1']}")
    except Exception:
        lines.append("  • Run `python -m backend.evaluate` to generate results.")
    lines.append("Limitations: small synthetic dataset, subtle or slow drift is often missed, curtailment can look like underperformance, "
                 "and causes are playbook hypotheses to verify on site.")
    return "\n".join(lines)


def answer(text, assets, events, evaluation=None):
    t = (text or "").strip()
    base = dict(input=text, events=[], suggestions=[], limitations=None)
    if not t:
        return dict(base, intent="empty", answer="Type a question about the detected anomalies, or pick a suggestion.",
                    reasoning_steps=["No text received"], suggestions=SUGGESTIONS[:3])

    # --- small talk
    g = _GREET.match(t)
    if g and _SOCIAL_REST.match(t[g.end():]):
        how = "I'm doing well, thanks for asking. " if re.search(r"how|what'?s up", t, re.I) else ""
        return dict(base, intent="greeting", answer=f"Hello! {how}I'm the anomaly copilot for this telemetry. I can tell you what was flagged, why, "
                    "the likely causes and what to check next. What would you like to know?",
                    reasoning_steps=["Recognised a greeting"], suggestions=SUGGESTIONS[:4])
    if _HOWAREYOU.match(t):
        return dict(base, intent="smalltalk", answer="I'm doing well, thanks! Ask me about any anomalies in the solar or wind data.",
                    reasoning_steps=["Recognised small talk"], suggestions=SUGGESTIONS[:3])
    if _THANKS.match(t):
        return dict(base, intent="thanks", answer="You're welcome! Anything else you'd like to check?", reasoning_steps=["Recognised thanks"], suggestions=SUGGESTIONS[:3])
    if _BYE.match(t):
        return dict(base, intent="bye", answer="Goodbye! Stay safe on site, and follow LOTO procedures before any inspection.", reasoning_steps=["Recognised goodbye"])
    if _ABOUT_ME.search(t):
        return dict(base, intent="about", answer="I explain anomalies found in solar-inverter and wind-turbine telemetry. Ask me: which assets were flagged and how severe, "
                    "why something was flagged, likely causes, what to check next, how the detection works, or how accurate it is. "
                    "I only use the data in this app, so I can't answer general questions.", reasoning_steps=["Recognised a capabilities question"], suggestions=SUGGESTIONS)

    # --- how it works / accuracy
    if _METHOD.search(t):
        return dict(base, intent="method", answer=_method(), reasoning_steps=["Recognised a how-it-works question"], suggestions=SUGGESTIONS[4:6])
    if _EVAL.search(t) and not re.search(r"flag|event|asset", t, re.I):
        return dict(base, intent="evaluation", answer=_eval(evaluation), reasoning_steps=["Read stored evaluation results"], suggestions=SUGGESTIONS[:2])

    # --- anomaly questions
    mentioned = [a for a in assets if a.lower() in t.lower()]
    if not (mentioned or _ANOM.search(t)):
        return dict(base, intent="out_of_scope", answer="I'm built to answer questions about the anomalies in this solar and wind telemetry, so I can't help with that one. "
                    "Here are some things I can answer:", reasoning_steps=["No anomaly-related intent found"], suggestions=SUGGESTIONS)

    steps = ["Ran detection pipeline (rules + IsolationForest) on bundled telemetry"]
    sel = events
    if mentioned:
        sel = [e for e in sel if e["asset_id"] in mentioned]; steps.append(f"Filtered to asset(s): {', '.join(mentioned)}")
    else:
        solar, wind = re.search(r"solar|inverter", t, re.I), re.search(r"wind|turbine", t, re.I)
        if solar and not wind:
            sel = [e for e in sel if e["asset_type"] == "solar"]; steps.append("Filtered to solar assets")
        elif wind and not solar:
            sel = [e for e in sel if e["asset_type"] == "wind"]; steps.append("Filtered to wind assets")
    m = re.search(r"\b(high|medium|low)\b", t, re.I)
    if m:
        sel = [e for e in sel if e["severity"] == m.group(1).lower()]; steps.append(f"Filtered to {m.group(1).lower()} severity")
    if re.search(r"worst|priority|critical|urgent|most serious", t, re.I):
        sel = sel[:1]; steps.append("Selected the highest-priority event")
    if not sel:
        return dict(base, intent="anomaly", answer="No matching anomalies were found for that selection.", reasoning_steps=steps, suggestions=SUGGESTIONS[:2])

    if re.search(r"check|inspect|next|action|should i|what do i do|fix", t, re.I):
        steps.append("Retrieved troubleshooting playbooks for each event"); ans, intent = _fmt_checks(sel), "next_checks"
    elif re.search(r"cause|reason|likely", t, re.I):
        steps.append("Listed playbook-ranked likely causes"); ans, intent = _fmt_causes(sel), "causes"
    elif re.search(r"\bwhy\b|explain|flag", t, re.I):
        steps.append("Explained which rules and signals triggered the flag"); ans, intent = _fmt_why(sel), "why"
    else:
        ans, intent = f"{len(sel)} anomaly event(s) found.\n\n" + "\n\n".join(e["explanation"]["summary"] for e in sel), "summary"
    return dict(base, intent=intent, answer=ans, events=sel, reasoning_steps=steps,
                limitations="Rule/statistical detection on a small synthetic dataset; verify on site before acting.")