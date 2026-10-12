import json
import random
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

HERE = Path(__file__).resolve().parent
OUT = HERE.parent.parent / "personal_data" / "lines.jsonl"
OUT.parent.mkdir(parents=True, exist_ok=True)

st.set_page_config(layout="wide", page_title="Handwriting capture")
capture = components.declare_component("capture", path=str(HERE / "component"))

prompts = [l.strip() for l in (HERE / "prompts.txt").read_text().splitlines()
           if l.strip() and not l.startswith("#")]
random.Random(0).shuffle(prompts)              


def load_saved():
    if not OUT.exists():
        return []
    return [json.loads(l) for l in OUT.read_text().splitlines() if l.strip()]


if "nonce" not in st.session_state:
    st.session_state.nonce = 0
    st.session_state.last_id = None

saved = load_saved()
done = {r["text"] for r in saved}
todo = [p for p in prompts if p not in done]

st.title("Handwriting capture")
st.progress(len(done) / len(prompts), text=f"{len(done)} / {len(prompts)} lines saved")

if st.sidebar.button("Delete last saved line") and saved:
    OUT.write_text("".join(json.dumps(r) + "\n" for r in saved[:-1]))
    st.session_state.nonce += 1
    st.rerun()

if not todo:
    st.success("All lines captured. Back up personal_data/lines.jsonl")
    st.stop()

text = todo[0]
st.write("Write this line:")
st.header(text)

result = capture(width=900, height=240, key=f"cap_{st.session_state.nonce}", default=None)

if result and result.get("id") != st.session_state.last_id:
    st.session_state.last_id = result["id"]
    with OUT.open("a") as f:
        f.write(json.dumps({"text": text, **result}) + "\n")
    st.session_state.nonce += 1                
    st.rerun()
