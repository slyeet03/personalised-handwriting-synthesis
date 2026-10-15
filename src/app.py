import glob
import os

import streamlit as st

import render

st.set_page_config(page_title="Handwriting generator", layout="wide")


@st.cache_resource
def get_model():
    return render.load()


model, d, c2i, device = get_model()
prime_ids = [i for i, t in enumerate(d['texts']) if 8 <= len(str(t)) <= 14]

st.title("Handwriting generator")

with st.sidebar:
    prime = st.selectbox("Style sample", prime_ids, format_func=lambda i: str(d['texts'][i]))
    temp = st.slider("Temperature (lower = neater)", 0.15, 0.5, 0.25, 0.05)
    tries = st.slider("Tries per chunk", 1, 8, 4)
    max_chars = st.slider("Chunk length (characters)", 10, 24, 14)
    seed = st.number_input("Layout seed", 0, 9999, 0)

text = st.text_area("Text to write", height=220,
                    placeholder="Type or paste text. A blank line starts a new paragraph.")

if st.button("Generate", type="primary") and text.strip():
    with st.spinner("Writing... this takes a while"):
        items, dropped, failed = render.generate_lines(
            text, model, d, c2i, device, prime,
            tries=tries, temp=temp, max_chars=max_chars)
    st.session_state['chunks'] = items
    st.session_state['dropped'] = dropped
    st.session_state['failed'] = failed

if 'chunks' in st.session_state:
    if st.session_state.dropped:
        st.warning(f"Characters removed (not in the model's vocabulary): {st.session_state.dropped}")
    if st.session_state.failed:
        st.error(f"These lines failed and were left blank: {st.session_state.failed}")

    for f in glob.glob('../outputs/app_*'):
        os.remove(f)
    paths = render.render(st.session_state['chunks'], '../outputs/app', seed=int(seed))

    for pg in sorted(p for p in paths if p.endswith('.png')):
        col_img, col_dl = st.columns([4, 1])
        col_img.image(pg)
        with open(pg, 'rb') as f:
            col_dl.download_button("Download PNG", f.read(),
                                   file_name=os.path.basename(pg), key=pg)
        svg = pg.replace('.png', '.svg')
        with open(svg, 'rb') as f:
            col_dl.download_button("Download SVG", f.read(),
                                   file_name=os.path.basename(svg), key=svg)
