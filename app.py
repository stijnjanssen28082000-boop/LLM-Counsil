"""LLM Council — Streamlit app.

Ask one question, get independent answers from several LLMs, let them rank
each other anonymously, and have a chairman model synthesize a final answer.
"""

import streamlit as st

from council import run_council

DEFAULT_MODELS = [
    "openai/gpt-4o-mini",
    "anthropic/claude-3.5-sonnet",
    "google/gemini-2.0-flash-001",
    "meta-llama/llama-3.3-70b-instruct",
]

st.set_page_config(page_title="LLM Council", page_icon="🏛️", layout="wide")


def get_api_key() -> str:
    return st.session_state.get("api_key") or st.secrets.get("OPENROUTER_API_KEY", "")


with st.sidebar:
    st.header("Instellingen")

    st.text_input(
        "OpenRouter API key",
        type="password",
        value=st.secrets.get("OPENROUTER_API_KEY", ""),
        key="api_key",
        help="Nodig via https://openrouter.ai/keys. Kan ook als Streamlit secret "
        "'OPENROUTER_API_KEY' worden gezet zodat je 'm niet elke keer hoeft in te typen.",
    )

    models_text = st.text_area(
        "Council modellen (één per regel, OpenRouter model-ID's)",
        value="\n".join(DEFAULT_MODELS),
        height=120,
    )
    models = [m.strip() for m in models_text.splitlines() if m.strip()]

    chairman_model = st.selectbox(
        "Chairman model (schrijft het eindantwoord)",
        options=models or DEFAULT_MODELS,
        index=0,
    )

    execution_mode = st.radio(
        "Modus",
        options=["full", "chat_ranking", "chat_only"],
        format_func=lambda m: {
            "chat_only": "Alleen antwoorden",
            "chat_ranking": "Antwoorden + ranking",
            "full": "Antwoorden + ranking + chairman-synthese",
        }[m],
        index=0,
    )

    system_prompt = st.text_area(
        "System prompt (optioneel)",
        value="",
        height=80,
    )

    if st.button("Gesprek wissen"):
        st.session_state.pop("history", None)
        st.rerun()

st.title("🏛️ LLM Council")
st.caption(
    "Meerdere modellen beantwoorden je vraag onafhankelijk, ranken elkaars antwoorden "
    "anoniem, en een chairman-model stelt het eindantwoord samen."
)

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(turn["query"])
    with st.chat_message("assistant"):
        result = turn["result"]
        final = result.get("stage3")
        if final and not final.get("error"):
            st.markdown(final["response"])
        elif result["execution_mode"] != "full":
            st.info("Deze modus heeft geen chairman-synthese; zie de tabs hieronder voor de losse antwoorden.")
        else:
            st.error(final.get("response") if final else "Geen resultaat.")

        tabs = st.tabs(["Antwoorden per model", "Rankings", "Eindsynthese"])

        with tabs[0]:
            for r in result["stage1"]:
                with st.expander(r["model"], expanded=False):
                    if r.get("error"):
                        st.error(r.get("error_message", "Onbekende fout"))
                    else:
                        st.markdown(r["response"])

        with tabs[1]:
            if result.get("aggregate_rankings"):
                st.table(result["aggregate_rankings"])
            else:
                st.write("Geen rankings beschikbaar voor deze run.")
            if result.get("stage2"):
                for r in result["stage2"]:
                    with st.expander(f"Ranking door {r['model']}", expanded=False):
                        if r.get("error"):
                            st.error(r.get("error_message", "Onbekende fout"))
                        else:
                            st.markdown(r["ranking"])

        with tabs[2]:
            if final and not final.get("error"):
                st.markdown(f"**Chairman:** {final['model']}")
                st.markdown(final["response"])
            else:
                st.write("Geen chairman-synthese in deze run.")

query = st.chat_input("Stel je vraag aan de council...")

if query:
    api_key = get_api_key()
    if not api_key:
        st.error("Vul eerst een OpenRouter API key in de sidebar in.")
    elif not models:
        st.error("Selecteer minstens één council model in de sidebar.")
    else:
        with st.spinner("De council beraadslaagt..."):
            result = run_council(
                api_key=api_key,
                models=models,
                chairman_model=chairman_model,
                user_query=query,
                execution_mode=execution_mode,
                system_prompt=system_prompt or None,
            )
        st.session_state.history.append({"query": query, "result": result})
        st.rerun()
