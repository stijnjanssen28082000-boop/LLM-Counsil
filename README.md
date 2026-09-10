# LLM Council

Stel één vraag, en laat meerdere LLM's die onafhankelijk beantwoorden. De modellen
ranken elkaars antwoorden anoniem, en een "chairman"-model synthetiseert er één
definitief eindantwoord uit. Geïnspireerd op [Karpathy's LLM Council](https://github.com/karpathy/llm-council).

Gebouwd als één Streamlit-app (`app.py` + `council.py`), zodat hij direct op
[Streamlit Community Cloud](https://share.streamlit.io) te hosten is.

## Hoe het werkt

1. **Stage 1 — Antwoorden**: elk geselecteerd model beantwoordt de vraag onafhankelijk (parallel).
2. **Stage 2 — Ranking**: de antwoorden worden geanonimiseerd ("Response A", "Response B", ...) en elk
   model rankt ze van beste naar slechtste.
3. **Stage 3 — Chairman**: een gekozen model krijgt alle antwoorden + rankings te zien en schrijft het
   uiteindelijke antwoord.

Je kiest in de sidebar de modus: alleen antwoorden, antwoorden + ranking, of de volledige 3 stages.

Modellen worden aangeroepen via [OpenRouter](https://openrouter.ai), dat één API biedt voor
OpenAI-, Anthropic-, Google-, Meta- en vele andere modellen.

## Lokaal draaien

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# vul je OpenRouter API key in .streamlit/secrets.toml in
# (of vul de key direct in de sidebar van de app in)

streamlit run app.py
```

Open daarna `http://localhost:8501`.

Een OpenRouter API key maak je aan op [openrouter.ai/keys](https://openrouter.ai/keys).

## Deployen en verbinden met Streamlit Community Cloud

1. Push deze repo naar GitHub (al gedaan als je dit leest via GitHub).
2. Ga naar [share.streamlit.io](https://share.streamlit.io) en log in met je GitHub-account.
3. Klik **New app**, kies deze repository en branch, en zet **Main file path** op `app.py`.
4. Ga in de app-instellingen naar **Secrets** en voeg toe:
   ```toml
   OPENROUTER_API_KEY = "sk-or-v1-..."
   ```
5. Deploy. Streamlit Cloud herbouwt de app automatisch bij elke nieuwe push naar de gekoppelde branch.

## Configuratie in de app

- **Council modellen**: lijst van OpenRouter model-ID's (één per regel). Standaard staan hier
  gratis (`:free`) modellen zodat je zonder credit kan testen; actuele gratis modellen vind je op
  [openrouter.ai/models?max_price=0](https://openrouter.ai/models?max_price=0) (limiet: ongeveer
  20 verzoeken/minuut en 50/dag zonder opgeladen credit).
- **Chairman model**: het model dat in Stage 3 het eindantwoord schrijft.
- **Modus**: `chat_only`, `chat_ranking` of `full`.
- **System prompt**: optioneel, wordt aan alle council-modellen meegegeven.

## Bestanden

| Bestand | Functie |
|---|---|
| `app.py` | Streamlit UI: sidebar-instellingen, chatgeschiedenis, resultaten per stage |
| `council.py` | De 3-stage orkestratie-logica en OpenRouter-aanroepen |
| `requirements.txt` | Python-dependencies |
| `.streamlit/secrets.toml.example` | Template voor lokale secrets |

## Licentie

MIT.
