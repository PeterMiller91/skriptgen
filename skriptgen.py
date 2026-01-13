import os
import json
import math
import time
import sqlite3
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional

import streamlit as st

# ----------------------------
# Optional: OpenAI (openai>=1.x)
# ----------------------------
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# ============================
# Config
# ============================
APP_TITLE = "ReelScripter Pro (MVP)"
DB_PATH = os.getenv("APP_DB_PATH", "reelscripter.db")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2")  # change if you want
MAX_SUBTOPICS = 10
MAX_IDEAS = 12

# Heuristics weights for OBSERVED performance score
W_SAVE_RATE = 0.60
W_VIEWS = 0.40


# ============================
# DB Helpers
# ============================
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = db()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        pw_hash TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS projects (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        name TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS generations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        project_id INTEGER,
        niche TEXT NOT NULL,
        subtopic TEXT NOT NULL,
        idea_title TEXT NOT NULL,
        idea_format TEXT,
        predicted_score INTEGER,
        predicted_rationale TEXT,
        hooks_json TEXT,
        script_json TEXT,
        created_at TEXT NOT NULL,
        is_favorite INTEGER NOT NULL DEFAULT 0,
        FOREIGN KEY(user_id) REFERENCES users(id),
        FOREIGN KEY(project_id) REFERENCES projects(id)
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS performance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        generation_id INTEGER NOT NULL,
        views INTEGER NOT NULL,
        saves INTEGER NOT NULL,
        notes TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY(generation_id) REFERENCES generations(id)
    )
    """)
    conn.commit()
    conn.close()

def hash_password(pw: str) -> str:
    # Simple salted hash (PBKDF2-like via sha256 iterations)
    # For production: use bcrypt/argon2. MVP okay.
    salt = os.getenv("APP_PW_SALT", "change-me-in-env")
    data = (salt + pw).encode("utf-8")
    h = hashlib.sha256(data).hexdigest()
    for _ in range(50_000):
        h = hashlib.sha256((h + salt).encode("utf-8")).hexdigest()
    return h

def create_user(email: str, pw: str) -> bool:
    conn = db()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO users (email, pw_hash, created_at) VALUES (?, ?, ?)",
            (email.lower().strip(), hash_password(pw), datetime.utcnow().isoformat())
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def authenticate(email: str, pw: str) -> Optional[Dict[str, Any]]:
    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE email = ?", (email.lower().strip(),))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    if row["pw_hash"] != hash_password(pw):
        return None
    return {"id": row["id"], "email": row["email"]}

def list_projects(user_id: int) -> List[Dict[str, Any]]:
    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM projects WHERE user_id = ? ORDER BY created_at DESC", (user_id,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows

def create_project(user_id: int, name: str) -> int:
    conn = db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO projects (user_id, name, created_at) VALUES (?, ?, ?)",
        (user_id, name.strip(), datetime.utcnow().isoformat())
    )
    conn.commit()
    pid = cur.lastrowid
    conn.close()
    return pid

def save_generation(
    user_id: int,
    project_id: Optional[int],
    niche: str,
    subtopic: str,
    idea_title: str,
    idea_format: str,
    predicted_score: int,
    predicted_rationale: str,
    hooks: Dict[str, Any],
    script: Dict[str, Any],
) -> int:
    conn = db()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO generations (
            user_id, project_id, niche, subtopic, idea_title, idea_format,
            predicted_score, predicted_rationale, hooks_json, script_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id, project_id, niche, subtopic, idea_title, idea_format,
        int(predicted_score), predicted_rationale,
        json.dumps(hooks, ensure_ascii=False),
        json.dumps(script, ensure_ascii=False),
        datetime.utcnow().isoformat()
    ))
    conn.commit()
    gid = cur.lastrowid
    conn.close()
    return gid

def toggle_favorite(gen_id: int, is_fav: bool):
    conn = db()
    cur = conn.cursor()
    cur.execute("UPDATE generations SET is_favorite = ? WHERE id = ?", (1 if is_fav else 0, gen_id))
    conn.commit()
    conn.close()

def add_performance(gen_id: int, views: int, saves: int, notes: str = ""):
    conn = db()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO performance (generation_id, views, saves, notes, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (gen_id, int(views), int(saves), notes.strip(), datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

def get_history(user_id: int, only_favs: bool = False, project_id: Optional[int] = None) -> List[Dict[str, Any]]:
    conn = db()
    cur = conn.cursor()
    q = "SELECT * FROM generations WHERE user_id = ?"
    params = [user_id]
    if only_favs:
        q += " AND is_favorite = 1"
    if project_id is not None:
        q += " AND project_id = ?"
        params.append(project_id)
    q += " ORDER BY created_at DESC LIMIT 200"
    cur.execute(q, tuple(params))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows

def get_generation(gen_id: int) -> Dict[str, Any]:
    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM generations WHERE id = ?", (gen_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else {}

def get_performance_rows(gen_id: int) -> List[Dict[str, Any]]:
    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM performance WHERE generation_id = ? ORDER BY created_at DESC", (gen_id,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


# ============================
# Scoring
# ============================
def observed_score(views: int, saves: int) -> int:
    v = max(int(views), 0)
    s = max(int(saves), 0)
    if v == 0 and s == 0:
        return 0

    save_rate = s / max(v, 1)
    # Map save_rate to 0..1 with saturation around 3%
    save_component = min(save_rate / 0.03, 1.0)

    # Views component log-scaled; 1k ≈ decent, 100k ≈ strong
    views_component = min(math.log10(max(v, 1)) / 5.0, 1.0)  # log10(100k)=5 -> 1.0
    score = (W_SAVE_RATE * save_component + W_VIEWS * views_component) * 100
    return int(round(score))


# ============================
# LLM helpers
# ============================
def openai_client() -> Optional[Any]:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key or OpenAI is None:
        return None
    return OpenAI(api_key=key)

def llm_json(prompt: str, schema_hint: str) -> Dict[str, Any]:
    """
    Returns JSON. If OpenAI key missing, returns a safe fallback stub.
    """
    client = openai_client()
    if client is None:
        return {"error": "OPENAI_API_KEY fehlt oder openai Paket nicht installiert.", "schema_hint": schema_hint}

    sys = (
        "Du bist ein deutscher Social-Media-Copywriter und Content-Stratege für Instagram Reels.\n"
        "Antworte ausschließlich als valides JSON ohne Markdown.\n"
        "Kein zusätzlicher Text.\n"
    )

    # Use responses API (openai>=1.x)
    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": sys},
            {"role": "user", "content": prompt},
        ],
        temperature=0.8,
    )
    text = resp.output_text.strip()
    try:
        return json.loads(text)
    except Exception:
        return {"error": "Konnte JSON nicht parsen.", "raw": text, "schema_hint": schema_hint}

def gen_subtopics(niche: str) -> List[str]:
    prompt = f"""
Erzeuge exakt {MAX_SUBTOPICS} unterschiedliche Oberbegriffe/Sub-Themen für die Nische: "{niche}".
Regeln:
- Deutsch
- Keine Duplikate
- Kurze Phrasen (max 6 Wörter)
Gib JSON im Format:
{{"subtopics":["...","..."]}}
"""
    data = llm_json(prompt, 'subtopics')
    subs = data.get("subtopics", [])
    return [s.strip() for s in subs if isinstance(s, str) and s.strip()][:MAX_SUBTOPICS]

def gen_ideas(niche: str, subtopic: str) -> List[Dict[str, Any]]:
    prompt = f"""
Für Instagram Reels.
Nische: "{niche}"
Subthema: "{subtopic}"

Erzeuge {MAX_IDEAS} konkrete Video-Ideen. Jede Idee muss enthalten:
- title: prägnanter Titel (max 10 Wörter)
- format: ein bewährtes Reel-Format (z.B. Myth Busting, 3 Fehler, POV, Before/After, Step-by-step, Storytime)
- predicted_score: 0-100 (Proxy auf Views+Saves)
- rationale: 1-2 Sätze warum
- improvements: genau 2 konkrete Optimierungen (bullet strings)

Gib JSON:
{{"ideas":[{{"title":"...","format":"...","predicted_score":77,"rationale":"...","improvements":["...","..."]}}]}}
"""
    data = llm_json(prompt, 'ideas')
    ideas = data.get("ideas", [])
    out = []
    for it in ideas:
        if not isinstance(it, dict):
            continue
        try:
            score = int(it.get("predicted_score", 0))
        except Exception:
            score = 0
        out.append({
            "title": str(it.get("title", "")).strip(),
            "format": str(it.get("format", "")).strip(),
            "predicted_score": max(0, min(score, 100)),
            "rationale": str(it.get("rationale", "")).strip(),
            "improvements": it.get("improvements", []) if isinstance(it.get("improvements", []), list) else []
        })
    # Sort by predicted_score desc
    out = [x for x in out if x["title"]]
    out.sort(key=lambda x: x["predicted_score"], reverse=True)
    return out[:MAX_IDEAS]

def gen_package(niche: str, subtopic: str, idea_title: str, idea_format: str) -> Dict[str, Any]:
    prompt = f"""
Erstelle ein drehfertiges Reel-Paket.
Kontext:
- Plattform: Instagram Reels
- Sprache: Deutsch
- Nische: "{niche}"
- Subthema: "{subtopic}"
- Idee: "{idea_title}"
- Format: "{idea_format}"

Liefere:
1) hooks: exakt 5 Varianten, je:
   - spoken (gesprochen)
   - onscreen (On-Screen Text, max 7 Wörter)
2) structure:
   - beats: Liste von Abschnitten mit time_range (z.B. "0-2s"), purpose, content
3) script:
   - voiceover: kompletter Text (kurze Sätze, natürlich)
   - onscreen_texts: Liste der Overlays in Reihenfolge
4) shotlist:
   - Liste: shot (z.B. "Talking head", "B-Roll"), description, duration_hint
5) caption:
   - caption_text (kurz)
   - hashtags (8-12, relevant, ohne Spam)
6) engagement:
   - pinned_comments: 3 Vorschläge (Frage, provokant aber fair, Call-to-save)

Gib JSON:
{{
 "hooks":[{{"spoken":"...","onscreen":"..."}}],
 "structure":{{"beats":[{{"time_range":"0-2s","purpose":"...","content":"..."}}]}},
 "script":{{"voiceover":"...","onscreen_texts":["..."]}},
 "shotlist":[{{"shot":"...","description":"...","duration_hint":"..."}}],
 "caption":{{"caption_text":"...","hashtags":["#..."]}},
 "engagement":{{"pinned_comments":["...","...","..."]}}
}}
"""
    return llm_json(prompt, 'package')


# ============================
# UI
# ============================
def require_login():
    if "user" not in st.session_state:
        st.session_state.user = None

    st.sidebar.header("Login")

    if st.session_state.user:
        st.sidebar.success(f"Eingeloggt: {st.session_state.user['email']}")
        if st.sidebar.button("Logout"):
            st.session_state.user = None
            st.rerun()
        return

    tab1, tab2 = st.sidebar.tabs(["Einloggen", "Registrieren"])

    with tab1:
        email = st.text_input("E-Mail", key="login_email")
        pw = st.text_input("Passwort", type="password", key="login_pw")
        if st.button("Login"):
            u = authenticate(email, pw)
            if u:
                st.session_state.user = u
                st.rerun()
            else:
                st.error("Login fehlgeschlagen.")

    with tab2:
        email = st.text_input("E-Mail ", key="reg_email")
        pw1 = st.text_input("Passwort ", type="password", key="reg_pw1")
        pw2 = st.text_input("Passwort bestätigen", type="password", key="reg_pw2")
        if st.button("Account erstellen"):
            if not email or "@" not in email:
                st.error("Bitte eine gültige E-Mail eingeben.")
            elif len(pw1) < 8:
                st.error("Passwort mindestens 8 Zeichen.")
            elif pw1 != pw2:
                st.error("Passwörter stimmen nicht überein.")
            else:
                ok = create_user(email, pw1)
                if ok:
                    st.success("Account erstellt. Bitte einloggen.")
                else:
                    st.error("E-Mail existiert bereits.")

def main_app():
    st.title(APP_TITLE)
    st.caption("MVP: kaskadierende Dropdowns → Ideenranking → Hooks/Skript → Historie/Projekte → manuelles Tracking (Views/Saves)")

    user = st.session_state.user
    user_id = user["id"]

    # Sidebar: Projects
    st.sidebar.divider()
    st.sidebar.subheader("Projekte")

    projects = list_projects(user_id)
    project_options = ["(Kein Projekt)"] + [p["name"] for p in projects]
    project_choice = st.sidebar.selectbox("Aktives Projekt", project_options)

    if st.sidebar.button("Neues Projekt"):
        name = st.sidebar.text_input("Projektname", key="new_project_name")
        st.sidebar.info("Tip: Namen eingeben und Enter drücken, dann nochmal klicken.")
        if name and name.strip():
            pid = create_project(user_id, name.strip())
            st.sidebar.success("Projekt erstellt.")
            st.rerun()

    project_id = None
    if project_choice != "(Kein Projekt)":
        # map name -> id (first match)
        for p in projects:
            if p["name"] == project_choice:
                project_id = p["id"]
                break

    tab_create, tab_history, tab_track = st.tabs(["Generator", "Historie", "Tracking"])

    # ----------------------------
    # Generator Tab
    # ----------------------------
    with tab_create:
        st.subheader("1) Kaskadierende Auswahl")

        niche = st.text_input("Nische (z.B. Fitness, Ernährung, Mindset, Beziehung)", value="Fitness")

        colA, colB = st.columns(2)
        with colA:
            if st.button("Subthemen generieren"):
                st.session_state.subtopics = gen_subtopics(niche)
                st.session_state.ideas = []
                st.session_state.selected_subtopic = None
                st.session_state.selected_idea = None

        subtopics = st.session_state.get("subtopics", [])
        if subtopics and isinstance(subtopics, list):
            selected_subtopic = st.selectbox("Dropdown 2: Subthema", subtopics)
            st.session_state.selected_subtopic = selected_subtopic

            with colB:
                if st.button("Videoideen generieren"):
                    st.session_state.ideas = gen_ideas(niche, selected_subtopic)
                    st.session_state.selected_idea = None

        ideas = st.session_state.get("ideas", [])
        if ideas and isinstance(ideas, list):
            st.subheader("2) Videoideen (gerankt)")
            # Show as selectbox with score
            labels = [f"{it['predicted_score']:>3}/100 • {it['title']}  ({it['format']})" for it in ideas]
            idx = st.selectbox("Dropdown 3: Idee wählen", range(len(labels)), format_func=lambda i: labels[i])
            selected = ideas[idx]
            st.session_state.selected_idea = selected

            st.markdown("**Warum dieser Score?**")
            st.write(selected.get("rationale", ""))
            imps = selected.get("improvements", [])
            if imps:
                st.markdown("**2 konkrete Verbesserungen:**")
                for x in imps[:2]:
                    st.write(f"- {x}")

            if st.button("Reel-Paket erzeugen (Hooks + Aufbau + Skript)"):
                pkg = gen_package(niche, st.session_state.selected_subtopic, selected["title"], selected["format"])
                if "error" in pkg:
                    st.error(pkg["error"])
                    st.code(pkg, language="json")
                else:
                    # Persist to DB
                    gid = save_generation(
                        user_id=user_id,
                        project_id=project_id,
                        niche=niche,
                        subtopic=st.session_state.selected_subtopic,
                        idea_title=selected["title"],
                        idea_format=selected["format"],
                        predicted_score=selected["predicted_score"],
                        predicted_rationale=selected.get("rationale", ""),
                        hooks={"hooks": pkg.get("hooks", [])},
                        script=pkg
                    )
                    st.session_state.last_gen_id = gid
                    st.success("Gespeichert in Historie.")
                    st.session_state.last_pkg = pkg

        # Render last package
        pkg = st.session_state.get("last_pkg")
        if pkg and isinstance(pkg, dict) and "error" not in pkg:
            st.subheader("3) Output (drehfertig)")
            st.markdown("### Hooks (5)")
            for i, h in enumerate(pkg.get("hooks", [])[:5], start=1):
                st.markdown(f"**Hook {i} – gesprochen:** {h.get('spoken','')}")
                st.markdown(f"**On-Screen:** {h.get('onscreen','')}")

            st.markdown("### Aufbau (Beats)")
            beats = pkg.get("structure", {}).get("beats", [])
            for b in beats:
                st.write(f"- **{b.get('time_range','')}** • {b.get('purpose','')}: {b.get('content','')}")

            st.markdown("### Skript (Voiceover)")
            st.text_area("Voiceover", value=pkg.get("script", {}).get("voiceover", ""), height=220)

            st.markdown("### On-Screen Texte")
            for t in pkg.get("script", {}).get("onscreen_texts", []):
                st.write(f"- {t}")

            st.markdown("### Shotlist")
            for s in pkg.get("shotlist", []):
                st.write(f"- **{s.get('shot','')}** ({s.get('duration_hint','')}): {s.get('description','')}")

            st.markdown("### Caption + Hashtags")
            st.write(pkg.get("caption", {}).get("caption_text", ""))
            st.write(" ".join(pkg.get("caption", {}).get("hashtags", [])))

            st.markdown("### Pinned Comments")
            for c in pkg.get("engagement", {}).get("pinned_comments", []):
                st.write(f"- {c}")

            gen_id = st.session_state.get("last_gen_id")
            if gen_id:
                fav = st.checkbox("Als Favorit markieren", value=False)
                if st.button("Favorit speichern"):
                    toggle_favorite(gen_id, fav)
                    st.success("Favorit-Status gespeichert.")

    # ----------------------------
    # History Tab
    # ----------------------------
    with tab_history:
        st.subheader("Historie & Favoriten")
        col1, col2 = st.columns([1, 1])
        with col1:
            only_favs = st.checkbox("Nur Favoriten", value=False)
        with col2:
            filter_project = st.checkbox("Nur aktives Projekt", value=False)

        pid = project_id if filter_project else None
        rows = get_history(user_id, only_favs=only_favs, project_id=pid)
        if not rows:
            st.info("Noch keine Einträge.")
        else:
            options = [f"#{r['id']} • {r['predicted_score']:>3}/100 • {r['idea_title']} ({r['subtopic']})" for r in rows]
            sel = st.selectbox("Eintrag auswählen", range(len(options)), format_func=lambda i: options[i])
            r = rows[sel]
            details = get_generation(r["id"])

            st.markdown(f"**Nische:** {details['niche']}  \n**Subthema:** {details['subtopic']}  \n**Format:** {details.get('idea_format','')}")
            st.markdown(f"**Predicted Score:** {details.get('predicted_score',0)}/100")
            st.write(details.get("predicted_rationale",""))

            hooks = json.loads(details.get("hooks_json") or "{}")
            script = json.loads(details.get("script_json") or "{}")

            with st.expander("Hooks anzeigen"):
                for i, h in enumerate(hooks.get("hooks", [])[:5], start=1):
                    st.write(f"{i}. {h.get('spoken','')}  |  On-screen: {h.get('onscreen','')}")

            with st.expander("Skript / Shotlist anzeigen"):
                st.text_area("Voiceover", value=script.get("script", {}).get("voiceover", ""), height=200)
                st.markdown("**Shotlist**")
                for s in script.get("shotlist", []):
                    st.write(f"- {s.get('shot','')}: {s.get('description','')}")

            st.divider()
            st.markdown("### Performance hinzufügen (manuell)")
            views = st.number_input("Views", min_value=0, value=0, step=100)
            saves = st.number_input("Saves", min_value=0, value=0, step=10)
            notes = st.text_input("Notizen (optional)")
            if st.button("Performance speichern"):
                add_performance(r["id"], views, saves, notes)
                st.success(f"Gespeichert. Observed Score: {observed_score(views, saves)}/100")
                st.rerun()

            perf = get_performance_rows(r["id"])
            if perf:
                st.markdown("### Performance-Historie")
                for p in perf[:20]:
                    sc = observed_score(p["views"], p["saves"])
                    st.write(f"- {p['created_at'][:19]} • Views {p['views']} • Saves {p['saves']} • Observed {sc}/100 • {p.get('notes','')}")

    # ----------------------------
    # Tracking Tab
    # ----------------------------
    with tab_track:
        st.subheader("Tracking (überblick)")
        rows = get_history(user_id, only_favs=False, project_id=project_id)
        if not rows:
            st.info("Noch keine Daten.")
        else:
            # For each generation, compute last observed score if available
            table = []
            for r in rows[:60]:
                perf = get_performance_rows(r["id"])
                last = perf[0] if perf else None
                obs = observed_score(last["views"], last["saves"]) if last else None
                table.append({
                    "ID": r["id"],
                    "Projekt": project_choice,
                    "Idee": r["idea_title"],
                    "Predicted": r["predicted_score"],
                    "Views": last["views"] if last else "",
                    "Saves": last["saves"] if last else "",
                    "Observed": obs if obs is not None else "",
                    "Datum": r["created_at"][:10],
                    "Fav": "★" if r["is_favorite"] else ""
                })

            st.dataframe(table, use_container_width=True)


# ============================
# App entry
# ============================
def run():
    st.set_page_config(page_title=APP_TITLE, page_icon="🎬", layout="wide")
    init_db()

    require_login()
    if st.session_state.user:
        main_app()
    else:
        st.title(APP_TITLE)
        st.info("Bitte einloggen oder registrieren (links in der Sidebar).")
        st.markdown("**Hinweis:** Für KI-Generierung brauchst du `OPENAI_API_KEY` als Environment Variable oder Streamlit Secret.")

if __name__ == "__main__":
    run()

