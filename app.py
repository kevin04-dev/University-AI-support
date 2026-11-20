import html
import json
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

# Import Streamlit before numpy/sklearn on purpose. The two are only needed when a question is
# actually asked, so a bare page load no longer pays for importing them.
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "support.db"
LEARNED_FAQ_PATH = BASE_DIR / "learned_faq.json"

INTENT_EXAMPLES = {
    "admissions": ["apply for admission", "application requirements", "enrollment"],
    "fees_and_finance": ["pay tuition fees", "payment deadline", "financial aid"],
    "student_portal": ["cannot log in to student portal", "reset my password", "portal account"],
    "examinations": ["exam timetable", "examination schedule", "results"],
    "registration": ["course registration", "register for classes", "unit enrollment"],
    "contacts": ["contact the university", "help desk", "office hours"],
}

FAQ = [
    {"intent": "admissions", "question": "How do I apply for admission?", "answer": "Check the university admissions page for current application dates, entry requirements, and the official application form. Contact the Admissions Office if you need help with a specific programme.", "keywords": ["application", "apply", "admission"]},
    {"intent": "fees_and_finance", "question": "How can I pay my fees?", "answer": "Use the payment methods and account details published by the university Finance Office. Confirm your student reference number and retain your payment receipt. Contact Finance to verify current deadlines and balances.", "keywords": ["fees", "tuition", "pay", "payment"]},
    {"intent": "student_portal", "question": "I cannot log in to the student portal", "answer": "Check that you are using your university username and the correct portal address. Use the password-reset option if available, or contact the IT Help Desk for account recovery.", "keywords": ["login", "log in", "password", "portal"]},
    {"intent": "examinations", "question": "Where can I find the exam timetable?", "answer": "Look for the latest examination timetable on the student portal or the university examinations page. If your course is missing or the schedule appears incorrect, contact your department or Examinations Office.", "keywords": ["exam", "examination", "timetable", "schedule"]},
    {"intent": "registration", "question": "How do I register for courses?", "answer": "Course registration is normally completed through the student portal during the registration period. Check with your academic department or Registrar for programme-specific requirements and deadlines.", "keywords": ["register", "registration", "courses", "units"]},
    {"intent": "contacts", "question": "How can I contact university support?", "answer": "Please use the contact details listed on the official university website or student portal so you reach the correct department and the latest support team.", "keywords": ["contact", "support", "help"]},
]

# ------------------------------------------------------------------ DATABASE

def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


@contextmanager
def db_connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _columns(conn, table):
    return {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}


def init_db():
    """Create tables and migrate older databases in place, keeping existing chats."""
    with db_connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL DEFAULT 'New chat',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                intent TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(conversation_id) REFERENCES conversations(id)
            );
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                rating TEXT NOT NULL,
                correction TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
        """)
        # Migrations for databases created by earlier versions of the app.
        conv_cols = _columns(conn, "conversations")
        if "owner_id" not in conv_cols:
            conn.execute("ALTER TABLE conversations ADD COLUMN owner_id TEXT")
        if "archived" not in conv_cols:
            conn.execute("ALTER TABLE conversations ADD COLUMN archived INTEGER NOT NULL DEFAULT 0")
        if "updated_at" not in conv_cols:
            conn.execute("ALTER TABLE conversations ADD COLUMN updated_at TEXT")
            conn.execute("UPDATE conversations SET updated_at = created_at WHERE updated_at IS NULL")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conversation_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_conv_owner ON conversations(owner_id, archived)")


def _owned(conn, conversation_id, owner_id):
    """Return the conversation only if it belongs to this user, else None.

    Every read and write goes through here, so one visitor can never open, rename,
    archive or delete another visitor's conversation by guessing an id."""
    if conversation_id is None or not owner_id:
        return None
    return conn.execute(
        "SELECT * FROM conversations WHERE id=? AND owner_id=?", (conversation_id, owner_id)
    ).fetchone()


def get_conversations(owner_id, archived=0, include_empty=False):
    """History for this user. Untouched 'New chat' shells are hidden unless asked for,
    so a first visit never litters the list with empty conversations."""
    with db_connect() as conn:
        rows = conn.execute(
            "SELECT c.* FROM conversations c "
            "WHERE c.owner_id=? AND c.archived=? "
            + ("" if include_empty else
               "AND (c.title <> 'New chat' OR EXISTS "
               "(SELECT 1 FROM messages m WHERE m.conversation_id = c.id)) ")
            + "ORDER BY COALESCE(c.updated_at, c.created_at) DESC, c.id DESC",
            (owner_id, archived),
        ).fetchall()
        return [dict(row) for row in rows]


def get_conversation(conversation_id, owner_id):
    with db_connect() as conn:
        row = _owned(conn, conversation_id, owner_id)
        return dict(row) if row else None


def reusable_draft(owner_id):
    """An existing empty conversation, if there is one, to avoid stacking up blank chats."""
    with db_connect() as conn:
        row = conn.execute(
            "SELECT c.id FROM conversations c WHERE c.owner_id=? AND c.archived=0 "
            "AND NOT EXISTS (SELECT 1 FROM messages m WHERE m.conversation_id=c.id) "
            "ORDER BY c.id DESC LIMIT 1",
            (owner_id,),
        ).fetchone()
        return row["id"] if row else None


def create_conversation(owner_id):
    with db_connect() as conn:
        cursor = conn.execute(
            "INSERT INTO conversations (title, owner_id, archived, updated_at) VALUES (?, ?, 0, ?)",
            ("New chat", owner_id, _now()),
        )
        return cursor.lastrowid


def get_messages(conversation_id, owner_id):
    with db_connect() as conn:
        if not _owned(conn, conversation_id, owner_id):
            return []
        return [dict(r) for r in conn.execute(
            "SELECT * FROM messages WHERE conversation_id=? ORDER BY id", (conversation_id,)
        ).fetchall()]


@st.cache_data(ttl=5, show_spinner=False)
def cached_messages(conversation_id, owner_id, version=0):
    """Messages for one chat. version is bumped after every write, so a rerun never shows
    a stale transcript. Reads are a single indexed SQLite query."""
    return get_messages(conversation_id, owner_id)


@st.cache_data(ttl=5, show_spinner=False)
def cached_conversations(owner_id, archived=0, version=0):
    return get_conversations(owner_id, archived)


def save_message(conversation_id, owner_id, role, content, intent=None):
    with db_connect() as conn:
        if not _owned(conn, conversation_id, owner_id):
            raise PermissionError("Conversation not found")
        cursor = conn.execute(
            "INSERT INTO messages (conversation_id, role, content, intent, created_at) VALUES (?, ?, ?, ?, ?)",
            (conversation_id, role, content, intent, _now()),
        )
        conn.execute("UPDATE conversations SET updated_at=? WHERE id=?", (_now(), conversation_id))
        return cursor.lastrowid


def delete_message(message_id, owner_id):
    """Delete one message, scoped to a conversation the caller owns."""
    with db_connect() as conn:
        row = conn.execute(
            "SELECT m.conversation_id FROM messages m JOIN conversations c "
            "ON c.id = m.conversation_id WHERE m.id=? AND c.owner_id=?",
            (message_id, owner_id),
        ).fetchone()
        if not row:
            return False
        conn.execute("DELETE FROM messages WHERE id=?", (message_id,))
        conn.execute("UPDATE conversations SET updated_at=? WHERE id=?", (_now(), row["conversation_id"]))
        return True


def update_message(message_id, owner_id, content, intent=None):
    with db_connect() as conn:
        row = conn.execute(
            "SELECT m.conversation_id FROM messages m JOIN conversations c "
            "ON c.id = m.conversation_id WHERE m.id=? AND c.owner_id=?",
            (message_id, owner_id),
        ).fetchone()
        if not row:
            return False
        conn.execute(
            "UPDATE messages SET content=?, intent=?, created_at=? WHERE id=?",
            (content, intent, _now(), message_id),
        )
        conn.execute("UPDATE conversations SET updated_at=? WHERE id=?", (_now(), row["conversation_id"]))
        return True


def rename_conversation(conversation_id, owner_id, title):
    title = " ".join((title or "").split())[:60]
    if not title:
        return False
    with db_connect() as conn:
        if not _owned(conn, conversation_id, owner_id):
            return False
        conn.execute("UPDATE conversations SET title=? WHERE id=?", (title, conversation_id))
        return True


def set_archived(conversation_id, owner_id, archived):
    with db_connect() as conn:
        if not _owned(conn, conversation_id, owner_id):
            return False
        conn.execute("UPDATE conversations SET archived=? WHERE id=?", (1 if archived else 0, conversation_id))
        return True


def delete_conversation(conversation_id, owner_id):
    """Permanently delete a conversation and every message attached to it."""
    with db_connect() as conn:
        if not _owned(conn, conversation_id, owner_id):
            return False
        # messages has no ON DELETE CASCADE in older databases, so remove them explicitly.
        conn.execute("DELETE FROM messages WHERE conversation_id=?", (conversation_id,))
        conn.execute("DELETE FROM conversations WHERE id=?", (conversation_id,))
        return True


def save_feedback(conversation_id, question, answer, rating, correction=None):
    with db_connect() as conn:
        conn.execute(
            "INSERT INTO feedback (conversation_id, question, answer, rating, correction) VALUES (?, ?, ?, ?, ?)",
            (conversation_id, question, answer, rating, correction),
        )


def load_learned_faq():
    try:
        with LEARNED_FAQ_PATH.open(encoding="utf-8") as file:
            entries = json.load(file)
        return entries if isinstance(entries, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def save_learned_faq(entries):
    with LEARNED_FAQ_PATH.open("w", encoding="utf-8") as file:
        json.dump(entries, file, ensure_ascii=False, indent=2)


# ------------------------------------------------------------------ NLP

def retrieve_answer(question):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    faq_items = FAQ + load_learned_faq()
    documents = [
        f"{item['question']} {' '.join(item.get('keywords', []))}"
        for item in faq_items
    ]
    # Fit the vectorizer on the knowledge base only, then transform the question separately.
    # Refitting over the question on every call was the single slowest line in the app.
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    matrix = vectorizer.fit_transform(documents)
    scores = cosine_similarity(vectorizer.transform([question]), matrix).ravel()
    best_item = faq_items[int(scores.argmax())]
    score = float(scores.max())

    intent_documents = [
        (example, intent)
        for intent, examples in INTENT_EXAMPLES.items()
        for example in examples
    ]
    intent_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    intent_matrix = intent_vectorizer.fit_transform([example for example, _ in intent_documents])
    intent_scores = cosine_similarity(
        intent_vectorizer.transform([question]), intent_matrix
    ).ravel()
    intent_index = int(intent_scores.argmax())
    intent = intent_documents[intent_index][1]
    confidence = float(intent_scores[intent_index])

    if score < 0.08:
        return {
            "answer": "I’m not sure about that. Please rephrase your question or contact the relevant university office for verified information.",
            "intent": intent if confidence >= 0.25 else "unknown",
            "score": score,
            "matched": None,
            "known": False,
        }
    return {
        "answer": best_item["answer"],
        "intent": intent if confidence >= 0.25 else best_item.get("intent", "unknown"),
        "score": score,
        "matched": best_item["question"],
        "known": True,
    }


# Warm the NLP stack in the background so the first question does not pay the import.
# This runs in a worker thread and never blocks the page from rendering.
def _warm_model():
    try:
        retrieve_answer("warm up")
    except Exception:
        pass


if "_model_warm" not in st.session_state:
    st.session_state._model_warm = True
    import threading
    threading.Thread(target=_warm_model, daemon=True).start()


def contextualize(question, history):
    q = question.strip()
    if not history:
        return q
    followups = {"what about that", "and then", "how about it", "tell me more", "explain more", "what next", "how"}
    if q.lower() in followups or (
        len(q.split()) <= 3
        and q.lower().startswith(("what about", "and ", "how about", "when is", "where is"))
    ):
        previous_user = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
        if previous_user:
            return f"{previous_user}. Follow-up question: {q}"
    return q


def title_from_question(question):
    """Build a short conversation title from the first question."""
    text = " ".join((question or "").split())
    text = re.sub(r"^(hi|hello|hey|please|kindly|can you|could you|i want to|i need to)\b[,\s]*", "", text, flags=re.I)
    text = text[:60].rstrip(" ?.!,")
    return (text[:1].upper() + text[1:]) if text else "New chat"


# ------------------------------------------------------------------ IDs, TIME, MARKDOWN

OWNER_COOKIE = "uai_owner"

def owner_id():
    """Stable per-browser identity, used to scope every conversation.

    Streamlit cannot set cookies from Python, so the id lives in the browser's
    localStorage. A tiny script puts it in the URL as ?owner=..., and Python reads it
    straight back from st.query_params. The URL is the single source of truth: it is the
    one place both sides can agree on, and it survives refreshes and reconnects.

    Returns None while the browser is still adding the parameter; the caller shows a
    brief loading state instead of inventing an identity that would be orphaned."""
    param = st.query_params.get("owner")
    if isinstance(param, list):
        param = param[0] if param else None
    if param and re.fullmatch(r"[0-9a-f]{8,64}", param):
        return param
    return None


def parse_dt(value):
    try:
        return datetime.strptime(str(value)[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return datetime.now(timezone.utc)


def stamp(value):
    """Short timestamp with no external dependency (e.g. \"14:32\" today, \"Mar 4, 14:32\" older)."""
    dt = parse_dt(value).astimezone()
    if dt.date() == datetime.now().astimezone().date():
        return dt.strftime("%H:%M")
    if dt.year == datetime.now().year:
        return dt.strftime("%b %d, %H:%M")
    return dt.strftime("%b %d %Y, %H:%M")


def date_group(value):
    """ChatGPT-style buckets: Today / Yesterday / Previous 7 days / Previous 30 days / older."""
    dt = parse_dt(value).astimezone()
    days = (datetime.now().astimezone().date() - dt.date()).days
    if days <= 0:
        return "Today"
    if days == 1:
        return "Yesterday"
    if days <= 7:
        return "Previous 7 days"
    if days <= 30:
        return "Previous 30 days"
    return "Older"


def markdown_to_html(text):
    """Small, dependency-free Markdown renderer: escaped HTML, fenced code, inline code,
    bold/italic, links, lists and paragraphs. Everything is escaped first so message
    content can never inject markup."""
    text = (text or "").replace("\r\n", "\n")
    code_blocks = []

    def stash(match):
        idx = len(code_blocks)
        code_blocks.append((match.group(1) or "", match.group(2)))
        return f"\x00CODE{idx}\x00"

    text = re.sub(r"```([\w+-]*)\n?(.*?)```", stash, text, flags=re.S)
    escaped = html.escape(text)

    def inline(line):
        line = re.sub(r"`([^`]+)`", r"<code>\1</code>", line)
        line = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", line)
        line = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", line)
        line = re.sub(
            r"\[([^\]]+)\]\((https?://[^\s)]+)\)",
            r'<a href="\2" target="_blank" rel="noopener noreferrer">\1</a>',
            line,
        )
        return line

    blocks, para, bullets = [], [], []

    def flush_para():
        if para:
            blocks.append("<p>" + "<br>".join(inline(l) for l in para) + "</p>")
            para.clear()

    def flush_list():
        if bullets:
            blocks.append("<ul>" + "".join(f"<li>{inline(b)}</li>" for b in bullets) + "</ul>")
            bullets.clear()

    for line in escaped.split("\n"):
        stripped = line.strip()
        if re.fullmatch(r"\x00CODE\d+\x00", stripped):
            flush_para(); flush_list()
            blocks.append(stripped)
        elif not stripped:
            flush_para(); flush_list()
        elif re.match(r"^[-*+]\s+", stripped):
            flush_para()
            bullets.append(re.sub(r"^[-*+]\s+", "", stripped))
        elif stripped.startswith("#"):
            flush_para(); flush_list()
            level = len(stripped) - len(stripped.lstrip("#"))
            blocks.append(f"<h{min(level, 6)}>{inline(stripped[level:].strip())}</h{min(level, 6)}>")
        else:
            flush_list()
            para.append(stripped)
    flush_para(); flush_list()

    out = "\n".join(blocks)

    def restore(match):
        lang, code = code_blocks[int(match.group(1))]
        label = f'<span class="cb-lang">{html.escape(lang)}</span>' if lang else ""
        return (
            '<div class="cb-wrap">' + label
            + f'<pre><code class="language-{html.escape(lang)}">{html.escape(code.rstrip(chr(10)))}</code></pre>'
            + '<button class="cb-copy" type="button" aria-label="Copy code">Copy</button></div>'
        )

    return re.sub(r"\x00CODE(\d+)\x00", restore, out)


# ------------------------------------------------------------------ PAGE + THEME

st.set_page_config(
    page_title="University AI Support",
    page_icon="✳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Give this browser a durable identity before anything else renders.
# Give this browser a durable identity. Streamlit cannot set a cookie from Python, so the id
# lives in localStorage and is mirrored into the URL as ?owner=... - the one value both the
# browser and Python can see. history.replaceState updates it without a reload; the page is
# only reloaded when the app has not yet seen an owner, which is the bare first visit.
st.html(
    """<script>
    (function () {
      const KEY = 'uai_owner';
      const gen = () => (crypto.randomUUID ? crypto.randomUUID().replace(/-/g, '')
        : Date.now().toString(36) + Math.random().toString(36).slice(2));
      let id = null;
      try { id = localStorage.getItem(KEY); } catch (e) {}
      const sent = new URLSearchParams(window.location.search).get('owner');
      if (!id) {
        id = sent || gen();
        try { localStorage.setItem(KEY, id); } catch (e) {}
      }
      if (sent === id) return;                 // app already has this identity
      const url = new URL(window.location.href);
      url.searchParams.set('owner', id);
      url.hash = '';
      if (!sent) {
        window.location.replace(url.toString());  // first visit: reload once with the id
      } else {
        history.replaceState(null, '', url.toString());
      }
    })();
    </script>""",
    unsafe_allow_javascript=True,
)

# The theme switch (Light / Dark / Use system setting) lives in Streamlit's own Settings menu.
# Streamlit repaints its widgets the moment it changes, but it does NOT rerun this script, so
# custom HTML would keep its old colours. The script below closes that gap: it mirrors the
# theme Streamlit is really painting onto <html>, including "Use system setting" flipping
# while the app is open.
THEME_STYLE = r"""
:root{--uai-accent:#10a37f;--uai-radius:18px}
.brand{font-size:1.02rem;font-weight:700;letter-spacing:.01em;margin:0 0 .15rem}
.brand span{color:var(--uai-accent)}
.small-muted{font-size:.72rem;letter-spacing:.08em}
/* only the small page furniture is uppercased - never message labels like the detected topic */
.group-label,.brand-sub{text-transform:uppercase} 
.hero{padding:2.2rem 1rem 1.5rem;text-align:center}
.hero h1{font-size:2.1rem;font-weight:600;margin-bottom:.4rem}
.hero p{margin:0}
.msg-row{display:flex;flex-direction:column;gap:2px;margin:14px 0}
.msg-row.user{align-items:flex-end}
.message-user{border-radius:var(--uai-radius);padding:11px 16px;margin:0;max-width:min(78%,680px);width:fit-content;line-height:1.6;white-space:pre-wrap;word-break:break-word}
.message-assistant{border-radius:var(--uai-radius);padding:13px 18px;margin:0;max-width:min(92%,820px);line-height:1.65;word-break:break-word}
.message-assistant p{margin:.35rem 0}
.message-assistant p:first-child{margin-top:0}
.message-assistant p:last-child{margin-bottom:0}
.message-assistant ul{margin:.4rem 0;padding-left:1.25rem}
.message-assistant code{font-size:.87em;padding:.12em .35em;border-radius:5px}
.message-assistant pre{margin:.6rem 0;padding:.85rem 1rem;border-radius:12px;overflow-x:auto}
.message-assistant pre code{padding:0;background:none}
.message-assistant a{color:var(--uai-accent)}
.msg-meta{display:flex;align-items:center;font-size:.72rem;opacity:.62;padding:0;line-height:1.2;white-space:nowrap}
/* Streamlit paints the chat input border and focus rings with its primary colour (red by
   default); use the university green instead */
[data-testid="stChatInput"] > div:focus-within{border-color:var(--uai-accent)!important;box-shadow:0 0 0 1px var(--uai-accent)!important}
[data-testid="stChatInput"] textarea{outline:none!important}
[data-testid="stBaseButton-secondary"]:focus-visible{border-color:var(--uai-accent)!important;box-shadow:0 0 0 1px var(--uai-accent)!important}
.answer-label{font-weight:600;margin-top:4px;font-size:.88rem}
.avatar{color:var(--uai-accent);margin-right:7px}
.cb-wrap{position:relative}
.cb-lang{display:block;font-size:.7rem;letter-spacing:.06em;text-transform:uppercase;opacity:.6;margin-bottom:.25rem}
.cb-copy{position:absolute;top:6px;right:8px;font-size:.7rem;padding:3px 9px;border-radius:7px;border:1px solid currentColor;background:transparent;color:inherit;opacity:.55;cursor:pointer}
.cb-copy:hover{opacity:1}
/* typing indicator */
.typing{display:inline-flex;gap:5px;align-items:center;padding:6px 2px}
.typing i{width:7px;height:7px;border-radius:50%;background:var(--uai-accent);opacity:.35;animation:uai-blink 1.2s infinite}
.typing i:nth-child(2){animation-delay:.2s}.typing i:nth-child(3){animation-delay:.4s}
@keyframes uai-blink{0%,80%,100%{opacity:.25;transform:translateY(0)}40%{opacity:1;transform:translateY(-2px)}}
/* chat rows in the sidebar */
/* chat row titles: allow a second line before clipping, like ChatGPT's history list */
[data-testid="stSidebar"] [class*="st-key-chatrow_"] button p{
  white-space:normal!important;overflow:visible!important;text-overflow:unset!important;
  line-height:1.25;text-align:left;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;
  overflow:hidden!important}
[data-testid="stSidebar"] [class*="st-key-chatrow_"] button{height:auto!important;min-height:2.1rem;padding-top:.4rem!important;padding-bottom:.4rem!important;align-items:center!important}
.chat-active-dot{color:var(--uai-accent);font-size:.6rem;vertical-align:middle;margin-right:2px}
/* University identity green on the primary action (Streamlit's default is red).
   Scoped to :root so it works wherever Streamlit places the button. */
[data-testid="stBaseButton-primary"]{background-color:var(--uai-accent)!important;border-color:var(--uai-accent)!important;color:#fff!important}
[data-testid="stBaseButton-primary"]:hover,[data-testid="stBaseButton-primary"]:focus{background-color:#0d8b6d!important;border-color:#0d8b6d!important;color:#fff!important}
[data-testid="stBaseButton-primary"] p{color:#fff!important}
/* three-dot menu: text glyph, centred, no chevron, quiet until hover. Material icon fonts are
   not available in every install (their names would paint as literal text), so the kebab is a
   plain character - always renders, no extra download. */
[data-testid="stSidebar"] [data-testid="stPopoverButton"]{
  min-width:30px!important;min-height:30px!important;height:30px!important;padding:0!important;
  justify-content:center!important;gap:0!important;box-sizing:border-box;
  border:none!important;background:transparent!important;box-shadow:none!important;
  color:inherit!important;opacity:.6;line-height:1!important;font-size:1.15rem!important;
  transition:opacity .15s ease,background .15s ease}
[data-testid="stSidebar"] [data-testid="stPopoverButton"]:hover{opacity:1;background:rgba(128,128,128,.18)!important}
/* Hide the popover's own chevron. Streamlit appends an "expand_more" Material icon after the
   label; the icon font is not installed here, so it would otherwise paint as literal text and
   crowd out our label. Matched by its text so the label itself is never touched. */
[data-testid="stPopoverButton"] [data-testid="stIconMaterial"]{display:none!important}
/* some builds render the chevron as a bare aria-hidden sibling node */
[data-testid="stPopoverButton"] > div[aria-hidden="true"]{display:none!important}
[data-testid="stPopoverButton"] > span[aria-hidden="true"]{display:none!important}
/* the ⋮ glyph we pass as the label renders as plain text in a <p> */
[data-testid="stSidebar"] [data-testid="stPopoverButton"] p{margin:0!important;line-height:1!important}
/* the profile popover is a normal labelled button: full width, label + glyph */
[data-testid="stSidebar"] [data-testid="stPopoverButton"]:has(p){
  min-width:0!important;width:100%!important;justify-content:flex-start!important;
  gap:.45rem!important;opacity:1!important;font-size:.95rem!important;padding:.4rem .6rem!important}
/* keep the action row tight against the bubble above it */
[data-testid="stMain"] [class*="st-key-msg_"] > div > [data-testid="stHorizontalBlock"]{
  margin-top:-.2rem;gap:.2rem!important;align-items:center!important}
[data-testid="stMain"] [class*="st-key-msg_"] [data-testid="stColumn"]{padding:0!important}
/* message action row: one small popover button that shows "17:02  ⋯". The inner content span is
   given the full size so the timestamp is never clipped to an empty box. */
[data-testid="stMain"] [data-testid="stPopoverButton"]{
  border:none!important;background:transparent!important;box-shadow:none!important;color:inherit!important;
  opacity:.6;padding:.1rem .35rem!important;min-height:0!important;height:auto!important;
  width:auto!important;min-width:0!important;max-width:none!important;
  font-size:.72rem!important;line-height:1.1!important;border-radius:7px;
  justify-content:flex-start!important;gap:.25rem!important;
  transition:opacity .15s ease,background .15s ease}
[data-testid="stMain"] [data-testid="stPopoverButton"] > div{
  width:auto!important;min-width:0!important;flex:0 1 auto!important;overflow:visible!important}
[data-testid="stMain"] [data-testid="stPopoverButton"] [data-testid="stMarkdownContainer"],
[data-testid="stMain"] [data-testid="stPopoverButton"] p{
  overflow:visible!important;text-overflow:clip!important;white-space:nowrap!important;
  max-width:none!important;width:auto!important;margin:0!important;
  font-size:.72rem!important;line-height:1.1!important;color:inherit!important;opacity:1!important}
[data-testid="stMain"] [data-testid="stPopoverButton"]:hover{opacity:1;background:rgba(128,128,128,.15)!important}
/* feedback buttons size to their label instead of stretching */
[data-testid="stMain"] [class*="st-key-feedback_yes"] button,
[data-testid="stMain"] [class*="st-key-feedback_no"] button{width:auto!important;min-width:0!important}
/* archived header row: keep the label and Back button on one line */
[data-testid="stSidebar"] [class*="st-key-archived_head"] p{text-transform:uppercase;font-size:.72rem;letter-spacing:.08em;opacity:.6;margin:0}
.archive-head{display:flex;align-items:center;justify-content:space-between;font-size:.72rem;font-weight:600;letter-spacing:.08em;text-transform:uppercase;opacity:.6;margin:16px 0 6px 2px}
/* the profile popover keeps its label */
[data-testid="stSidebar"] [class*="st-key-"] [data-testid="stPopover"] button.profile-btn{font-size:.95rem!important}
@media (max-width: 900px){
  [data-testid="stSidebar"] [data-testid="stPopover"] button{font-size:1rem!important}
  [data-testid="stSidebar"] [data-testid="stPopover"] button::before{content:none}
  [data-testid="stSidebar"] [data-testid="stPopover"] button [data-testid="stIconMaterial"]{display:inline}
}
.group-label{font-size:.72rem;font-weight:600;opacity:.55;margin:14px 0 4px 2px}
.empty-hint{font-size:.8rem;opacity:.6;padding:6px 2px}
.profile-row{display:flex;align-items:center;gap:9px}
.profile-avatar{width:30px;height:30px;border-radius:50%;background:var(--uai-accent);color:#fff;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:.82rem}
/* ChatGPT light */
.message-user{background:#f4f4f4;color:#0d0d0d}
.message-assistant{background:#ffffff;color:#0d0d0d;border:1px solid #ececec}
.message-assistant code{background:#f4f4f4}
.message-assistant pre{background:#f7f7f8;border:1px solid #ececec}
/* ChatGPT dark - the class the sync script toggles on <html> */
.uai-dark .hero h1,.uai-dark .hero p,.uai-dark .answer-label{color:#ececec}
.uai-dark .small-muted{color:#b4b4b4}
.uai-dark .message-user{background:#303030;color:#ececec}
.uai-dark .message-assistant{background:#212121;color:#ececec;border:1px solid #303030}
.uai-dark .message-assistant code{background:#303030}
.uai-dark .message-assistant pre{background:#171717;border:1px solid #303030}
/* compact icon buttons used for message actions */
[data-testid="stSidebar"] .stButton button{text-align:left}
div[data-testid="stHorizontalBlock"] .stButton button[kind="secondary"]{transition:background .15s ease,color .15s ease}
@media (max-width: 900px){
  .message-user{max-width:92%}
  .message-assistant{max-width:100%}
  .hero h1{font-size:1.7rem}
}
"""
st.markdown(f"<style id='uai-theme-style'>{THEME_STYLE}</style>", unsafe_allow_html=True)
st.html(
    """<script>
    (function () {
      const LIGHT = {bg:'#ffffff', surface:'#f4f4f4', ink:'#0d0d0d'};
      const DARK  = {bg:'#212121', surface:'#303030', ink:'#ececec', sidebar:'#171717'};
      function apply() {
        const app = document.querySelector('[data-testid="stApp"]');
        if (!app) return;
        const nums = getComputedStyle(app).backgroundColor.match(/\\d+/g);
        if (!nums || nums.length < 3) return;
        const [r, g, b] = nums.slice(0, 3).map(Number);
        const dark = (0.299 * r + 0.587 * g + 0.114 * b) < 128;
        document.documentElement.classList.toggle('uai-dark', dark);
        if (document.documentElement.dataset.uaiTheme === (dark ? 'dark' : 'light')) return;
        document.documentElement.dataset.uaiTheme = dark ? 'dark' : 'light';
        const p = dark ? DARK : LIGHT;
        let css = 'html{--background-color:' + p.bg + ';--secondary-background-color:' + p.surface +
                  ';--text-color:' + p.ink + ';}';
        if (dark) css += '[data-testid="stSidebar"]{background-color:' + DARK.sidebar + ';}';
        let el = document.getElementById('uai-theme-vars');
        if (!el) { el = document.createElement('style'); el.id = 'uai-theme-vars'; document.head.appendChild(el); }
        el.textContent = css;
      }
      apply();
      new MutationObserver(apply).observe(document.body, {
        attributes: true, subtree: true, attributeFilter: ['class', 'style']
      });
      window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', apply);

      // Copy buttons on code blocks and messages (delegated so it survives reruns).
      document.addEventListener('click', function (ev) {
        const btn = ev.target.closest('.cb-copy');
        if (!btn) return;
        const code = btn.closest('.cb-wrap').querySelector('code');
        navigator.clipboard.writeText(code ? code.innerText : '').then(function () {
          btn.textContent = 'Copied';
          setTimeout(function () { btn.textContent = 'Copy'; }, 1400);
        });
      });
    })();
    </script>""",
    unsafe_allow_javascript=True,
)

# ------------------------------------------------------------------ SESSION STATE

init_db()
owner = owner_id()

# No identity in the URL yet: the bootstrap script is adding ?owner=... and reloading.
# Render a short loading state instead of inventing an id, which would strand this
# visitor's chats under an identity their browser never keeps.
if owner is None:
    st.markdown(
        '<div class="hero"><h1>✳ University AI Support</h1>'
        '<p>Preparing your session…</p></div>',
        unsafe_allow_html=True,
    )
    st.stop()

if "data_version" not in st.session_state:
    st.session_state.data_version = 0
if "pending_feedback" not in st.session_state:
    st.session_state.pending_feedback = None
if "show_admin" not in st.session_state:
    st.session_state.show_admin = False
if "last_exchange" not in st.session_state:
    st.session_state.last_exchange = None
if "show_archived" not in st.session_state:
    st.session_state.show_archived = False
if "history_search" not in st.session_state:
    st.session_state.history_search = ""

# Dialogs hold the target id so a dialog opened this run renders after the rerun.
for _key in ("rename_target", "archive_target", "delete_target", "restore_target", "purge_target"):
    st.session_state.setdefault(_key, None)

# Adopt conversations from before ownership existed (single-user history): the first
# visitor to open the app takes them over instead of leaving them orphaned.
with db_connect() as _conn:
    _orphans = _conn.execute("SELECT COUNT(*) FROM conversations WHERE owner_id IS NULL").fetchone()[0]
    if _orphans:
        _conn.execute("UPDATE conversations SET owner_id=? WHERE owner_id IS NULL", (owner,))

if "conversation_id" not in st.session_state:
    _active = get_conversations(owner, archived=0)
    if _active:
        st.session_state.conversation_id = _active[0]["id"]
    else:
        st.session_state.conversation_id = reusable_draft(owner) or create_conversation(owner)

cid = st.session_state.conversation_id
version = st.session_state.data_version


def refresh(keep_conversation=None):
    """Bump the cache version and rerun the script."""
    st.session_state.data_version += 1
    if keep_conversation is not None:
        st.session_state.conversation_id = keep_conversation
    st.rerun()


def open_conversation(conversation_id):
    st.session_state.conversation_id = conversation_id
    st.session_state.pending_feedback = None
    st.session_state.last_exchange = None
    st.session_state.show_admin = False
    st.rerun()


def start_new_chat():
    draft = reusable_draft(owner)
    if draft is not None or st.session_state.get("conversation_id") == draft:
        # Already on an empty chat - reuse it instead of creating another blank one.
        st.session_state.conversation_id = draft or st.session_state["conversation_id"]
        st.session_state.show_admin = False
        st.session_state.show_archived = False
        st.rerun()
    st.session_state.conversation_id = create_conversation(owner)
    st.session_state.pending_feedback = None
    st.session_state.last_exchange = None
    st.session_state.show_admin = False
    st.session_state.show_archived = False
    st.session_state.data_version += 1
    st.rerun()


def ask(conversation_id, question):
    """Store the question, answer it, and store the reply."""
    history = get_messages(conversation_id, owner)
    result = retrieve_answer(contextualize(question, history))
    save_message(conversation_id, owner, "user", question)
    if not history:
        rename_conversation(conversation_id, owner, title_from_question(question))
    save_message(conversation_id, owner, "assistant", result["answer"], result["intent"])
    st.session_state.last_exchange = {
        "question": question, "answer": result["answer"], "intent": result["intent"],
    }


def regenerate_response(conversation_id, message_id):
    """Re-answer a message from its preceding user question, in place."""
    messages = get_messages(conversation_id, owner)
    index = next((i for i, m in enumerate(messages) if m["id"] == message_id), None)
    if index is None:
        return
    question = next((m["content"] for m in reversed(messages[:index]) if m["role"] == "user"), None)
    if not question:
        return
    result = retrieve_answer(question)
    update_message(message_id, owner, result["answer"], result["intent"])


# ------------------------------------------------------------------ DIALOGS

@st.dialog("Rename conversation", width="small")
def dialog_rename():
    conversation = get_conversation(st.session_state.rename_target, owner)
    if not conversation:
        st.session_state.rename_target = None
        st.error("That conversation could not be found.")
        return
    st.caption("Conversation titles help you spot a chat in your history.")
    title = st.text_input("Title", value=conversation["title"], max_chars=60, key="rename_input")
    col_save, col_cancel = st.columns(2)
    if col_save.button("Save", type="primary", use_container_width=True):
        if rename_conversation(conversation["id"], owner, title):
            st.session_state.rename_target = None
            refresh()
        else:
            st.error("Enter a title before saving.")
    if col_cancel.button("Cancel", use_container_width=True):
        st.session_state.rename_target = None
        refresh()


@st.dialog("Archive conversation?", width="small")
def dialog_archive():
    conversation = get_conversation(st.session_state.archive_target, owner)
    if not conversation:
        st.session_state.archive_target = None
        st.error("That conversation could not be found.")
        return
    st.write(f"**{conversation['title']}** will move to Archived chats.")
    st.caption("Nothing is lost. You can open it or restore it at any time.")
    col_yes, col_no = st.columns(2)
    if col_yes.button("Archive", type="primary", use_container_width=True):
        set_archived(conversation["id"], owner, True)
        st.session_state.archive_target = None
        if st.session_state.conversation_id == conversation["id"]:
            remaining = get_conversations(owner, archived=0)
            st.session_state.conversation_id = remaining[0]["id"] if remaining else create_conversation(owner)
        refresh()
    if col_no.button("Cancel", use_container_width=True):
        st.session_state.archive_target = None
        refresh()


@st.dialog("Delete conversation permanently?", width="small")
def dialog_delete():
    conversation = get_conversation(st.session_state.delete_target, owner)
    if not conversation:
        st.session_state.delete_target = None
        st.error("That conversation could not be found.")
        return
    count = len(get_messages(conversation["id"], owner))
    st.error("This cannot be undone.")
    st.write(f"**{conversation['title']}** and its {count} message(s) will be permanently deleted.")
    col_yes, col_no = st.columns(2)
    if col_yes.button("Delete forever", type="primary", use_container_width=True):
        delete_conversation(conversation["id"], owner)
        st.session_state.delete_target = None
        if st.session_state.conversation_id == conversation["id"]:
            remaining = get_conversations(owner, archived=0)
            st.session_state.conversation_id = remaining[0]["id"] if remaining else create_conversation(owner)
        refresh()
    if col_no.button("Keep conversation", use_container_width=True):
        st.session_state.delete_target = None
        refresh()


# ------------------------------------------------------------------ SIDEBAR

with st.sidebar:
    # Streamlit draws its own collapse control in the sidebar header; this line just anchors
    # the brand to it so the button sits beside the name.
    st.markdown('<div class="brand">✳ University <span>AI</span> Support</div>', unsafe_allow_html=True)
    st.markdown('<div class="small-muted">Student help desk</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    if st.button("＋  New chat", key="new_chat", use_container_width=True, type="primary"):
        start_new_chat()

    # The sidebar toggle: Streamlit's built-in collapsed control is only shown on hover, which
    # is easy to miss. This keeps it visible and gives it hover feedback.
    st.markdown(
        """<style>
        [data-testid="stSidebarCollapseButton"] button,[data-testid="stExpandSidebarButton"] button{
          border-radius:9px;opacity:.9;transition:background .15s ease}
        [data-testid="stSidebarCollapseButton"] button:hover,[data-testid="stExpandSidebarButton"] button:hover{
          background:rgba(128,128,128,.18)}
        </style>""",
        unsafe_allow_html=True,
    )

    # History search
    search = st.text_input(
        "Search chats", key="history_search", placeholder="Search chats…",
        label_visibility="collapsed",
        help="Find a conversation by title or by anything said in it.",
    )

    active_chats = cached_conversations(owner, 0, version)
    archived_chats = cached_conversations(owner, 1, version)

    def render_chat_row(chat, *, archived=False):
        is_current = (not archived) and chat["id"] == cid
        with st.container(key=f"chatrow_{chat['id']}"):
            col_name, col_menu = st.columns([14, 1], vertical_alignment="center", gap="small")
            with col_name:
                label = ("● " if is_current else "") + chat["title"]
                if st.button(label, key=f"open_{chat['id']}", use_container_width=True,
                             help=chat["title"]):
                    open_conversation(chat["id"])
            with col_menu:
                with st.popover("⋮", use_container_width=False,
                                help=f"More options for {chat['title']}"):
                    if archived:
                        if st.button("Open chat", key=f"a_open_{chat['id']}", use_container_width=True):
                            open_conversation(chat["id"])
                        if st.button("Restore", key=f"a_restore_{chat['id']}", use_container_width=True):
                            set_archived(chat["id"], owner, False)
                            refresh()
                        if st.button("Delete forever", key=f"a_del_{chat['id']}", use_container_width=True):
                            st.session_state.delete_target = chat["id"]
                            st.rerun()
                    else:
                        if st.button("Rename", key=f"r_{chat['id']}", use_container_width=True):
                            st.session_state.rename_target = chat["id"]
                            st.rerun()
                        if st.button("Archive", key=f"ar_{chat['id']}", use_container_width=True):
                            st.session_state.archive_target = chat["id"]
                            st.rerun()
                        if st.button("Delete", key=f"d_{chat['id']}", use_container_width=True):
                            st.session_state.delete_target = chat["id"]
                            st.rerun()

    def filter_chats(chats, query):
        if not query:
            return chats
        q = query.lower()
        matched = []
        for chat in chats:
            if q in chat["title"].lower() or any(
                q in m["content"].lower() for m in get_messages(chat["id"], owner)
            ):
                matched.append(chat)
        return matched

    if st.session_state.show_archived:
        # Header row: label on the left, Back on the right. Native Streamlit columns keep
        # both aligned without custom HTML fighting the sidebar width.
        st.markdown('<div class="group-label">Archived</div>', unsafe_allow_html=True)
        if st.button("← Back to chats", key="back_from_archived", use_container_width=True):
            st.session_state.show_archived = False
            st.rerun()
        shown = filter_chats(archived_chats, search)
        if not shown:
            st.markdown('<div class="empty-hint">No archived conversations.</div>', unsafe_allow_html=True)
        for chat in shown:
            render_chat_row(chat, archived=True)
    else:
        shown = filter_chats(active_chats, search)
        if not shown:
            st.markdown(
                '<div class="empty-hint">No conversations match your search.</div>'
                if search else '<div class="empty-hint">No conversations yet. Start a new chat.</div>',
                unsafe_allow_html=True,
            )
        last_group = None
        for chat in shown:
            group = date_group(chat.get("updated_at") or chat.get("created_at"))
            if group != last_group and not search:
                st.markdown(f'<div class="group-label">{group}</div>', unsafe_allow_html=True)
                last_group = group
            render_chat_row(chat)

        if archived_chats:
            if st.button(f"🗄  Archived ({len(archived_chats)})", key="open_archived", use_container_width=True):
                st.session_state.show_archived = True
                st.rerun()

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    if st.button("⚙  Knowledge & feedback", key="open_admin", use_container_width=True,
                 help="Manage the answer knowledge base and review feedback"):
        st.session_state.show_admin = not st.session_state.show_admin
        st.session_state.show_archived = False
        st.rerun()

    st.markdown("---")
    # Profile section
    with st.popover("👤  Student", use_container_width=True, help="Your profile and session"):
        st.markdown("**Guest student**")
        st.caption("Signed in with a browser session. Your chats stay private to this browser.")
        st.divider()
        total = len(active_chats) + len(archived_chats)
        st.caption(f"Conversations: {len(active_chats)} active · {len(archived_chats)} archived")
        st.caption(f"Total stored: {total}")
        st.caption("AI-powered university help desk · NLP · FAQ retrieval")

# ------------------------------------------------------------------ ADMIN / FEEDBACK LEARNING

if st.session_state.show_admin:
    st.title("Knowledge & feedback")
    st.caption("Feedback-driven improvement: reviewed corrections become searchable knowledge-base entries.")
    with st.expander("Add a verified answer to the knowledge base", expanded=True):
        with st.form("add_faq_form"):
            new_question = st.text_input("Example question")
            new_answer = st.text_area("Verified answer", height=120)
            new_intent = st.selectbox("Category", list(INTENT_EXAMPLES.keys()))
            submitted = st.form_submit_button("Save knowledge entry")
            if submitted:
                if new_question.strip() and new_answer.strip():
                    learned = load_learned_faq()
                    learned.append({"intent": new_intent, "question": new_question.strip(), "answer": new_answer.strip(), "keywords": []})
                    save_learned_faq(learned)
                    st.success("Knowledge entry saved. It will be included in future searches.")
                else:
                    st.error("Enter both a question and an answer.")
    with st.expander("Recent user feedback"):
        with db_connect() as conn:
            feedback_rows = conn.execute("SELECT * FROM feedback ORDER BY id DESC LIMIT 30").fetchall()
        if feedback_rows:
            for row in feedback_rows:
                st.markdown(f"**{row['rating'].upper()}** · {row['question']}")
                if row["correction"]:
                    st.write("Suggested correction:", row["correction"])
                st.caption(stamp(row["created_at"]))
        else:
            st.info("No feedback has been submitted yet.")
    st.markdown("---")

# ------------------------------------------------------------------ MAIN CHAT

conversation = get_conversation(cid, owner)
if conversation is None:
    # The selected chat is gone (deleted in another tab, or its id was stale). Recover by
    # choosing an existing chat, or creating a fresh one, and re-reading it.
    _fallback = get_conversations(owner, archived=0)
    cid = _fallback[0]["id"] if _fallback else create_conversation(owner)
    st.session_state.conversation_id = cid
    conversation = get_conversation(cid, owner)
    if conversation is None:  # extremely unlikely; never let the page crash on it
        st.error("That conversation could not be loaded. Please start a new chat.")
        st.stop()

messages = cached_messages(cid, owner, version)

head_l, head_r = st.columns([6, 2])
with head_l:
    st.markdown(f"#### {html.escape(conversation['title'])}")
    if conversation["archived"]:
        st.caption("📦 Archived — restore it from the sidebar to keep chatting.")
with head_r:
    if messages:
        transcript = "\n\n".join(
            f"{'You' if m['role'] == 'user' else 'University AI Support'}: {m['content']}" for m in messages
        )
        st.download_button(
            "⬇  Export chat", data=transcript, file_name=f"chat-{cid}.txt",
            mime="text/plain", key=f"export_{cid}", use_container_width=True,
        )

if not messages:
    st.markdown("""
    <div class="hero">
      <h1>How can I help you today?</h1>
      <p>Your university support assistant. Ask a question in your own words.</p>
    </div>
    """, unsafe_allow_html=True)
    suggestions = [
        "How do I apply for admission?", "How can I pay my fees?",
        "I cannot log in to the student portal", "Where can I find the exam timetable?",
    ]
    cols = st.columns(2)
    for i, suggestion in enumerate(suggestions):
        with cols[i % 2]:
            if st.button(suggestion, key=f"suggest_{i}", use_container_width=True):
                st.session_state.suggested_question = suggestion
                st.rerun()
else:
    for index, message in enumerate(messages):
        is_user = message["role"] == "user"
        with st.container(key=f"msg_{message['id']}"):
            if is_user:
                st.markdown(
                    '<div class="msg-row user">'
                    f'<div class="message-user">{markdown_to_html(message["content"])}</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )
            else:
                topic = ""
                if message["intent"]:
                    topic = f'<div class="small-muted" style="margin-top:-2px">Detected topic: {html.escape(message["intent"].replace("_", " ").title())}</div>'
                st.markdown(
                    '<div class="msg-row assistant">'
                    '<div class="answer-label"><span class="avatar">✳</span>University AI Support</div>'
                    f'<div class="message-assistant">{markdown_to_html(message["content"])}</div>'
                    f'{topic}</div>',
                    unsafe_allow_html=True,
                )

            # Action row: timestamp and the ⋯ menu on one line, right under the bubble and on
            # the bubble's own side - the ChatGPT convention. The label carries both so they
            # cannot be split into separate blocks.
            if is_user:
                _spacer, _menu, _edge = st.columns([0.855, 0.06, 0.085], gap="small",
                                                   vertical_alignment="center")
            else:
                _menu, _spacer = st.columns([0.06, 0.94], gap="small", vertical_alignment="center")
            with _menu:
                with st.popover(f'{stamp(message["created_at"])}  ⋯', use_container_width=False,
                                help="Message actions"):
                    if st.button("Copy", key=f"copy_{message['id']}", use_container_width=True,
                                 help="Copy this message to the clipboard"):
                        st.html(
                            "<script>(function(){var t=" + json.dumps(message["content"]) + ";"
                            "navigator.clipboard.writeText(t).catch(function(){});})();</script>",
                            unsafe_allow_javascript=True,
                        )
                        st.toast("Message copied to the clipboard")
                    if not is_user:
                        if st.button("Regenerate response", key=f"regen_{message['id']}", use_container_width=True,
                                     help="Answer the same question again"):
                            regenerate_response(cid, message["id"])
                            st.session_state.last_exchange = None
                            refresh()
                    if st.button("Delete message", key=f"delmsg_{message['id']}", use_container_width=True,
                                 help="Remove this message from the conversation"):
                        delete_message(message["id"], owner)
                        st.session_state.last_exchange = None
                        refresh()

question = st.chat_input(
    "Restore this conversation to continue chatting" if conversation["archived"]
    else "Message University AI Support...",
    disabled=bool(conversation["archived"]),
    submit_mode="submit",
)
if "suggested_question" in st.session_state:
    question = st.session_state.pop("suggested_question")

if question and question.strip() and not conversation["archived"]:
    question = question.strip()
    # Show the question and a typing indicator while the answer is being retrieved, so the
    # reply never appears to come from nowhere on a cold start.
    st.markdown(
        '<div class="msg-row user">'
        f'<div class="message-user">{markdown_to_html(question)}</div></div>'
        '<div class="msg-row assistant">'
        '<div class="answer-label"><span class="avatar">✳</span>University AI Support</div>'
        '<div class="typing" aria-label="Assistant is typing"><i></i><i></i><i></i></div>'
        '</div>',
        unsafe_allow_html=True,
    )
    ask(cid, question)
    refresh()

# ------------------------------------------------------------------ FEEDBACK

latest = st.session_state.last_exchange
if latest and messages:
    st.markdown("---")
    st.caption("Was this answer helpful?")
    fb1, fb2, _rest = st.columns([1.1, 3.2, 5.7], gap="small")
    with fb1:
        if st.button("👍 Yes", key="feedback_yes"):
            save_feedback(cid, latest["question"], latest["answer"], "positive")
            st.session_state.pending_feedback = None
            st.session_state.last_exchange = None
            st.toast("Thank you. Your feedback has been recorded.")
            refresh()
    with fb2:
        if st.button("👎 No — suggest a correction", key="feedback_no"):
            st.session_state.pending_feedback = latest
    if st.session_state.pending_feedback:
        pending = st.session_state.pending_feedback
        correction = st.text_area(
            "What should the correct answer be? Please provide verified university information.",
            key="correction_text",
        )
        if st.button("Submit correction", key="submit_correction"):
            if correction.strip():
                save_feedback(cid, pending["question"], pending["answer"], "negative", correction.strip())
                learned = load_learned_faq()
                learned.append({"intent": pending.get("intent", "unknown"), "question": pending["question"], "answer": correction.strip(), "keywords": []})
                save_learned_faq(learned)
                st.session_state.pending_feedback = None
                st.session_state.last_exchange = None
                st.toast("Correction saved to feedback and the knowledge base.")
                refresh()
            else:
                st.error("Please enter a correction before submitting.")

# ------------------------------------------------------------------ DIALOG RENDERING
# Rendered last so a dialog opened in the sidebar has already stored its target id.
if st.session_state.rename_target:
    dialog_rename()
elif st.session_state.archive_target:
    dialog_archive()
elif st.session_state.delete_target:
    dialog_delete()
