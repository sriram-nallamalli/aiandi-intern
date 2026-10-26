"""Classify forum comments as acceptable, toxic or spam with Cloudflare's Clef models,
then write results.html, a page that is easy to scan and review.

Usage:
    python classify.py --setup           # save your account ID and token (asks twice)
    python classify.py --check           # send one comment to each model to test credentials
    python classify.py                   # run both clef and clef-flash (default)
    python classify.py --model clef      # run only one model
    python classify.py --refresh         # ignore saved results and call the API again
    python classify.py --html-only       # rebuild results.html from saved results
    python classify.py --doctor          # check everything before you submit

Credentials are read from cloud-flare.local.sh, which .gitignore keeps out of git.
Only the Python standard library is used; nothing to install.
"""

import argparse
import html
import json
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMMENTS_FILE = HERE / "comments.json"
CREDS_FILE = HERE / "cloud-flare.local.sh"
HTML_FILE = HERE / "results.html"
API_BASE = "https://api.cloudflare.com/client/v4"
MODELS = ["clef", "clef-flash"]
LABELS = ["acceptable", "toxic", "spam"]
LOW_CONFIDENCE = 0.5  # answers below this are flagged for a human look

# The question sent with every comment. Clearer criteria mean fewer mistakes;
# edit these, run with --refresh, and compare.
QUESTION = {
    "type": "choice",
    "instructions": (
        "Classify `comment`, a post by `author` in the `thread` thread of an online forum "
        "for an Indian college course on building apps with AI tools (Claude, Codex). "
        "Students write informally and often mix Hindi and English (Hinglish); that is normal. "
        "Questions, complaints about the course, grading or deadlines, disagreement, and blunt "
        "feedback on someone's project are allowed as long as they do not attack a person."
    ),
    "criteria": {
        "acceptable": (
            "A genuine post about the course, assignments, AI tools, deadlines, grading or "
            "classmates' work: questions, answers, feedback, complaints, jokes or frustration, "
            "including sharing one's own project or a relevant resource, with no personal attack "
            "and no unrelated promotion."
        ),
        "toxic": (
            "Insults, mocks, shames, harasses or threatens a person or group (a student, teacher "
            "or TA), uses abusive language or slurs aimed at someone, accuses someone in order to "
            "humiliate them publicly, or urges others to pile on. Criticism of work, grades or "
            "course design without attacking a person is not toxic."
        ),
        "spam": (
            "Promotes a product, service, channel or link that is not a genuine contribution to "
            "the course discussion: paid assignment or exam help, phone or WhatsApp numbers for "
            "business, referral or affiliate links, crypto, giveaways, or repeated advertising."
        ),
    },
}


def load_creds():
    if not CREDS_FILE.exists():
        sys.exit(f"Missing {CREDS_FILE.name}. Run: python classify.py --setup")
    creds = dict(re.findall(r'export (\w+)="([^"]*)"', CREDS_FILE.read_text(encoding="utf-8")))
    account = creds.get("CLOUDFLARE_ACCOUNT_ID", "").strip()
    token = creds.get("CLOUDFLARE_API_TOKEN", "").strip()
    if not token or "your-cloudflare" in token or not account or account.startswith("0123456789"):
        sys.exit(f"{CREDS_FILE.name} still has sample values. Run: python classify.py --setup")
    return account, token


class FatalError(Exception):
    """An error that retrying cannot fix, such as rejected credentials."""


def ask_clef(comment, account, token, model, retries=6):
    payload = {
        # Navin's sample sends the model only in the URL; other published examples
        # (flaviocopes.com/clef) also send it in the body, where it must be "clef" or
        # "clef-flash". Sending the matching value works with both.
        "model": model,
        "state": {
            "comment": comment["comment"],
            "author": comment["author"],
            "thread": comment["thread"],
        },
        "questions": {"label": QUESTION},
    }
    request = urllib.request.Request(
        f"{API_BASE}/accounts/{account}/ai/run/@cf/cloudflare/{model}",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                data = json.load(response)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise FatalError(f"Cloudflare rejected the credentials (HTTP {e.code}). Check the "
                                 "account ID, and that the token has Workers AI Read and Edit.") from None
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(2 ** attempt)  # rate-limited or a hiccup: wait and retry
                continue
            body = e.read().decode("utf-8", "replace")[:300]
            raise FatalError(f"HTTP {e.code} on comment {comment['id']} ({model}): {body}") from None
        except urllib.error.URLError as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise FatalError(f"Network error on comment {comment['id']}: {e.reason}. "
                             "Check your internet connection.") from None
        if not data.get("success"):
            raise FatalError(f"Clef error on comment {comment['id']} ({model}): {data.get('errors')}")
        try:
            answer = data["result"]["answers"]["label"]
            if answer["choice"] not in LABELS:
                raise KeyError(answer["choice"])
            answer["confidence"] = float(answer["confidence"])
            return answer
        except (KeyError, TypeError, ValueError):
            raise FatalError(f"Unexpected answer from {model} on comment {comment['id']}: "
                             f"{json.dumps(data)[:300]}") from None
    raise FatalError(f"Gave up on comment {comment['id']} ({model}) after {retries} tries")


def cache_path(model):
    return HERE / f"results-{model}.json"


def load_cache(model):
    path = cache_path(model)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_cache(model, cache):
    cache_path(model).write_text(json.dumps(cache, indent=1, sort_keys=True), encoding="utf-8")


def classify_all(comments, model, refresh, workers):
    cache = {} if refresh else load_cache(model)
    todo = [c for c in comments if str(c["id"]) not in cache]
    if not todo:
        print(f"{model}: all {len(comments)} comments already saved (use --refresh to redo)")
        return cache
    account, token = load_creds()
    print(f"{model}: classifying {len(todo)} comments...")
    pool = ThreadPoolExecutor(max_workers=max(1, workers))
    futures = [(pool.submit(ask_clef, c, account, token, model), c) for c in todo]
    done = 0
    try:
        for future, c in futures:
            cache[str(c["id"])] = future.result()
            done += 1
            if done % 10 == 0 or done == len(todo):
                print(f"  {done}/{len(todo)}")
                save_cache(model, cache)
    except BaseException:
        for future, _ in futures:
            future.cancel()  # stop queued requests after an error or Ctrl+C
        raise
    finally:
        pool.shutdown(wait=True)
        if cache:
            save_cache(model, cache)  # keep every answer received so far
    return cache


def badge(label):
    return f'<span class="badge {label}">{label}</span>'


def prob_bar(answer):
    parts = "".join(
        f'<span class="p {k}" style="width:{max(answer["probabilities"].get(k, 0), 0) * 100:.1f}%" '
        f'title="{k}: {answer["probabilities"].get(k, 0):.2f}"></span>'
        for k in LABELS
    )
    return f'<div class="bar">{parts}</div>'


def label_cell(answer):
    return (f'<td class="lab">{badge(answer["choice"])}{prob_bar(answer)}'
            f'<small>conf {answer["confidence"]:.2f}</small></td>')


def render_row(c, main_answer, other_answer):
    cid = str(c["id"])
    label, conf = main_answer["choice"], main_answer["confidence"]
    is_low = conf < LOW_CONFIDENCE
    is_dis = other_answer is not None and other_answer["choice"] != label
    flags = ""
    if is_low:
        flags += '<span class="flag">low confidence</span>'
    if is_dis:
        flags += '<span class="flag dis">models disagree</span>'
    picks = "".join(f'<button type="button" class="pick" data-v="{k}">{k}</button>' for k in LABELS)
    return f"""
<tr class="row" tabindex="-1" data-id="{cid}" data-label="{label}" data-conf="{conf:.4f}" data-low="{int(is_low)}" data-dis="{int(is_dis)}">
  <td class="id">{cid}</td>
  <td class="text"><div class="meta">{html.escape(c['author'])} · {html.escape(c['thread'])} · {html.escape(c['posted_at'])}</div>
    <div class="body">{html.escape(c['comment'])}</div><div class="flags">{flags}</div></td>
  {label_cell(main_answer)}{label_cell(other_answer) if other_answer is not None else ""}
  <td class="review">
    <div class="picks"><button type="button" class="pick agree" data-v="agree">Clef is right</button>{picks}</div>
    <label class="bl"><input type="checkbox" class="border"> borderline</label>
    <textarea class="why" rows="2" placeholder="Why? (needed if you disagree or it's borderline)"></textarea>
  </td>
</tr>"""


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Clef Moderation Review</title>
<style>
:root { --bg:#fff; --fg:#1d1d1f; --muted:#6b6b70; --line:#e5e5ea; --soft:#f6f6f8;
  --ok:#1f7a3a; --okbg:#e3f4e8; --tox:#b42318; --toxbg:#fdeceb; --spam:#8a5100; --spambg:#fff3dc; --accent:#3655d4; }
@media (prefers-color-scheme: dark) { :root { --bg:#16161a; --fg:#ececf1; --muted:#9a9aa3; --line:#2c2c33; --soft:#1e1e24;
  --ok:#5fd38a; --okbg:#173322; --tox:#ff8a80; --toxbg:#3a1a19; --spam:#ffc266; --spambg:#3a2c12; --accent:#8ea2ff; } }
* { box-sizing:border-box; }
body { background:var(--bg); color:var(--fg); font:15px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; margin:0; padding:16px; }
.wrap { max-width:1200px; margin:0 auto; }
h1 { font-size:22px; margin:0 0 4px; }
.sub { color:var(--muted); margin-bottom:8px; }
.how { background:var(--soft); border:1px solid var(--line); border-radius:8px; padding:8px 12px; font-size:14px; margin:8px 0; }
.how kbd { border:1px solid var(--line); border-radius:4px; padding:0 4px; font-size:12px; background:var(--bg); }
.controls { display:flex; flex-wrap:wrap; gap:8px; align-items:center; position:sticky; top:0; background:var(--bg);
  padding:8px 0; border-bottom:1px solid var(--line); z-index:2; }
button { font:inherit; font-size:14px; border:1px solid var(--line); background:var(--soft); color:var(--fg);
  border-radius:999px; padding:4px 12px; cursor:pointer; }
button.on { background:var(--fg); color:var(--bg); border-color:var(--fg); }
button.primary { background:var(--accent); color:#fff; border-color:var(--accent); margin-left:auto; }
#progress { color:var(--muted); font-size:14px; }
#meter { flex-basis:100%; height:4px; background:var(--line); border-radius:2px; overflow:hidden; }
#meter span { display:block; height:100%; width:0; background:var(--accent); }
table { width:100%; border-collapse:collapse; margin-top:4px; }
th { text-align:left; font-size:13px; color:var(--muted); font-weight:600; padding:8px 6px; border-bottom:1px solid var(--line); }
td { border-bottom:1px solid var(--line); padding:10px 6px; vertical-align:top; }
tr.changed td { background:var(--soft); }
tr.cur td { box-shadow: inset 0 2px 0 var(--accent), inset 0 -2px 0 var(--accent); }
tr.cur td:first-child { box-shadow: inset 3px 0 0 var(--accent), inset 0 2px 0 var(--accent), inset 0 -2px 0 var(--accent); }
tr:focus { outline:none; }
.id { color:var(--muted); width:34px; text-align:right; }
tr.done .id::after { content:" ✓"; color:var(--ok); }
.meta { color:var(--muted); font-size:13px; margin-bottom:2px; }
.flags { margin-top:4px; }
.flag { display:inline-block; font-size:12px; color:var(--muted); border:1px solid var(--line); border-radius:999px; padding:0 8px; margin-right:4px; }
.flag.dis { color:var(--accent); border-color:var(--accent); }
.lab { width:120px; }
.badge { display:inline-block; font-weight:600; font-size:13px; padding:2px 8px; border-radius:6px; }
.badge.acceptable { color:var(--ok); background:var(--okbg); }
.badge.toxic { color:var(--tox); background:var(--toxbg); }
.badge.spam { color:var(--spam); background:var(--spambg); }
.bar { display:flex; height:6px; border-radius:3px; overflow:hidden; background:var(--line); margin:6px 0 2px; }
.p.acceptable { background:var(--ok); } .p.toxic { background:var(--tox); } .p.spam { background:var(--spam); }
small { color:var(--muted); }
.review { width:250px; }
.picks { display:flex; flex-wrap:wrap; gap:4px; }
.pick { font-size:13px; padding:2px 10px; }
.pick.on { background:var(--fg); color:var(--bg); border-color:var(--fg); }
.pick.agree.on { background:var(--ok); border-color:var(--ok); color:var(--bg); }
textarea { font:inherit; font-size:14px; width:100%; background:var(--bg); color:var(--fg); border:1px solid var(--line); border-radius:6px; padding:4px; }
textarea.need { border-color:var(--tox); }
.bl { display:block; font-size:13px; color:var(--muted); margin:4px 0; }
.hidden { display:none; }
body.blind tr:not(.done) .lab .badge, body.blind tr:not(.done) .lab .bar, body.blind tr:not(.done) .lab small,
body.blind tr:not(.done) .flag, body.blind .pick.agree { visibility:hidden; }
@media (max-width:760px) {
  table, tbody, tr, td { display:block; width:100%; } thead { display:none; }
  tr { border-bottom:1px solid var(--line); padding:8px 0; } td { border:0; padding:4px 0; }
  .id { text-align:left; } .lab, .review { width:100%; } }
</style></head>
<body><div class="wrap">
<h1>Clef moderation review</h1>
<div class="sub">__SUBTITLE__</div>
<div class="how">Read every comment and decide its label. Tap <b>Clef is right</b>, or tap your own label and write why.
Tick <i>borderline</i> for close calls. Turn on <b>Hide labels</b> to decide before you see Clef's answer; each row's labels
appear once you've decided. Answers save in this browser as you go. When all are reviewed, click <b>Download MISTAKES.md</b>.<br>
Keys: <kbd>j</kbd>/<kbd>k</kbd> next/previous · <kbd>a</kbd> Clef is right · <kbd>1</kbd> acceptable · <kbd>2</kbd> toxic ·
<kbd>3</kbd> spam · <kbd>b</kbd> borderline · <kbd>w</kbd> write reason · <kbd>Esc</kbd> leave the box · <kbd>n</kbd> next unreviewed</div>
<div class="controls">
  <button class="on" data-f="all">All</button>
  <button data-f="todo">Not reviewed</button>
  <button data-f="acceptable">Acceptable</button>
  <button data-f="toxic">Toxic</button>
  <button data-f="spam">Spam</button>
  <button data-f="low">Low confidence</button>
  __DIS_BUTTON__
  <button data-f="changed">My changes</button>
  <button id="sort">Sort: by id</button>
  <button id="blind">Hide labels</button>
  <span id="progress"></span>
  <button class="primary" id="export">Download MISTAKES.md</button>
  <div id="meter"><span></span></div>
</div>
<table><thead><tr><th>#</th><th>Comment</th>__HEADERS__<th>Your review</th></tr></thead>
<tbody id="rows">__ROWS__</tbody></table>
</div>
<script>
const KEY = 'clef-review-v2', MODEL = '__MODEL__';
const LABELS = ['acceptable', 'toxic', 'spam'];
const tbody = document.getElementById('rows');
const rows = [...tbody.querySelectorAll('tr')];
let saved = {};
try { saved = JSON.parse(localStorage.getItem(KEY) || '{}'); } catch (e) {}
const store = () => { try { localStorage.setItem(KEY, JSON.stringify(saved)); } catch (e) {} };
const get = r => saved[r.dataset.id] || { mine: '', border: false, why: '' };
const isChanged = r => { const s = get(r); return (s.mine && s.mine !== r.dataset.label) || s.border; };
const needsWhy = r => isChanged(r) && !get(r).why.trim();

function paint(r) {
  const s = get(r);
  r.querySelectorAll('.pick').forEach(b => {
    const v = b.dataset.v === 'agree' ? r.dataset.label : b.dataset.v;
    b.classList.toggle('on', !!s.mine && (b.dataset.v === 'agree' ? s.mine === r.dataset.label : s.mine === v && v !== r.dataset.label));
  });
  r.querySelector('.border').checked = !!s.border;
  const why = r.querySelector('.why');
  if (document.activeElement !== why) why.value = s.why || '';
  why.classList.toggle('need', needsWhy(r));
  r.classList.toggle('done', !!s.mine);
  r.classList.toggle('changed', isChanged(r));
}
function refresh() {
  const done = rows.filter(r => get(r).mine).length, marked = rows.filter(isChanged).length;
  document.getElementById('progress').textContent = done + '/' + rows.length + ' reviewed · ' + marked + ' marked';
  document.querySelector('#meter span').style.width = (100 * done / rows.length) + '%';
}
function update(r, patch) {
  saved[r.dataset.id] = Object.assign({}, get(r), patch);
  store(); paint(r); refresh();
}
function choose(r, v) {
  if (v === 'agree') {
    if (document.body.classList.contains('blind')) return;
    v = r.dataset.label;
  }
  update(r, { mine: v });
  if (v === r.dataset.label && !get(r).border) move(1);
  else r.querySelector('.why').focus();
}

let cur = null;
const visible = () => [...tbody.querySelectorAll('tr:not(.hidden)')];
function setCur(r, scroll) {
  if (cur) cur.classList.remove('cur');
  cur = r || null;
  if (!cur) return;
  cur.classList.add('cur');
  if (scroll) cur.scrollIntoView({ block: 'center', behavior: 'smooth' });
}
function move(step) {
  const v = visible(); if (!v.length) return;
  const i = cur ? v.indexOf(cur) : -1;
  setCur(v[Math.max(0, Math.min(v.length - 1, i + step))], true);
}
function nextTodo() {
  const v = visible(), i = cur ? v.indexOf(cur) : -1;
  const next = v.slice(i + 1).concat(v.slice(0, i + 1)).find(r => !get(r).mine);
  if (next) setCur(next, true);
}

rows.forEach(r => {
  paint(r);
  r.querySelectorAll('.pick').forEach(b => b.addEventListener('click', () => { setCur(r); choose(r, b.dataset.v); }));
  r.querySelector('.border').addEventListener('change', e => { setCur(r); update(r, { border: e.target.checked }); });
  r.querySelector('.why').addEventListener('input', e => { setCur(r); update(r, { why: e.target.value }); });
  r.addEventListener('click', e => { if (!e.target.closest('button, input, textarea, label')) setCur(r); });
});
refresh();

document.addEventListener('keydown', e => {
  if (e.target.tagName === 'TEXTAREA') { if (e.key === 'Escape') { e.target.blur(); cur && cur.focus(); } return; }
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  const k = e.key.toLowerCase();
  if (k === 'j') return move(1);
  if (k === 'k') return move(-1);
  if (k === 'n') return nextTodo();
  if (!cur) return;
  if (k === 'a') choose(cur, 'agree');
  else if (k === '1' || k === '2' || k === '3') choose(cur, LABELS[+k - 1]);
  else if (k === 'b') update(cur, { border: !get(cur).border });
  else if (k === 'w' || k === 'enter') { e.preventDefault(); cur.querySelector('.why').focus(); }
});

document.querySelectorAll('[data-f]').forEach(b => b.onclick = () => {
  document.querySelectorAll('[data-f]').forEach(x => x.classList.toggle('on', x === b));
  const f = b.dataset.f;
  rows.forEach(r => r.classList.toggle('hidden', !(f === 'all' || r.dataset.label === f
    || (f === 'todo' && !get(r).mine) || (f === 'low' && r.dataset.low === '1')
    || (f === 'dis' && r.dataset.dis === '1') || (f === 'changed' && isChanged(r)))));
});
let byConf = false;
document.getElementById('sort').onclick = e => {
  byConf = !byConf;
  e.target.textContent = byConf ? 'Sort: least confident first' : 'Sort: by id';
  const key = r => byConf ? +r.dataset.conf : +r.dataset.id;
  rows.slice().sort((a, b) => key(a) - key(b)).forEach(r => tbody.appendChild(r));
};
document.getElementById('blind').onclick = e => {
  const on = document.body.classList.toggle('blind');
  e.target.textContent = on ? 'Show labels' : 'Hide labels';
};
document.getElementById('export').onclick = () => {
  const todo = rows.filter(r => !get(r).mine).map(r => r.dataset.id).sort((a, b) => a - b);
  const noWhy = rows.filter(needsWhy).map(r => r.dataset.id).sort((a, b) => a - b);
  const warn = [];
  if (todo.length) warn.push(todo.length + ' comment(s) not reviewed yet: ' + todo.slice(0, 15).join(', ') + (todo.length > 15 ? '…' : ''));
  if (noWhy.length) warn.push('No reason written for: ' + noWhy.join(', '));
  if (warn.length && !confirm(warn.join('\n') + '\n\nDownload anyway?')) return;
  const marked = rows.filter(isChanged).sort((a, b) => a.dataset.id - b.dataset.id);
  let md = "# Mistakes in Clef's labels\n\nModel reviewed: " + MODEL + '\n';
  marked.forEach(r => {
    const s = get(r), clef = r.dataset.label, mine = s.mine || clef;
    const note = s.border ? (mine === clef ? " (borderline, I kept Clef's label)" : ' (borderline)') : '';
    const text = r.querySelector('.body').textContent.replace(/\s+/g, ' ').trim();
    md += '\n## Comment ' + r.dataset.id + note + '\n\n> ' + text + '\n\n';
    md += '- Clef: ' + clef + '\n- Mine: ' + mine + '\n- Why: ' + (s.why || '').trim().replace(/\s*\n\s*/g, ' ') + '\n';
  });
  if (!marked.length) md += '\nI found no mistakes.\n';
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([md], { type: 'text/markdown' }));
  a.download = 'MISTAKES.md'; a.click();
};
</script>
</body></html>"""


def render_html(meta, comments, results):
    models = [m for m in MODELS if m in results]
    main = models[0]  # the larger model's label is the one under review
    other = models[1] if len(models) > 1 else None
    counts = {k: 0 for k in LABELS}
    low = disagree = 0
    rows = []
    for c in comments:
        a = results[main][str(c["id"])]
        b = results[other][str(c["id"])] if other else None
        counts[a["choice"]] = counts.get(a["choice"], 0) + 1
        low += a["confidence"] < LOW_CONFIDENCE
        disagree += b is not None and b["choice"] != a["choice"]
        rows.append(render_row(c, a, b))

    subtitle = (f"{html.escape(meta.get('course', ''))} · {html.escape(meta.get('forum', ''))}<br>"
                f"Label under review: <b>{html.escape(main)}</b> · {len(comments)} comments · "
                + " · ".join(f"<b>{counts.get(k, 0)}</b> {k}" for k in LABELS)
                + f" · <b>{low}</b> below {LOW_CONFIDENCE} confidence")
    if other:
        subtitle += f" · <b>{disagree}</b> where the two models disagree"
    headers = f"<th>{html.escape(main)}</th>" + (f"<th>{html.escape(other)}</th>" if other else "")
    replacements = {
        "__SUBTITLE__": subtitle,
        "__DIS_BUTTON__": '<button data-f="dis">Models disagree</button>' if other else "",
        "__HEADERS__": headers,
        "__MODEL__": main,
        "__ROWS__": "".join(rows),
    }
    page = PAGE
    for token, value in replacements.items():
        page = page.replace(token, value)
    return page


def setup():
    """Ask for the account ID and token and write cloud-flare.local.sh."""
    import getpass
    if CREDS_FILE.exists() and input(f"{CREDS_FILE.name} already exists. Replace it? [y/N] ").strip().lower() != "y":
        return
    account = input("Cloudflare account ID (32 letters and digits): ").strip()
    if not re.fullmatch(r"[0-9a-fA-F]{32}", account):
        sys.exit("That doesn't look like an account ID. Copy it from Workers AI > Use REST API.")
    token = getpass.getpass("API token (hidden; on Windows, right-click to paste): ").strip()
    if len(token) < 20 or '"' in token or " " in token:
        sys.exit("That doesn't look like an API token. Copy it again from Cloudflare.")
    CREDS_FILE.write_text(f'export CLOUDFLARE_ACCOUNT_ID="{account}"\n'
                          f'export CLOUDFLARE_API_TOKEN="{token}"\n', encoding="utf-8")
    print(f"Saved {CREDS_FILE.name}. Now run: python classify.py --check")


def run_git(*args):
    import subprocess
    try:
        r = subprocess.run(["git", *args], cwd=HERE, capture_output=True, text=True, timeout=30)
        return r.returncode, r.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None, ""


def doctor(comments):
    """Check everything Navin's README asks for before you submit."""
    problems = 0

    def say(ok, text, fix=""):
        nonlocal problems
        problems += not ok
        print(("  OK   " if ok else "  FIX  ") + text + ("" if ok or not fix else f"\n         -> {fix}"))

    print("Setup")
    say(sys.version_info >= (3, 8), f"Python {sys.version.split()[0]}", "Install Python 3.8 or newer")
    say(len(comments) == 100, f"comments.json has {len(comments)} comments")
    token = ""
    if CREDS_FILE.exists():
        creds = dict(re.findall(r'export (\w+)="([^"]*)"', CREDS_FILE.read_text(encoding="utf-8")))
        token = creds.get("CLOUDFLARE_API_TOKEN", "")
        say(bool(token) and "your-cloudflare" not in token, f"{CREDS_FILE.name} has a token",
            "Run: python classify.py --setup")
    else:
        say(False, f"{CREDS_FILE.name} exists", "Run: python classify.py --setup")
    template = HERE / "cloud-flare.sh"
    if template.exists():
        say("your-cloudflare-api-token" in template.read_text(encoding="utf-8"),
            "cloud-flare.sh still has the sample values (it gets committed)",
            "Put the sample values back in cloud-flare.sh; your real token belongs only in the .local file")

    print("Results")
    for m in MODELS:
        n = len(load_cache(m))
        say(n == len(comments), f"{cache_path(m).name}: {n}/{len(comments)} answers", "Run: python classify.py")
    say(HTML_FILE.exists(), HTML_FILE.name + " exists", "Run: python classify.py")

    print("Your files")
    mistakes = HERE / "MISTAKES.md"
    if mistakes.exists():
        text = mistakes.read_text(encoding="utf-8")
        entries = re.findall(r"^## Comment (\d+)", text, flags=re.M)
        empty = re.findall(r"^## Comment (\d+)[^\n]*\n(?:(?!^## ).*\n)*?- Why:\s*$", text, flags=re.M)
        say(True, f"MISTAKES.md has {len(entries)} entries")
        say(not empty, "every MISTAKES.md entry has a reason", f"Write a reason for comment(s) {', '.join(empty)}")
    else:
        say(False, "MISTAKES.md exists", "Review results.html and click Download MISTAKES.md")
    notes = HERE / "NOTES.md"
    say(notes.exists() and "[Write 3 to 5 lines" not in notes.read_text(encoding="utf-8"),
        "NOTES.md has your own 'What I felt' section", "Replace the [Write 3 to 5 lines...] placeholder in NOTES.md")

    print("Git and safety")
    code, _ = run_git("rev-parse", "--is-inside-work-tree")
    if code is None:
        say(False, "git is installed", "Install git or use GitHub Desktop, then run --doctor again")
    elif code != 0:
        say(False, "this folder is a git clone", "Clone your copy of the repo and put these files in it")
    else:
        code, _ = run_git("check-ignore", "-q", CREDS_FILE.name)
        say(code == 0, f"{CREDS_FILE.name} is ignored by git", "Add it to .gitignore before committing anything")
        code, _ = run_git("ls-files", "--error-unmatch", CREDS_FILE.name)
        say(code != 0, f"{CREDS_FILE.name} is not committed", "Remove it: git rm --cached cloud-flare.local.sh")
        if token and "your-cloudflare" not in token:
            _, hits = run_git("log", "--all", "--format=%h", "-S", token)
            say(not hits, "your token is not in any commit",
                "Your token was committed. Delete it in Cloudflare now and make a new one")
        _, remote = run_git("remote", "get-url", "origin")
        say("ngkabra/" not in remote, f"origin is your copy ({remote or 'none'})",
            "You cloned Navin's repo. Clone your own copy made with Use this template")
        _, status = run_git("status", "--porcelain")
        pending = [line[3:] for line in status.splitlines()
                   if line[3:] in ("classify.py", "results.html", "results-clef.json",
                                   "results-clef-flash.json", "MISTAKES.md", "NOTES.md")]
        say(not pending, "all six files are committed", "Commit: " + ", ".join(pending))
        _, branch = run_git("status", "-sb")
        say("ahead" not in branch.splitlines()[0] if branch else False, "everything is pushed to GitHub",
            "Run: git push (or Push origin in GitHub Desktop)")

    print("\nAll good. Email the repo link to navin@genwise.in." if not problems
          else f"\n{problems} thing(s) to fix before you submit.")


def print_summary(comments, results):
    for m, res in results.items():
        counts = {k: sum(1 for a in res.values() if a["choice"] == k) for k in LABELS}
        low = sum(1 for a in res.values() if a["confidence"] < LOW_CONFIDENCE)
        print(f"  {m}: " + ", ".join(f"{v} {k}" for k, v in counts.items()) + f"; {low} low confidence")
    if len(results) == 2:
        a, b = results.values()
        diff = sum(1 for c in comments if a[str(c["id"])]["choice"] != b[str(c["id"])]["choice"])
        print(f"  The two models disagree on {diff} comments.")


def main():
    parser = argparse.ArgumentParser(description="Classify forum comments with Cloudflare Clef and build results.html.")
    parser.add_argument("--model", default="both", choices=["both"] + MODELS)
    parser.add_argument("--refresh", action="store_true", help="call the API again for every comment")
    parser.add_argument("--check", action="store_true", help="test credentials (2 requests)")
    parser.add_argument("--setup", action="store_true", help="save your account ID and token")
    parser.add_argument("--doctor", action="store_true", help="check everything before you submit")
    parser.add_argument("--html-only", action="store_true", help="rebuild results.html from saved results")
    parser.add_argument("--workers", type=int, default=4, help="requests in parallel (default 4)")
    args = parser.parse_args()

    if args.setup:
        return setup()
    data = json.loads(COMMENTS_FILE.read_text(encoding="utf-8"))
    comments = data["comments"]
    if args.doctor:
        return doctor(comments)
    models = MODELS if args.model == "both" else [args.model]

    try:
        if args.check:
            account, token = load_creds()
            for m in models:
                answer = ask_clef(comments[0], account, token, m)
                print(f"OK: {m} answered ({answer['choice']}, confidence {answer['confidence']:.2f}).")
            print("Credentials work. Now run: python classify.py")
            return

        results = {}
        for m in models:
            if args.html_only:
                results[m] = load_cache(m)
            else:
                results[m] = classify_all(comments, m, args.refresh, args.workers)
            missing = [c["id"] for c in comments if str(c["id"]) not in results[m]]
            if missing:
                sys.exit(f"{m}: no saved result for comments {missing[:10]}. Run without --html-only.")
    except FatalError as e:
        sys.exit(f"Stopped: {e}")
    except KeyboardInterrupt:
        sys.exit("Stopped. Answers received so far are saved; run again to continue.")

    HTML_FILE.write_text(render_html(data, comments, results), encoding="utf-8")
    print_summary(comments, results)
    print(f"Wrote {HTML_FILE.name}. Open it in your browser.")


if __name__ == "__main__":
    main()
