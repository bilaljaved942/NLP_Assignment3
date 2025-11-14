"""
Part 1 — Grammar Correction and Summary Refinement (Assignment 3)
---------------------------------------------------------------
This script implements a robust grammar-correction + POS-based refinement
pipeline that follows the assignment instructions.

Features:
- Loads abstractive summaries (JSON with records containing `case_id` and `abstractive_summary`).
- Applies a hybrid pipeline:
    1) optional LanguageTool grammar correction (if installed),
    2) POS- and dependency-aware refinements using spaCy,
    3) conservative rule-based fallbacks for common legal patterns.
- POS-based fixes implemented:
    - remove consecutive duplicate tokens
    - safe possessive normalization for patterns like "defendant case" -> "defendant's case"
    - insert missing definite article "the" for common legal nouns when appropriate
    - subject-verb correction for simple third-person/singular present mistakes (e.g., "court dismiss" -> "court dismissed")
    - special-case passive phrasing for patterns like "appeal file" -> "appeal was filed"
- Produces outputs:
    - CSV and JSON comparison table: CaseId, original, corrected
    - reflection_part1.txt with process description

Usage (example):
    python part1_grammar_fixed.py --input summaries/abstractive_summaries.json --outdir assignment3_outputs --use_lt

Notes:
- The script prefers to use the spaCy model 'en_core_web_sm'. If spaCy or the model
  is missing, the script will exit with an informative error. LanguageTool is optional.
- The POS-based fixes are intentionally conservative to avoid changing legal meaning.
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd

# Try importing spaCy and language_tool
try:
    import spacy
except Exception as e:
    raise ImportError("spaCy is required for this script. Please install spaCy and the English model: `pip install spacy && python -m spacy download en_core_web_sm`")

try:
    import language_tool_python
    LT_AVAILABLE = True
except Exception:
    LT_AVAILABLE = False


# -------------------- Utilities --------------------
IRREGULAR_PAST = {
    'be': 'was', 'have': 'had', 'do': 'did', 'seek': 'sought', 'go': 'went',
    'bring': 'brought', 'take': 'took', 'come': 'came', 'see': 'saw', 'make': 'made',
    'file': 'filed', 'dismiss': 'dismissed', 'grant': 'granted', 'deny': 'denied'
}

LEGAL_NOUNS = set(["decision", "verdict", "case", "appeal", "judgment", "order", "damages", "claim", "petition"])

# safe simple verb past tense converter (conservative)
def to_past_simple(lemma: str, token_text: str) -> str:
    """Return a past-tense form for a verb lemma or fall back to token_text + 'ed'.
    This is intentionally conservative; irregulars in IRREGULAR_PAST are handled.
    """
    lemma_l = lemma.lower()
    if lemma_l in IRREGULAR_PAST:
        return IRREGULAR_PAST[lemma_l]
    # if token_text already ends with 'ed' or 'd', keep it
    if token_text.lower().endswith(('ed', 'd')):
        return token_text
    # simple rules for adding -ed (handle final 'e')
    if lemma_l.endswith('e'):
        return lemma_l + 'd'
    return lemma_l + 'ed'


# -------------------- Core Refinement Logic --------------------

def apply_language_tool(text: str, lt_tool) -> str:
    """Run LanguageTool correction if lt_tool is provided."""
    if lt_tool is None:
        return text
    try:
        matches = lt_tool.check(text)
        corrected = language_tool_python.utils.correct(text, matches)
        return corrected
    except Exception:
        return text


def pos_refinement(text: str, nlp, conservative: bool = True) -> str:
    """Apply POS- and dependency-aware refinements using spaCy.

    The function constructs a new token list (conservative edits only) to avoid
    damaging legal meaning. Edits performed:
      - remove consecutive duplicate tokens (case-insensitive)
      - possessive normalization for patterns: <X> case -> <X>'s case
      - insert 'the' before legal nouns when missing determiners
      - subject-verb agreement fixes for simple cases: singular subj + bare verb -> past form
      - special-case: 'appeal file' -> 'appeal was filed'
    """
    doc = nlp(text)

    out_tokens: List[str] = []
    prev_tok_text: Optional[str] = None

    # gather token attributes to allow lookahead and lookbehind
    tokens = list(doc)
    L = len(tokens)

    i = 0
    while i < L:
        tok = tokens[i]
        ttext = tok.text

        # 1) Remove consecutive duplicates (case-insensitive), keep punctuation
        if prev_tok_text is not None and ttext.isalpha() and prev_tok_text.isalpha() and ttext.lower() == prev_tok_text.lower():
            # skip duplicate
            i += 1
            continue

        # 2) Possessive pattern: [PROPN|NOUN] "case"  -> "X's case"
        if i + 1 < L:
            tok_next = tokens[i + 1]
            if tok_next.text.lower() == 'case' and tok.pos_ in ('PROPN', 'NOUN'):
                # Avoid if already possessive or token includes apostrophe
                if not re.search(r"'s$", tok.text):
                    out_tokens.append(tok.text + "'s")
                else:
                    out_tokens.append(tok.text)
                out_tokens.append('case')
                prev_tok_text = 'case'
                i += 2
                continue

        # 3) Insert 'the' before legal nouns when appropriate
        if tok.pos_ == 'NOUN' and tok.lemma_.lower() in LEGAL_NOUNS:
            # check previous token for determiner or possessive
            prev = tokens[i - 1] if i - 1 >= 0 else None
            if prev is None or prev.dep_ in ('ROOT', 'prep', 'cc') or prev.pos_ not in ('DET', 'PRON', 'AUX', 'ADJ', 'NOUN', 'PROPN'):
                # very conservative: insert 'the' only if noun is not part of a compound with determiner
                out_tokens.append('the')
        
        # 4) Special-case: 'appeal file' -> 'appeal was filed'
        if tok.lemma_.lower() == 'appeal' and i + 1 < L and tokens[i + 1].lemma_.lower() == 'file':
            out_tokens.append('The' if tok.is_title else tok.text)
            out_tokens.append('was')
            out_tokens.append('filed')
            i += 2
            prev_tok_text = 'filed'
            continue

        # 5) Subject-verb correction (simple pattern): subj (NN/NNP) -> verb (VB/VBP) -> convert verb to past
        #    if token is a verb and has nominal subject that is singular
        if tok.pos_ == 'VERB' and tok.dep_ in ('ROOT', 'advcl', 'ccomp', 'xcomp', 'aux'):
            # find nominal subject child
            nsubj = None
            for ch in tok.children:
                if ch.dep_ in ('nsubj', 'nsubjpass'):
                    nsubj = ch
                    break
            if nsubj and nsubj.tag_ in ('NN', 'NNP', 'NNPS'):  # singular/proper noun
                # if verb is base form or present plural (VBP/VB) -> convert to past simple
                if tok.tag_ in ('VBP', 'VB'):
                    past = to_past_simple(tok.lemma_, tok.text)
                    out_tokens.append(past)
                    prev_tok_text = past
                    i += 1
                    continue
        
        # 6) Normal token: reproduce text preserving whitespace by relying on spaCy spacing when later joining
        out_tokens.append(ttext)
        prev_tok_text = ttext
        i += 1

    # Reconstruct sentence using a conservative spacing algorithm: join tokens with single space, then fix spacing before punctuation
    s = ' '.join(out_tokens)
    # Fix common spacing with punctuation
    s = re.sub(r"\s+([,.;:?!])", r"\1", s)
    # Fix spaces after opening brackets
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)

    # ensure sentence capitalization and ending punctuation
    s = s.strip()
    if s and not re.search(r"[.!?]$", s):
        s = s + '.'
    if s:
        s = s[0].upper() + s[1:]

    return s


# -------------------- High-level pipeline --------------------

def refine_summary_text(text: str, nlp, lt_tool: Optional[object] = None) -> str:
    """Apply LanguageTool (optional) followed by POS-based and rule-based refinements."""
    if not isinstance(text, str):
        text = str(text or '')
    t = text.strip()
    if not t:
        return t

    # Stage 1: LanguageTool (if available)
    if lt_tool is not None:
        t = apply_language_tool(t, lt_tool)

    # Stage 2: POS-based refinement
    t = pos_refinement(t, nlp)

    # Stage 3: Additional conservative regex-based fixes (legal phrasing small patterns)
    # e.g., 'file against decision' -> 'filed against the decision'
    t = re.sub(r"\bfile against\b", "filed against", t, flags=re.IGNORECASE)
    t = re.sub(r"\bappeal file against decision\b", "The appeal was filed against the decision", t, flags=re.IGNORECASE)
    t = re.sub(r"\bdismiss the\b", "dismissed the", t, flags=re.IGNORECASE)

    # Ensure no duplicate spaces
    t = re.sub(r"\s+", " ", t).strip()
    # Ensure final punctuation
    if t and not re.search(r"[.!?]$", t):
        t = t + '.'

    return t


# -------------------- CLI and I/O --------------------

def load_json_summaries(path: Path) -> List[Dict]:
    with path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def save_outputs(rows: List[Dict], outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    csv_out = outdir / 'abstractive_corrections_comparison.csv'
    json_out = outdir / 'abstractive_corrections.json'
    df.to_csv(csv_out, index=False)
    df.to_json(json_out, orient='records', force_ascii=False, indent=2)

    # reflection
    reflection = (
        "Reflection (Part 1):\n"
        "We applied a hybrid pipeline combining LanguageTool grammar correction (if available)\n"
        "with POS- and dependency-aware refinements using spaCy. The POS stage handles\n"
        "conservative, targeted changes: duplicate removal, safe possessive normalization,\n"
        "insertions of definite article for specific legal nouns, and simple subject-verb\n"
        "corrections by converting bare present verbs to past where a singular subject\n"
        "was detected. These edits are conservative to avoid altering legal meaning."
    )
    with (outdir / 'reflection_part1.txt').open('w', encoding='utf-8') as f:
        f.write(reflection)

    print(f"Saved CSV -> {csv_out}")
    print(f"Saved JSON -> {json_out}")
    print(f"Saved reflection -> {outdir / 'reflection_part1.txt'}")


def main(args):
    input_path = Path(args.input)
    outdir = Path(args.outdir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # load spaCy model
    nlp = spacy.load('en_core_web_sm')

    # init LanguageTool if requested and available
    lt_tool = None
    if args.use_lt:
        if not LT_AVAILABLE:
            print("language_tool_python not installed — continuing without it.")
        else:
            try:
                lt_tool = language_tool_python.LanguageTool('en-US')
                print('LanguageTool initialized.')
            except Exception as e:
                print('LanguageTool initialization failed:', e)
                lt_tool = None

    data = load_json_summaries(input_path)
    rows = []
    for rec in data:
        cid = rec.get('case_id') or rec.get('CaseNo') or rec.get('caseNo') or ''
        original = rec.get('abstractive_summary') or rec.get('summary') or rec.get('text') or ''
        corrected = refine_summary_text(original, nlp, lt_tool)
        rows.append({'case_id': cid, 'original': original, 'corrected': corrected})

    save_outputs(rows, outdir)
    print('Part 1 completed.')


if __name__ == '__main__':
    # Simplified: no CLI arguments
    input_path = Path('summaries/abstractive_summaries.json')
    outdir = Path('assignment3_outputs')

    # load spaCy model
    nlp = spacy.load('en_core_web_sm')

    # Try initializing LanguageTool; fallback if fails
    lt_tool = None
    if LT_AVAILABLE:
        try:
            lt_tool = language_tool_python.LanguageTool('en-US')
            print('LanguageTool initialized.')
        except Exception as e:
            print('LanguageTool initialization failed:', e)
            lt_tool = None

    data = load_json_summaries(input_path)
    rows = []
    for rec in data:
        cid = rec.get('case_id') or rec.get('CaseNo') or rec.get('caseNo') or ''
        original = rec.get('abstractive_summary') or rec.get('summary') or rec.get('text') or ''
        corrected = refine_summary_text(original, nlp, lt_tool)
        rows.append({'case_id': cid, 'original': original, 'corrected': corrected})

    save_outputs(rows, outdir)
    print('Part 1 completed.')