
from __future__ import annotations
from dataclasses import dataclass
from collections import Counter, defaultdict
from typing import Dict, Tuple, List, FrozenSet, Optional, Any

Tag = FrozenSet[str]
Form = str  # includes < and >
Lexeme = str

@dataclass(frozen=True)
class TagDelta:
    adds: FrozenSet[str]
    rems: FrozenSet[str]

    def apply(self, tags: Tag) -> Tag:
        return frozenset((tags - self.rems) | self.adds)

    def __str__(self) -> str:
        a = "{" + ",".join(sorted(self.adds)) + "}"
        r = "{" + ",".join(sorted(self.rems)) + "}"
        return f"+{a} -{r}"

@dataclass(frozen=True)
class Script:
    # Replace the substring between prefix length p and suffix length q
    # with 'ins' (can be empty). Works for any source string with len>=p+q.
    p: int
    q: int
    ins: str

    def apply(self, src: str) -> Optional[str]:
        if len(src) < self.p + self.q:
            return None
        return src[:self.p] + self.ins + src[-self.q:] if self.q > 0 else src[:self.p] + self.ins

    def __str__(self) -> str:
        # compact representation
        mid = self.ins.replace("\n", "\\n")
        return f"[keep {self.p} | ins '{mid}' | keep_right {self.q}]"

def longest_common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i

def longest_common_suffix_len(a: str, b: str, max_drop_left: int) -> int:
    # Ensure prefix and suffix don't overlap; max_drop_left = prefix length
    i = 0
    while i < len(a) - max_drop_left and i < len(b) - max_drop_left and a[-(i+1)] == b[-(i+1)]:
        i += 1
    return i

def derive_script(src: str, tgt: str) -> Script:
    p = longest_common_prefix_len(src, tgt)
    q = longest_common_suffix_len(src, tgt, p)
    ins = tgt[p: len(tgt) - q] if q > 0 else tgt[p:]
    return Script(p=p, q=q, ins=ins)

def symmetric_diff_size(a: Tag, b: Tag) -> int:
    return len(a ^ b)

class ImplicativePathModel:
    """
    Implicative-Path Analogies on the Tag Lattice.
    - Learns edge scripts grouped by tag-deltas (adds, rems).
    - Scores each script by cross-lexeme reliability (leave-donor-out).
    - At inference, searches short paths that reduce tag distance and composes scripts.
    """
    def __init__(self, max_depth: int = 3, beam_size: int = 6, top_k_per_edge: int = 3, smoothing: float = 1.0):
        self.max_depth = max_depth
        self.beam_size = beam_size
        self.top_k = top_k_per_edge
        self.smoothing = smoothing
        self.forms: Dict[Lexeme, Dict[Tag, Form]] = {}
        self.edge_scripts: Dict[TagDelta, List[Tuple[Script, float, int]]] = {}  # delta -> list of (script, reliability, support)
        self._deltas_seen: set[TagDelta] = set()

    @staticmethod
    def _canonicalize_forms(lexeme_dict: Dict[Lexeme, Dict[Tuple[Form, Tag], int]]) -> Dict[Lexeme, Dict[Tag, Form]]:
        res: Dict[Lexeme, Dict[Tag, Form]] = {}
        for L, counter in lexeme_dict.items():
            best: Dict[Tag, Tuple[Form, int]] = {}
            for (form, tag), freq in counter.items():
                if tag not in best or freq > best[tag][1] or (freq == best[tag][1] and form < best[tag][0]):
                    best[tag] = (form, freq)
            res[L] = {tag: form for tag, (form, _) in best.items()}
        return res

    def fit(self, lexeme_dict: Dict[Lexeme, Dict[Tuple[Form, Tag], int]]) -> "ImplicativePathModel":
        self.forms = self._canonicalize_forms(lexeme_dict)

        # Collect occurrences by delta
        by_delta_pairs: Dict[TagDelta, List[Tuple[Lexeme, Tag, Tag, Form, Form]]] = defaultdict(list)
        by_delta_scripts: Dict[TagDelta, List[Script]] = defaultdict(list)

        for L, t2f in self.forms.items():
            tags = list(t2f.keys())
            n = len(tags)
            for i in range(n):
                for j in range(n):
                    if i == j: 
                        continue
                    t, u = tags[i], tags[j]
                    adds = frozenset(u - t)
                    rems = frozenset(t - u)
                    delta = TagDelta(adds=adds, rems=rems)
                    s = derive_script(t2f[t], t2f[u])
                    by_delta_pairs[delta].append((L, t, u, t2f[t], t2f[u]))
                    by_delta_scripts[delta].append(s)

        # For each delta, estimate reliability per unique script via leave-donor-out across lexemes
        self.edge_scripts = {}
        for delta, pairs in by_delta_pairs.items():
            # Index donors of each script
            script_donors: Dict[Script, List[int]] = defaultdict(list)
            for idx, s in enumerate(by_delta_scripts[delta]):
                script_donors[s].append(idx)

            # Prepare arrays of sources and targets for fast testing
            sources = [src for (_, _, _, src, _) in pairs]
            targets = [tgt for (_, _, _, _, tgt) in pairs]

            scored: List[Tuple[Script, float, int]] = []
            total = len(pairs)
            for s, donor_idx_list in script_donors.items():
                donors = set(donor_idx_list)
                tests = total - len(donors)
                success = 0
                for i in range(total):
                    if i in donors:
                        continue
                    out = s.apply(sources[i])
                    if out == targets[i]:
                        success += 1
                # Laplace smoothing
                alpha = self.smoothing
                reliability = (success + alpha) / (tests + 2 * alpha) if tests > 0 else 1.0
                scored.append((s, reliability, len(donors)))

            # keep top-k by (reliability, support)
            scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
            self.edge_scripts[delta] = scored[: self.top_k]
            if scored:
                self._deltas_seen.add(delta)

        return self

    def _expand(self, tag: Tag) -> List[TagDelta]:
        # All deltas are available globally
        return list(self._deltas_seen)

    def predict(self, lexeme: Lexeme, target_tag: Tag, return_explanation: bool = False) -> Optional[str] | Tuple[Optional[str], Any]:
        if lexeme not in self.forms:
            return (None, {"reason": "unknown_lexeme"}) if return_explanation else None

        t2f = self.forms[lexeme]
        if target_tag in t2f:
            return (t2f[target_tag], {"path": [], "anchor": target_tag, "score": 1.0}) if return_explanation else t2f[target_tag]

        # Prepare initial states (one per observed anchor cell)
        init_states = []
        for t0, f0 in t2f.items():
            d0 = symmetric_diff_size(t0, target_tag)
            # state: (form, current_tag, score, path, distance, anchor_tag)
            init_states.append((f0, t0, 1.0, [], d0, t0))

        best_candidate = None  # (score, form, explanation)
        for _depth in range(self.max_depth):
            next_states = []
            for form, tag, score, path, dist, anchor in init_states:
                # expand with deltas that strictly reduce distance to target
                for delta in self._expand(tag):
                    new_tag = delta.apply(tag)
                    new_dist = symmetric_diff_size(new_tag, target_tag)
                    if new_dist >= dist:
                        continue
                    # try top scripts for this delta
                    for script, rel, support in self.edge_scripts.get(delta, []):
                        new_form = script.apply(form)
                        if new_form is None:
                            continue
                        new_score = score * rel
                        new_path = path + [(delta, script, rel, support)]
                        if new_tag == target_tag:
                            expl = {
                                "anchor": anchor,
                                "steps": [(str(d), str(s), rel, support) for (d, s, rel, support) in new_path],
                                "score": new_score,
                            }
                            if best_candidate is None or new_score > best_candidate[0]:
                                best_candidate = (new_score, new_form, expl)
                        else:
                            next_states.append((new_form, new_tag, new_score, new_path, new_dist, anchor))
            # prune beam
            next_states.sort(key=lambda x: x[2], reverse=True)
            init_states = next_states[: self.beam_size]
            if not init_states:
                break

        if best_candidate:
            if return_explanation:
                return best_candidate[1], best_candidate[2]
            return best_candidate[1]

        # fallback: try direct delta if available
        best_direct = None
        for t0, f0 in t2f.items():
            adds = frozenset(target_tag - t0)
            rems = frozenset(t0 - target_tag)
            delta = TagDelta(adds=adds, rems=rems)
            for script, rel, support in self.edge_scripts.get(delta, []):
                out = script.apply(f0)
                if out is not None:
                    score = rel
                    expl = {"anchor": t0, "steps": [(str(delta), str(script), rel, support)], "score": score}
                    if best_direct is None or score > best_direct[0]:
                        best_direct = (score, out, expl)
        if best_direct:
            return (best_direct[1], best_direct[2]) if return_explanation else best_direct[1]
        return (None, {"reason": "no_path"}) if return_explanation else None

# -------------------------- Minimal tutorial example --------------------------

def _demo() -> None:
    # A tiny toy lexeme_dict for illustration (Hungarian 'kalap' and a made-up lexeme 'sapka')
    lexeme_dict: Dict[Lexeme, Dict[Tuple[Form, Tag], int]] = {
        "kalap": Counter({
            ("<kalap>", frozenset({"Nom"})): 4,
            ("<kalapot>", frozenset({"Acc"})): 4,
            ("<kalapok>", frozenset({"Pl","Nom"})): 2,
            ("<kalapokat>", frozenset({"Pl","Acc"})): 1,
            ("<kalappal>", frozenset({"Ins"})): 1,
            ("<kalapunk>", frozenset({"Poss","Nom","1Pl"})): 1,
        }),
        "sapka": Counter({
            ("<sapka>", frozenset({"Nom"})): 5,
            ("<sapk\u00E1t>", frozenset({"Acc"})): 3,   # á
            ("<sapk\u00E1k>", frozenset({"Pl","Nom"})): 2,
            ("<sapk\u00E1kat>", frozenset({"Pl","Acc"})): 1,
        }),
    }

    model = ImplicativePathModel(max_depth=3, beam_size=6, top_k_per_edge=3, smoothing=1.0).fit(lexeme_dict)

    # Predict a missing cell: Instrumental of 'sapka' (not attested in training)
    pred, expl = model.predict("sapka", frozenset({"Ins"}), return_explanation=True)
    print("Prediction for sapka + {Ins}:", pred)
    print("Explanation (steps):")
    for step in expl.get("steps", []):
        print("  delta", step[0], "script", step[1], "rel", round(step[2], 3), "support", step[3])
    print("Score:", round(expl.get("score", 0.0), 4))

if __name__ == "__main__":
    _demo()
