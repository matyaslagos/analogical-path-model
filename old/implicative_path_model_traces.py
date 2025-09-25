
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
        mid = self.ins.replace("\\n", "\\\\n")
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

    def predict(self, lexeme: Lexeme, target_tag: Tag, return_explanation: bool = False, return_trace: bool = False, nbest: int = 1):
        """
        Predict a target form. If return_trace=True, also return a rich trace with
        all expansions by depth, beam evolution, and all completed paths.
        If nbest>1, return the top-n completed paths (forms) in score order.
        """
        trace = {"anchor_states": [], "depths": [], "completed_paths": []}

        if lexeme not in self.forms:
            if return_trace and return_explanation:
                return None, {"reason": "unknown_lexeme"}, trace
            if return_trace:
                return None, trace
            if return_explanation:
                return None, {"reason": "unknown_lexeme"}
            return None

        t2f = self.forms[lexeme]
        if target_tag in t2f:
            if return_trace and return_explanation:
                return t2f[target_tag], {"path": [], "anchor": target_tag, "score": 1.0}, trace
            if return_trace:
                return t2f[target_tag], trace
            if return_explanation:
                return t2f[target_tag], {"path": [], "anchor": target_tag, "score": 1.0}
            return t2f[target_tag]

        # Prepare initial states (one per observed anchor cell)
        init_states = []
        for t0, f0 in t2f.items():
            d0 = symmetric_diff_size(t0, target_tag)
            init_states.append((f0, t0, 1.0, [], d0, t0))  # (form, tag, score, path, dist, anchor)
            trace["anchor_states"].append({"tag": t0, "form": f0, "distance": d0})

        completed = []  # (score, form, steps, anchor)

        for depth in range(self.max_depth):
            depth_log = {"depth": depth, "expansions": [], "beam_after": []}
            next_states = []
            for form, tag, score, path, dist, anchor in init_states:
                # expand with deltas that strictly reduce distance to target
                for delta in self._expand(tag):
                    new_tag = delta.apply(tag)
                    new_dist = symmetric_diff_size(new_tag, target_tag)
                    if new_dist >= dist:
                        continue
                    for script, rel, support in self.edge_scripts.get(delta, []):
                        out = script.apply(form)
                        if out is None:
                            continue
                        new_score = score * rel
                        step = (delta, script, rel, support, form, out, tag, new_tag)
                        depth_log["expansions"].append({
                            "from_tag": tag, "to_tag": new_tag,
                            "delta": str(delta), "script": str(script),
                            "input": form, "output": out,
                            "rel": rel, "support": support,
                            "prev_score": score, "new_score": new_score,
                            "dist_before": dist, "dist_after": new_dist
                        })
                        new_path = path + [step]
                        if new_tag == target_tag:
                            # store completed path
                            completed.append((new_score, out, new_path, anchor))
                        else:
                            next_states.append((out, new_tag, new_score, new_path, new_dist, anchor))

            # prune beam
            next_states.sort(key=lambda x: x[2], reverse=True)
            # log beam summary
            for f, tg, sc, pth, di, anch in next_states[: self.beam_size]:
                depth_log["beam_after"].append({
                    "tag": tg, "form": f, "score": sc, "distance": di, "anchor": anch,
                    "path_len": len(pth)
                })
            trace["depths"].append(depth_log)
            init_states = next_states[: self.beam_size]
            if not init_states:
                break

        # Sort completed by score
        completed.sort(key=lambda x: x[0], reverse=True)
        for sc, out, path, anchor in completed[:max(nbest, 1)]:
            # Make a user-facing steps list
            steps_readable = [{
                "delta": str(d), "script": str(s), "rel": r, "support": sup,
                "from_tag": t_from, "to_tag": t_to,
                "input": before, "output": after
            } for (d, s, r, sup, before, after, t_from, t_to) in path]
            trace["completed_paths"].append({
                "form": out, "score": sc, "anchor": anchor, "steps": steps_readable
            })

        if completed:
            # Return n-best or single best
            if nbest > 1:
                forms = [out for (sc, out, path, anchor) in completed[:nbest]]
                if return_trace and return_explanation:
                    expls = []
                    for sc, out, path, anchor in completed[:nbest]:
                        expls.append({"anchor": anchor,
                                      "steps": [(str(d), str(s), r, sup) for (d, s, r, sup, _, _, _, _) in path],
                                      "score": sc})
                    return forms, expls, trace
                if return_trace:
                    return forms, trace
                if return_explanation:
                    expls = []
                    for sc, out, path, anchor in completed[:nbest]:
                        expls.append({"anchor": anchor,
                                      "steps": [(str(d), str(s), r, sup) for (d, s, r, sup, _, _, _, _) in path],
                                      "score": sc})
                    return forms, expls
                return forms

            best_score, best_out, best_path, best_anchor = completed[0]
            if return_trace and return_explanation:
                expl = {"anchor": best_anchor,
                        "steps": [(str(d), str(s), r, sup) for (d, s, r, sup, _, _, _, _) in best_path],
                        "score": best_score}
                return best_out, expl, trace
            if return_trace:
                return best_out, trace
            if return_explanation:
                expl = {"anchor": best_anchor,
                        "steps": [(str(d), str(s), r, sup) for (d, s, r, sup, _, _, _, _) in best_path],
                        "score": best_score}
                return best_out, expl
            return best_out

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
                    if best_direct is None or score > best_direct[0]:
                        best_direct = (score, out, [(delta, script, rel, support, f0, out, t0, target_tag)], t0)
        if best_direct:
            sc, out, path, anchor = best_direct
            steps_readable = [{
                "delta": str(d), "script": str(s), "rel": r, "support": sup,
                "from_tag": t_from, "to_tag": t_to,
                "input": before, "output": after
            } for (d, s, r, sup, before, after, t_from, t_to) in path]
            trace["completed_paths"].append({"form": out, "score": sc, "anchor": anchor, "steps": steps_readable})
            if return_trace and return_explanation:
                expl = {"anchor": anchor,
                        "steps": [(str(d), str(s), r, sup) for (d, s, r, sup, _, _, _, _) in path],
                        "score": sc}
                return out, expl, trace
            if return_trace:
                return out, trace
            if return_explanation:
                expl = {"anchor": anchor,
                        "steps": [(str(d), str(s), r, sup) for (d, s, r, sup, _, _, _, _) in path],
                        "score": sc}
                return out, expl
            return out

        # No path found
        if return_trace and return_explanation:
            return None, {"reason": "no_path"}, trace
        if return_trace:
            return None, trace
        if return_explanation:
            return None, {"reason": "no_path"}
        return None

# -------------------------- Debug utilities --------------------------

def format_step(step: dict) -> str:
    return (f"{step['from_tag']} --{step['delta']} / {step['script']}--> {step['to_tag']}  "
            f"'{step['input']}' -> '{step['output']}'  [rel={step['rel']:.3f}, support={step['support']}]")

def print_trace(trace: dict, max_expansions_per_depth: int = 20) -> None:
    print("Anchors:")
    for a in trace["anchor_states"]:
        print(" -", a["tag"], a["form"], "dist", a["distance"])
    for depth in trace["depths"]:
        print(f"\nDepth {depth['depth']}: {len(depth['expansions'])} expansions")
        for e in depth["expansions"][:max_expansions_per_depth]:
            print("  ", e["from_tag"], "->", e["to_tag"], "|", e["delta"], "|", e["script"],
                  "|", e["input"], "→", e["output"],
                  f"| rel={e['rel']:.3f} prev={e['prev_score']:.3f} new={e['new_score']:.3f}",
                  f"| d {e['dist_before']}→{e['dist_after']}")
        if len(depth["expansions"]) > max_expansions_per_depth:
            print(f"  … ({len(depth['expansions']) - max_expansions_per_depth} more)")
        print("  Beam after:")
        for b in depth["beam_after"]:
            print("   ", b["tag"], b["form"], f"score={b['score']:.4f}", f"dist={b['distance']}", f"path_len={b['path_len']}")
    if trace["completed_paths"]:
        print("\nCompleted paths (best first):")
        for i, cp in enumerate(trace["completed_paths"], 1):
            print(f"#{i}: form={cp['form']}  score={cp['score']:.4f}  anchor={cp['anchor']}")
            for st in cp["steps"]:
                print("   ", st["from_tag"], "--", st["delta"], "/", st["script"], "-->", st["to_tag"],
                      "|", st["input"], "→", st["output"], f"(rel={st['rel']:.3f}, supp={st['support']})")
    else:
        print("\nNo completed path found.")

def demo_debug() -> None:
    # Small toy data as before
    from collections import Counter
    lexeme_dict = {
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
            ("<sapk\u00E1t>", frozenset({"Acc"})): 3,
            ("<sapk\u00E1k>", frozenset({"Pl","Nom"})): 2,
            ("<sapk\u00E1kat>", frozenset({"Pl","Acc"})): 1,
        }),
    }
    m = ImplicativePathModel().fit(lexeme_dict)
    forms, expls, trace = m.predict("sapka", frozenset({"Ins"}), return_explanation=True, return_trace=True, nbest=3)
    print("NBEST forms:", forms)
    print_trace(trace)

if __name__ == "__main__":
    demo_debug()
