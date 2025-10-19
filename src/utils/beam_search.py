import torch
import kenlm
import numpy as np

def logsumexp(a: float, b: float) -> float:
    if a == -np.inf: return b
    if b == -np.inf: return a
    if a > b:
        return a + np.log1p(np.exp(b - a))
    else:
        return b + np.log1p(np.exp(a - b))
    
class Beam:
    def __init__(self, alpha, beta):
        self.p_b = -np.inf
        self.p_nb = -np.inf
        self.lm_state = None
        self.lm_logp = 0.0
        self.last_char = None
        self.word_count = 0
        self.alpha = alpha
        self.beta = beta

    def score(self):
        return logsumexp(self.p_b, self.p_nb) + self.alpha * self.lm_logp + self.beta * self.word_count

def kenlm_start(lm):
    st = kenlm.State()
    lm.BeginSentenceWrite(st)
    return st

def to_char_sequence(word: str) -> str:
    return " ".join(list(word))

def kenlm_advance_word(lm, state, word: str):
    new_state = kenlm.State()
    score_log10 = lm.BaseScore(state, word, new_state)
    return new_state, float(score_log10) * np.log(10.0)

def beam_search_decode_one(
        lp: np.ndarray,
        ind2tok: dict,
        blank_id: int,
        beam_size: int = 20,
        alpha: float = 0.7,
        beta: float = 1.5,
        lm = None,
        space_token: str = " ",
        prune_topk = 50,
    ) -> str:
        T, V = lp.shape

        def create_beam():
            return Beam(alpha, beta)

        beams = {"": create_beam()}
        beams[""].p_b = 0.0

        if lm is not None :
            beams[""].lm_state = kenlm_start(lm)

        for t in range(T):
            if prune_topk is not None and prune_topk < V:
                values = lp[t]
                sorted_indices = np.argsort(values)[::-1]
                topk = sorted_indices[:prune_topk]
            else:
                topk = range(V)

            nb = {}
            def get(prefix: str) -> Beam:
                if prefix not in nb:
                    nb[prefix] = create_beam()
                return nb[prefix]

            for prefix, src in beams.items():
                p = float(lp[t, blank_id])
                dst = get(prefix)
    
                if dst.lm_state is None: 
                    dst.lm_state, dst.lm_logp, dst.last_char, dst.word_count = src.lm_state, src.lm_logp, src.last_char, src.word_count
                p_total = logsumexp(src.p_b, src.p_nb)
                dst.p_b = logsumexp(dst.p_b, p_total + p)

                for c in topk:
                    if c == blank_id: 
                        continue
                    ch = ind2tok[int(c)]
                    if ch == "": 
                        continue

                    p = float(lp[t, c])

                    if src.last_char == ch:
                        dst_same = get(prefix)
                        if dst_same.lm_state is None:
                            dst_same.lm_state = src.lm_state
                            dst_same.lm_logp = src.lm_logp
                            dst_same.last_char = src.last_char
                            dst_same.word_count = src.word_count
                        dst_same.p_nb = logsumexp(dst_same.p_nb, src.p_nb + p)
                        continue

                    new_pref = prefix + ch
                    dst_new = get(new_pref)

                    if dst_new.lm_state is None:
                        dst_new.lm_state = src.lm_state
                        dst_new.lm_logp = src.lm_logp
                        dst_new.word_count = src.word_count
                    dst_new.last_char = ch
                    dst_new.p_nb = logsumexp(dst_new.p_nb, logsumexp(src.p_b, src.p_nb) + p)

                    if lm is not None and ch == space_token and src.last_char not in (space_token, None):
                        prev = prefix.rstrip()
                        last_word = prev.split(" ")[-1] if prev else ""
                        if last_word:
                            new_state, add_lp = kenlm_advance_word(lm, src.lm_state, to_char_sequence(last_word))
                            dst_new.lm_state = new_state
                            dst_new.lm_logp = src.lm_logp + add_lp
                            dst_new.word_count = src.word_count + 1

            beams = dict(sorted(nb.items(), key=lambda kv: kv[1].score(), reverse=True)[:beam_size])

        best_prefix, best_entry = max(beams.items(), key=lambda kv: kv[1].score())

        if lm is not None and best_prefix and not best_prefix.endswith(space_token):
            last_word = best_prefix.split(" ")[-1]
            if last_word:
                new_state, add_lp = kenlm_advance_word(lm, best_entry.lm_state, last_word)
                best_entry.lm_logp += add_lp
                end_state = kenlm.State()
                add_end = lm.BaseScore(new_state, "</s>", end_state) * np.log(10.0)
                best_entry.lm_logp += add_end
                best_entry.word_count += 1

        return " ".join(best_prefix.strip().split())

@torch.no_grad()
def beam_search_decode_lp(
        log_probs: torch.Tensor,
        log_probs_length: torch.Tensor,
        ind2tok: dict,
        blank_id: int,
        beam_size: int = 20,
        alpha: float = 0.7,
        beta: float = 1.5,
        lm = None,
        prune_topk = 50,
    ) -> list[str]:
        log_probs = log_probs.detach().cpu()
        log_probs_length = log_probs_length.detach().cpu()
        outs = []
        for i in range(log_probs.shape[0]):
            T_i = int(log_probs_length[i].item())
            lp_i = log_probs[i, :T_i].numpy()
            hyp = beam_search_decode_one(
                lp=lp_i,
                ind2tok=ind2tok,
                blank_id=blank_id,
                beam_size=beam_size,
                alpha=alpha,
                beta=beta,
                lm=lm,
                space_token=" ",
                prune_topk=prune_topk,
            )
            outs.append(hyp)
        return outs
