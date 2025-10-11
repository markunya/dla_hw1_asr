import Levenshtein

def calc_cer(ref, hyp):
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    dist = Levenshtein.distance(ref, hyp)
    return dist / len(ref)

def calc_wer(ref, hyp):
    ref_words = ref.strip().split()
    hyp_words = hyp.strip().split()
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    dist = Levenshtein.distance(ref_words, hyp_words)
    return dist / len(ref_words)
