from collections import defaultdict

class SequenceDistributionHash:
    """
    An alternative, more direct (and often more memory-efficient) approach
    than using tries is to store all distribution information in dictionaries
    keyed by the exact sequences (as tuples). This avoids building a large trie
    and may be more efficient if overlapping prefixes are not that common.

    Specifically:
      - We keep a dictionary 'context_map' where context_map[context][filler] = count.
      - We keep a dictionary 'filler_map' where filler_map[filler][context] = count.
      - A 'context' here is any contiguous subsequence of a sentence with one contiguous
        chunk replaced by '_'. The corresponding 'filler' is that chunk.

    As with the trie version, insert_sentence enumerates all subranges to form
    (context, filler) pairs. The lookup methods (fillers, contexts, shared_fillers,
    shared_contexts) just do direct dictionary lookups.
    """

    def __init__(self):
        # context_map[context][filler] = count
        # filler_map[filler][context] = count
        self.context_map = defaultdict(lambda: defaultdict(int))
        self.filler_map = defaultdict(lambda: defaultdict(int))

    def insert_sentence(self, sentence):
        """
        For every contiguous subrange [a..b] of 'sentence', and every sub-chunk [i..j]
        inside that subrange, let the 'filler' = sentence[i..j], and replace that chunk
        with '_' in the 'context' = sentence[a..i-1] + ('_',) + sentence[j+1..b].
        We then record these events in context_map and filler_map.
        """
        n = len(sentence)
        for a in range(n):
            for b in range(a, n):
                # Subrange is sentence[a..b]
                for i in range(a, b + 1):
                    for j in range(i, b + 1):
                        # The filler is the contiguous slice [i..j]
                        filler = sentence[i : j + 1]
                        # The context is everything from [a..b], except [i..j] replaced with '_'
                        context = sentence[a : i] + ("_",) + sentence[j + 1 : b + 1]

                        # Update distribution counts
                        self.context_map[context][filler] += 1
                        self.filler_map[filler][context] += 1

    def fillers(self, context):
        """
        Given a context (a tuple with exactly one '_'), return all filler sequences
        that were seen in that context, with their counts.
        """
        return dict(self.context_map.get(context, {}))

    def contexts(self, filler):
        """
        Given a filler (a tuple of tokens), return all contexts in which it was
        observed, with their counts.
        """
        return dict(self.filler_map.get(filler, {}))

    def shared_fillers(self, context1, context2):
        """
        For two contexts, return a dictionary {filler: min_count} of all fillers
        that have appeared in both contexts, using the minimum of the two counts
        as the shared count.
        """
        fillers1 = self.context_map.get(context1, {})
        fillers2 = self.context_map.get(context2, {})
        common_fillers = set(fillers1.keys()).intersection(fillers2.keys())

        result = {}
        for f in common_fillers:
            result[f] = min(fillers1[f], fillers2[f])
        return result

    def shared_contexts(self, filler1, filler2):
        """
        For two fillers, return a dictionary {context: min_count} of all contexts
        in which both fillers have appeared, using the minimum of the two counts
        as the shared count.
        """
        contexts1 = self.filler_map.get(filler1, {})
        contexts2 = self.filler_map.get(filler2, {})
        common_contexts = set(contexts1.keys()).intersection(contexts2.keys())

        result = {}
        for c in common_contexts:
            result[c] = min(contexts1[c], contexts2[c])
        return result