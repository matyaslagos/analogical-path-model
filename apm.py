from enum import Enum, auto
from collections import defaultdict, namedtuple
from copy import deepcopy
from pprint import pp

# Function implementation

# Import a txt file (containing one sentence per line) as a list whose each
# element is a list of words corresponding to a line in the txt file:
def txt2list(filename):
    """Import a txt list of sentences as a list of lists of words.

    Argument:
        - filename (string), e.g.: 'grimm_corpus.txt'

    Returns:
        - list (of lists of strings), e.g.:
          [['my', 'name', 'is', 'jolán'], ['i', 'am', 'cool'], ..., ['bye']]
    """
    with open(filename, mode='r', encoding='utf-8-sig') as file:
        lines = file.readlines()
    return [tuple(line.strip().split()) for line in lines]

def corpus_setup():
    return txt2list('grimm_full_commas.txt')

def slot_insert(cdy, sentence):
    slot_insert_aux(cdy, sentence)
    for i in range(1, len(sentence)):
        slot_insert_aux(cdy, sentence[i:], count_slots=False)

def slot_insert_aux(cdy, sentence, slotted=False, filler=None, count_slots=True):
    if sentence == ():
        if slotted:
            try:
                cdy['$fillers'].append(filler)
            except:
                cdy['$fillers'] = [filler]
    else:
        # If not slotted, start a slot
        if not slotted:
            try:
                ndy = cdy['_']
                if count_slots:
                    ndy['$count'] += 1
            except:
                cdy['_'] = {'$fillers': [], '$count': 1}
                ndy = cdy['_']
            for i in range(1, len(sentence)+1):
                slot_insert_aux(ndy, sentence[i:], True, sentence[:i], count_slots)
        # If already slotted, record filler
        elif slotted:
            try:
                cdy['$fillers'].append(filler)
            except:
                cdy['$fillers'] = [filler]
        # Either way, proceed in recording next word
        try:
            ndy = cdy[sentence[0]]
            if count_slots:
                ndy['$count'] += 1
        except:
            cdy[sentence[0]] = {'$fillers': [], '$count': 1}
            ndy = cdy[sentence[0]]
        slot_insert_aux(ndy, sentence[1:], slotted, filler, count_slots)

# Class implementation

class SlotStatus(Enum):
    UNSLOTTED = auto()
    SLOTTING = auto()
    PREFIX = auto()
    INFIX = auto()
    SUFFIX = auto()

class FreqNodeQ:
    def __init__(self, label):
        self.label = label
        self.children = {}
        self.count = 0
    
    def get_or_make_branch(self, tuple_of_strings, count_each_word=False):
        current_node = self
        for word in tuple_of_strings:
            if count_each_word:
                current_node.count += 1
            current_node = current_node.get_or_make_child(word)
        current_node.count += 1
        return current_node
    
    def get_or_make_child(self, child_label):
        if child_label not in self.children:
            child_type = type(self)
            self.children[child_label] = child_type(child_label)
        return self.children[child_label]

class FreqTrieQ:
    def __init__(self):
        self.root = FreqNodeQ('~')

class DistrNodeQ(FreqNodeQ):
    def __init__(self, label):
        super().__init__(label)
        self.filler_finder = FreqTrieQ() # Left-to-right left contexts
        self.context_finder = FreqTrieQ() # Right-to-left left contexts

class DistrTrieQ:
    def __init__(self):
        self.root = DistrNodeQ('~')
    
    def insert_distr(self, sentence):
        context_pairs = ((sentence[:i], sentence[i:]) for i in range(len(sentence)))
        for left_context, right_context in context_pairs:
            current_node = self.root
            for word in right_context:
                current_node = current_node.get_or_make_child(word)
                current_node.count += 1
                # Record right-to-left left context
                context_finder_node = current_node.context_finder.root
                context_finder_node.get_or_make_branch(tuple(reversed(left_context)), count_each_word=True)
                # Record all suffixes of left-to-right context
                filler_finder_node = current_node.filler_finder.root
                left_context_suffixes = (left_context[j:] for j in range(len(left_context)))
                for left_context_suffix in left_context_suffixes:
                    filler_finder_node.get_or_make_branch(left_context_suffix)
                
    
    def insert_fillers(self, sentence):
        filler_context_pairs = (
            (sentence[i:j], (sentence[:i], ('_',) + sentence[j:]))
            for i in range(len(sentence))
            for j in range(i+1, len(sentence)+1)
        )
        for filler, context in filler_context_pairs:
            left_context, right_context = context
            filler_node = self.root.get_or_make_branch(filler)
            filler_node.count += 1
            self.rec_add_context(filler_node.contexts.root, reversed(left_context), right_context)
    
    def rec_add_context(self, context_node, rev_left_context, right_context):
        # Non-recursively add right context to current left context
        context_node.get_or_make_branch(right_context, count=True)
        # Process next word of reversed left context or end recursion
        try:
            word = next(rev_left_context)
        except:
            return
        # Recursively extend left context
        self.rec_add_context(context_node.get_or_make_child(word), rev_left_context, right_context)
    
    def insert_contexts(self, sentence):
        filler_right_context_pairs = (
            (sentence[i:j], sentence[j:])
            for i in range(len(sentence))
            for j in range(i+1, len(sentence)+1)
        )
        for filler, right_context in filler_right_context_pairs:
            context_node = self.root
            for word in right_context:
                context_node = context_node.get_or_make_child(word)
                filler_node = context_node.fillers.root.get_or_make_branch(filler)
                filler_node.count += 1
        

class FillerNode:
    def __init__(self, label):
        self.label = label
        self.children = {}
        self.contexts = FreqTrie()
        self.count = 0
    
    def get_or_make_child(self, child_word):
        if child_word not in self.children:
            self.children[child_word] = FillerNode(child_word)
        return self.children[child_word]

class FillerTrie:
    def __init__(self):
        self.root = FillerNode('~')
    
    def get_node(self, string):
        sequence = tuple(string.split())
        current_node = self.root
        for word in sequence:
            if word in current_node.children:
                current_node = current_node.children[word]
            else:
                return
        return current_node
    
    def record_fillers(self, sentence):
        filler_context_pairs = (
            (sentence[i:j], (sentence[:i], ('_',) + sentence[j:]))
            for i in range(len(sentence))
            for j in range(i+1, len(sentence)+1)
        )
        for filler, context in filler_context_pairs:
            left_context, right_context = context
            filler_node = self.root
            # Make branch for filler
            for word in filler:
                filler_node = filler_node.get_or_make_child(word)
            filler_node.count += 1
            context_node = filler_node.contexts.root
            self.rec_add_context(context_node, reversed(left_context), right_context)
    
    def rec_add_context(self, context_node, rev_left_context, right_context):
        # Non-recursively add right context to current left context
        context_node.count += 1
        self.add_right_context(context_node, right_context)
        # Process next word of reversed left context or end recursion
        try:
            word = next(rev_left_context)
        except:
            return
        # Recursively extend left context
        context_node = context_node.get_or_make_child(word)
        self.rec_add_context(context_node, rev_left_context, right_context)
    
    def add_right_context(self, context_node, right_context):
        for word in right_context:
            context_node = context_node.get_or_make_child(word)
            context_node.count += 1
        
        
class ContextNode:
    def __init__(self, label):
        """Initialize a new node in the context trie.
        
        Attributes:
            children (dict): Maps words/slots to child TrieNodes
            fillers (list): List of word sequences that can fill this context
            count (int): Number of times this context has been seen
        """
        self.label = label
        self.children = {}  # Maps words to child nodes
        self.fillers = FreqTrie()
        self.count = 0      # Frequency counter for this context
    
    def get_or_make_child(self, child_word):
        if child_word not in self.children:
            self.children[child_word] = ContextNode(child_word)
        return self.children[child_word]
            
    
    def record_filler(self, filler, latifix):
        self.fillers._record_filler_aux(self.fillers.root, filler, latifix)

class ContextTrie:
    def __init__(self):
        """Initialize an empty context trie."""
        self.root = ContextNode('~')
    
    def get_node(self, string):
        sequence = tuple(string.split())
        current_node = self.root
        for word in sequence:
            if word in current_node.children:
                current_node = current_node.children[word]
            else:
                return
        return current_node
    
    def record_contexts(self, sentence):
        filler_context_pairs = (
            (sentence[i:j], (sentence[:i], ('_',) + sentence[j:]))
            for i in range(len(sentence))
            for j in range(i+1, len(sentence)+1)
        )
        for filler, context in filler_context_pairs:
            left_context, right_context = context
            context_node = self.root
            # TODO: see if non-reversed way is better
            self.insert_right_context(context_node, filler, right_context)
            for word in reversed(left_context):
                context_node = context_node.get_or_make_child(word)
                context_node.count += 1
                self.insert_right_context(context_node, filler, right_context)
    
    def insert_right_context(self, context_node, filler, right_context):
        # Record right context and filler
        for word in right_context:
            context_node = context_node.get_or_make_child(word)
            context_node.count += 1
        # Add filler to each node of right context (incl. slot node)
        filler_node = context_node.fillers.root
        for word in filler:
            filler_node = filler_node.get_or_make_child(word)
        filler_node.count += 1
    
    def dyn_record_contexts(self, sentence):
        prev_children = defaultdict(ContextNode)
        suffixes = (sentence[i:] for i in reversed(range(len(sentence))))
        for suffix in suffixes:
            suffix_trie = ContextTrie()
            # Case where first word of suffix is not in slot:
            # just attach previous trie
            nonslot_node = ContextNode(suffix[0])
            nonslot_node.count += 1
            nonslot_node.children = prev_children
            suffix_trie.root.children[suffix[0]] = nonslot_node
            # Case where first word of suffix is in slot:
            # make trie for each possible (filler, right context) pair
            slot_node = ContextNode('_')
            slot_node.count += 1
            filler_right_context_pairs = (
                (suffix[:j], suffix[j:])         # (filler, right context)
                for j in range(1, len(suffix)+1) # for all possible fillers
            )
            for filler, right_context in filler_right_context_pairs:
                slot_node.fillers[filler] += 1
                curr_node = slot_node
                for word in right_context:
                    if word not in curr_node.children:
                        new_node = ContextNode(word)
                        curr_node.children[word] = new_node
                    curr_node = curr_node.children[word]
                    curr_node.fillers[filler] += 1
            suffix_trie.root.children['_'] = slot_node
            self.rec_merge(self.root, suffix_trie.root)
            prev_children = deepcopy(suffix_trie.root.children)
    
    def rec_merge(self, original_node, new_node):
        original_node.count += 1
        for filler in new_node.fillers:
            original_node.fillers[filler] += 1
        for word, new_child_node in new_node.children.items():
            if word in original_node.children:
                self.rec_merge(original_node.children[word], new_child_node)
            else:
                original_node.children[word] = new_child_node
                original_node.children[word].count += 1
    
    def suffix_tries(self, sentence):
        suffix_trie_lookup_dict = {len(sentence): ContextTrie()}
        for i in reversed(range(len(sentence))):
            suffix_trie = ContextTrie()
            # Case where first word of suffix is not in slot:
            # just attach previous trie
            nonslot_node = ContextNode(sentence[i])
            nonslot_node.count += 1
            nonslot_node.children = deepcopy(suffix_trie_lookup_dict[i+1].root.children)
            suffix_trie.root.children[sentence[i]] = nonslot_node
            # Case where first word of suffix is in slot:
            # make trie for each possible (filler, right context) pair
            slot_node = ContextNode('_')
            slot_node.count += 1
            filler_right_context_pairs = (
                (sentence[i:j], sentence[j:])        # (filler, right context)
                for j in range(i+1, len(sentence)+1) # for all possible fillers
            )
            for filler, right_context in filler_right_context_pairs:
                slot_node.fillers[filler] += 1
                curr_node = slot_node
                for word in right_context:
                    if word not in curr_node.children:
                        new_node = ContextNode(word)
                        curr_node.children[word] = new_node
                    curr_node = curr_node.children[word]
                    curr_node.fillers[filler] += 1
            suffix_trie.root.children['_'] = slot_node
            suffix_trie_lookup_dict[i] = suffix_trie
        del suffix_trie_lookup_dict[len(sentence)]
        return suffix_trie_lookup_dict
    
    def insert_context(self, sentence):
        """Insert a sentence into these context trie, recording all possible contexts.
        
        Args:
            sentence (tuple): A tuple of words representing the sentence
        """
        # Insert the full sentence
        self._insert_context_aux(self.root, sentence)
        # Insert all suffixes of the sentence (starting from index 1)
        for i in range(1, len(sentence)):
            self._insert_context_aux(self.root, sentence[i:])
    
    def _insert_context_aux(self, current_node, sentence, starting=True, slot_status=SlotStatus.UNSLOTTED, filler=None):
        """Auxiliary method for slot_insert that handles the recursive insertion of contexts.
        
        Args:
            current_node (TrieNode): The current node in the trie
            sentence (tuple): The remaining words to process
            slotted (bool): Whether we're currently processing a slot
            filler (tuple): The words that fill the current slot
            count_slots (bool): Whether to count this context in frequency calculations
        """
        if sentence == ():
            if slot_status == SlotStatus.SUFFIX:
                # Exactly these fillers are suffixes of sentence, indicate with
                # "latifix" = either prefix or suffix, from Latin "latus" ~ "side"
                current_node.record_filler(filler, latifix=True)
            elif slot_status == SlotStatus.INFIX:
                current_node.record_filler(filler, latifix=False)
            elif slot_status == SlotStatus.PREFIX:
                current_node.record_filler(tuple(reversed(filler)), latifix=True)
            return
        if filler is not None and len(filler) > 5:
            if slot_status == SlotStatus.INFIX:
                current_node.record_filler(filler, latifix=False)
            elif slot_status == SlotStatus.PREFIX:
                current_node.record_filler(tuple(reversed(filler)), latifix=True)
            return
        # Handle slot creation if slot hasn't yet been added, unless it's a starting
        # word of a suffix of the original sentence
        if slot_status == SlotStatus.UNSLOTTED and (not starting or counting):
            try:
                slot_node = current_node.children['_']
                if counting:
                    slot_node.count += 1
            except:
                current_node.children['_'] = ContextNode()
                slot_node = current_node.children['_']
            # Record all possible filler lengths
            # When filler is not a suffix of original sentence, don't record it
            # for slot node
            slot_status = SlotStatus.SLOTTING if not starting else SlotStatus.PREFIX
            for i in range(1, len(sentence)):
                self._insert_context_aux(
                    slot_node,
                    sentence[i:],
                    counting,
                    False,
                    slot_status,
                    filler=sentence[:i],
                )
            # When filler is suffix of original sentence, record it even for slot node
            self._insert_context_aux(
                slot_node,
                (),
                counting,
                False,
                SlotStatus.SUFFIX,
                filler=sentence
            )
            slot_status = SlotStatus.UNSLOTTED
        
        # Record filler if a slot has already been added
        elif slot_status == SlotStatus.INFIX:
            # Non-prefix and non-suffix fillers
            current_node.record_filler(filler, latifix=False)
        
        elif slot_status == SlotStatus.PREFIX:
            # Exactly these fillers are prefixes, so record it in reverse
            current_node.record_filler(tuple(reversed(filler)), latifix=True)
        
        # If this is a slot node, don't record filler, just set to slotted
        elif slot_status == SlotStatus.SLOTTING:
            slot_status = SlotStatus.INFIX
        
        # Continue building the trie with the next word
        next_word = sentence[0]
        try:
            next_node = current_node.children[next_word]
            next_node.count += 1
        except:
            current_node.children[next_word] = ContextNode()
            next_node = current_node.children[next_word]
        
        # Recursive call with the rest of the sentence
        self._insert_context_aux(
            next_node,
            sentence[1:],
            counting,
            False,
            slot_status,
            filler,
        )
    
    def get_fillers(self, context):
        """Get the list of grams that have occurred in this context."""
        filler_trie = self._get_context_node(context).fillers
        return filler_trie._get_fillers_aux(filler_trie.root)
    
    def _get_context_node(self, context):
        current_node = self.root
        for word in context:
            current_node = current_node.children[word]
        return current_node
    
    def common_fillers(self, context1, context2):
        context_node1 = self._get_context_node(context1)
        context_node2 = self._get_context_node(context2)
        return self._common_fillers_aux(context_node1, context_node2)
    
    def _common_fillers_aux(self, current_node1, current_node2):
        if current_node1.children == {} or current_node2.children == {}:
            return []
        common_
        for word in filter(current_node1.children, lambda x: x in current_node2.children):
            pass
            

class FillerNodeOfContext:
    def __init__(self):
        self.children = {}  # Maps words to child nodes
        self.count = 0      # Frequency counter for this gram

class FillerTrieOfContext:
    def __init__(self):
        self.root = FillerNodeOfContext()
    
    def _record_filler_aux(self, current_node, filler, latifix):
        if filler == ():
            current_node.count += 1
            return
        if latifix:
            current_node.count += 1
        try:
            child_node = current_node.children[filler[0]]
            self._record_filler_aux(child_node, filler[1:], latifix)
        except KeyError:
            current_node.children[filler[0]] = FillerNodeOfContext()
            child_node = current_node.children[filler[0]]
            self._record_filler_aux(child_node, filler[1:], latifix)

    def _get_fillers_aux(self, current_node, path=()):
        if current_node.children == {}:
            return []
        filler_list = []
        for child_word in current_node.children:
            path += (child_word,)
            child_node = current_node.children[child_word]
            filler_list += self._get_fillers_aux(child_node, path)
            if child_node.count > 0:
                filler_list.append((path, child_node.count))
            path = path[:-1]
        return filler_list