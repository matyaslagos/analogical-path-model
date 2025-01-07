from enum import Enum, auto
from collections import defaultdict

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

class ContextNode:
    def __init__(self):
        """Initialize a new node in the context trie.
        
        Attributes:
            children (dict): Maps words/slots to child TrieNodes
            fillers (list): List of word sequences that can fill this context
            count (int): Number of times this context has been seen
        """
        self.children = defaultdict(ContextNode)  # Maps words to child nodes
        self.fillers = defaultdict(int)
        #self.fillers = FillerTrieOfContext()   # Trie of grams that can fill this context
        self.count = 0      # Frequency counter for this context
        
    def record_filler(self, filler, latifix):
        self.fillers._record_filler_aux(self.fillers.root, filler, latifix)

class ContextTrie:
    def __init__(self):
        """Initialize an empty context trie."""
        self.root = ContextNode()
    
    def record_sentence_contexts(self, sentence):
        for i in reversed(range(len(sentence))):
            suffix_node = ContextNode()
            suffix_node.count += 1
            # Insert slot for each filler starting from i
            for j in range(i, len(sentence)+1):
                filler = sentence[i:j]
                current_node = suffix_node.children['_']
                for word in sentence[j:]:
                    current_node.fillers[filler] += 1
                    current_node.count += 1
                    current_node = current_node.children[word]
            # Don't insert slot at i, dynamically lookup suffix trie from i+1
            suffix_node.children[sentence[i]] = dynamic_lookup_table[i+1]
            dynamic_lookup_table[i] = suffix_node
            pass
        
                
    
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
        try:],
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