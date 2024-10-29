# Function implementation

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

class TrieNode:
    def __init__(self):
        """
        Initialize a new node in the context trie.
        
        Attributes:
            children (dict): Maps words/slots to child TrieNodes
            fillers (list): List of word sequences that can fill this context
            count (int): Number of times this context has been seen
        """
        self.children = {}  # Maps words to child nodes
        self.fillers = []   # List of sequences that can fill this slot
        self.count = 1      # Frequency counter for this context

class ContextTrie:
    def __init__(self):
        """Initialize an empty context trie."""
        self.root = TrieNode()
    
    def slot_insert(self, sentence):
        """
        Insert a sentence into the context trie, recording all possible contexts.
        
        Args:
            sentence (tuple): A tuple of words representing the sentence
        """
        # Insert the full sentence
        self._slot_insert_aux(self.root, sentence)
        # Insert all suffixes of the sentence (starting from index 1)
        for i in range(1, len(sentence)):
            self._slot_insert_aux(self.root, sentence[i:], count_slots=False)
    
    def _slot_insert_aux(self, current_node, sentence, slotted=False, filler=None, count_slots=True):
        """
        Auxiliary method for slot_insert that handles the recursive insertion of contexts.
        
        Args:
            current_node (TrieNode): The current node in the trie
            sentence (tuple): The remaining words to process
            slotted (bool): Whether we're currently processing a slot
            filler (tuple): The words that fill the current slot
            count_slots (bool): Whether to count this context in frequency calculations
        """
        if sentence == ():
            if slotted:
                current_node.fillers.append(filler)
            return
        
        # Handle slot creation if slot hasn't yet been added
        if not slotted:
            try:
                slot_node = current_node.children['_']
                if count_slots:
                    slot_node.count += 1
            except:
                current_node.children['_'] = TrieNode()
                slot_node = current_node.children['_']
            # Record all possible filler lengths
            for i in range(1, len(sentence) + 1):
                self._slot_insert_aux(
                    slot_node,
                    sentence[i:],
                    True,
                    sentence[:i],
                    count_slots
                )
        
        # Record fillers if a slot has already been added
        elif slotted:
            current_node.fillers.append(filler)
        
        # Continue building the trie with the next word
        next_word = sentence[0]
        try:
            next_node = current_node.children[next_word]
            if count_slots:
                next_node.count += 1
        except:
            current_node.children[next_word] = TrieNode()
            next_node = current_node.children[next_word]
        
        # Recursive call with the rest of the sentence
        self._slot_insert_aux(
            next_node,
            sentence[1:],
            slotted,
            filler,
            count_slots
        )