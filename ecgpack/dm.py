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

class ContextNode:
    def __init__(self):
        """Initialize a new node in the context trie.
        
        Attributes:
            children (dict): Maps words/slots to child TrieNodes
            fillers (list): List of word sequences that can fill this context
            count (int): Number of times this context has been seen
        """
        self.children = {}  # Maps words to child nodes
        self.fillers = FillerTrieOfContext()   # Trie of grams that can fill this context
        self.count = 1      # Frequency counter for this context
        
    def record_filler(self, filler):
        self.fillers._record_filler_aux(self.fillers.root, filler)

class ContextTrie:
    def __init__(self):
        """Initialize an empty context trie."""
        self.root = ContextNode()
    
    def insert_context(self, sentence):
        """Insert a sentence into the context trie, recording all possible contexts.
        
        Args:
            sentence (tuple): A tuple of words representing the sentence
        """
        # Insert the full sentence
        self._insert_context_aux(self.root, sentence, counting=True)
        # Insert all suffixes of the sentence (starting from index 1)
        for i in range(1, len(sentence)):
            self._insert_context_aux(self.root, sentence[i:], counting=False)
    
    def _insert_context_aux(self, current_node, sentence, counting, slotted=False, filler=None):
        """Auxiliary method for slot_insert that handles the recursive insertion of contexts.
        
        Args:
            current_node (TrieNode): The current node in the trie
            sentence (tuple): The remaining words to process
            slotted (bool): Whether we're currently processing a slot
            filler (tuple): The words that fill the current slot
            count_slots (bool): Whether to count this context in frequency calculations
        """
        if sentence == ():
            if slotted:
                current_node.record_filler(filler)
            return
        
        # Handle slot creation if slot hasn't yet been added
        if not slotted:
            try:
                slot_node = current_node.children['_']
                if counting:
                    slot_node.count += 1
            except:
                current_node.children['_'] = ContextNode()
                slot_node = current_node.children['_']
            # Record all possible filler lengths
            for i in range(1, len(sentence) + 1):
                self._insert_context_aux(
                    slot_node,
                    sentence[i:],
                    counting,
                    slotted=True,
                    filler=sentence[:i],
                )
        
        # Record fillers if a slot has already been added
        elif slotted:
            current_node.record_filler(filler)
        
        # Continue building the trie with the next word
        next_word = sentence[0]
        try:
            next_node = current_node.children[next_word]
            if counting:
                next_node.count += 1
        except:
            current_node.children[next_word] = ContextNode()
            next_node = current_node.children[next_word]
        
        # Recursive call with the rest of the sentence
        self._insert_context_aux(
            next_node,
            sentence[1:],
            counting,
            slotted,
            filler,
        )
        
    def _get_context_node(self, context):
        current_node = self.root
        for word in context:
            current_node = current_node.children[word]
        return current_node
    
    def get_fillers(self, context):
        """Get the list of grams that have occurred in this context."""
        filler_trie = self._get_context_node(context).fillers
        return filler_trie._get_fillers_aux(filler_trie.root)
            
            

class FillerNodeOfContext:
    def __init__(self):
        self.children = {}  # Maps words to child nodes
        self.count = 0      # Frequency counter for this gram

class FillerTrieOfContext:
    def __init__(self):
        self.root = FillerNodeOfContext()
    
    def _record_filler_aux(self, current_node, filler):
        if filler == ():
            current_node.count += 1
            return
        try:
            child_node = current_node.children[filler[0]]
            self._record_filler_aux(child_node, filler[1:])
        except KeyError:
            current_node.children[filler[0]] = FillerNodeOfContext()
            child_node = current_node.children[filler[0]]
            self._record_filler_aux(child_node, filler[1:])
    
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