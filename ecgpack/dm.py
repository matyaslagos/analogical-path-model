def slot_insert(cdy, sentence):
    slot_insert_aux(cdy, sentence)
    for i in range(1, len(sentence)):
        slot_insert_aux(cdy, sentence[i:], count_slots=False)

def slot_insert_aux(cdy, sentence, slotted=False, filler=None, count_slots=True):
    if sentence == ():
        if slotted:
            try:
                cdy['$'].append(filler)
            except:
                cdy['$'] = [filler]
    else:
        # If not slotted, start a slot
        if not slotted:
            try:
                ndy = cdy['_']
                if count_slots:
                    ndy['#'] += 1
            except:
                cdy['_'] = {'$': [], '#': 1}
                ndy = cdy['_']
            for i in range(1, len(sentence)+1):
                slot_insert_aux(ndy, sentence[i:], True, sentence[:i], count_slots)
        # If already slotted, record filler
        elif slotted:
            try:
                cdy['$'].append(filler)
            except:
                cdy['$'] = [filler]
        # Either way, proceed in recording next word
        try:
            ndy = cdy[sentence[0]]
            if count_slots:
                ndy['#'] += 1
        except:
            cdy[sentence[0]] = {'$': [], '#': 1}
            ndy = cdy[sentence[0]]
        slot_insert_aux(ndy, sentence[1:], slotted, filler, count_slots)