# Tree-printing functions

def ppt(tree, coords=(), ones_out=[]):
    if isinstance(tree, str):
        print(bars(coords, ones_out) + tree)
    else:
        ppt(tree[0], coords + (0,), ones_out)
        ppt(tree[1], coords + (1,), ones_out + [len(coords) + 1])

def bars(coords, ones_out):
    bar_tup = ()
    stop = False
    for i, coord in enumerate(reversed(coords)):
        if coord == 0 and not stop:
            if i != 0:
                bar_tup += ('┌',)
            else:
                bar_tup += ('┌ ',)
        elif not stop:
            stop = True
            if i != 0:
                bar_tup += ('└',)
            else:
                bar_tup += ('└ ',)
        elif (len(coords) - i) in ones_out:
            bar_tup += (' ',)
        else:
            bar_tup += ('│',)
    return ''.join(reversed(bar_tup))

def ppt_annot(tree, coords=(), ones_out=[], annots=[], annot_lengths=[-2]):
    """Nicely prints an annotated binary tree structure.
    
    Argument:
        - tree (tuple): annotated binary tree structure, i.e.
                        either (string1, string2),
                        or ((string1, annot_tree1), (string1, annot_tree1)), where
                        - `string`s are analogical substitutes for `annot_tree`s, and
                        - `annot_tree`s are annotated binary tree structures, e.g.
                        (
                            ('the', 'this'),
                            ('gardener', (('proud', 'nice'), ('queen', 'king')))
                        )
    
    Effect:
        - Nice sideways printing of tree structure with annotations as node labels.
    """
    if isinstance(tree, str):
        print(bars_annot(coords, ones_out, annots, annot_lengths) + '╶─ ' + tree)
    else:
        ppt_annot(tree[0][1], coords + (0,), ones_out,
                  annots + [tree[0][0]], annot_lengths + [len(tree[0][0])])
        ppt_annot(tree[1][1], coords + (1,), ones_out + [len(coords) + 1],
                  annots + [tree[1][0]], annot_lengths + [len(tree[1][0])])

def bars_annot(coords, ones_out, annots, annot_lengths):
    bar_tup = ()
    stop = False
    for i, coord in enumerate(reversed(coords)):
        annot_length = annot_lengths[:len(coords)][-(i+1)] + 2
        annot = annots[-(i+1)]
        annot_form = '(' + annot + ')'
        if coord == 0 and not stop:
            if i != 0:
                bar_tup += (annot_form, '┌╴')
            else:
                bar_tup += (annot_form, '┌╴')
        elif not stop:
            stop = True
            if i != 0:
                bar_tup += (annot_form, '└╴', annot_length * ' ')
            else:
                bar_tup += (annot_form, '└╴', annot_length * ' ')
        elif (len(coords) - i) in ones_out:
            bar_tup += ('  ', annot_length * ' ')
        else:
            bar_tup += ('│ ', annot_length * ' ')
    return ''.join(reversed(bar_tup))

def sum_paths(path_dict):
    sums_dy = defaultdict(float)
    for item in path_dict:
        #sums_dy[item['subst'][0] + item['subst'][1]]
        #sums_dy[item['path']]
        sums_dy[item['path']] += item['score']
    return sorted(list(sums_dy.items()), key=lambda x: x[1], reverse=True)