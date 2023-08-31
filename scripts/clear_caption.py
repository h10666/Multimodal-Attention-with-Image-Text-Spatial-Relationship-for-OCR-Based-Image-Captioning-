
# replace with ' '
common_punc = ['&', '"', '!', '#', '$', '°', '/', '(', ')', ';', \
              '*', '+', '?', '£', '“', '”', '=', '\\', '{', '<', '́', \
              '\xa0', '[', ']', '>', '_', '^', '~', 'ø', '`', '’', '|', '‘']
# if not in ocr tokens, then process (concat the words if can match ocr, else split the words)
special_punc = [
    ':', # 11:43
    '.', # 9.0 K.O. www.eme.com 
    '-', # Coca-Cola
    ',', # 1,500
    '@', # xxx@xx
    "'", # xxx's xxx've xxxn't Z'ivo , need replace
    '%', # 75% , need replace
    ]
special_punc_strip = [
    ' :', 
    ' .', 
    ' -', 
    ' @', 
    " '", 
    ' %', 
    ': ', 
    '. ', 
    '- ', 
    '@ ', 
    "' ", 
    ]

keep_words = [
    't-shirt',
    't-shirts',
    'o\'clock',
    'i\'m',
    'do\'s',
    'don\'ts'
]

def clear_ocr(ocr_token):
    for p in common_punc:
        ocr_token = ocr_token.replace(p, '')
    if ocr_token == 'a.':
        pass
    else:
        for p in [':', '.', '-', '@', "'"]:
            ocr_token = ocr_token.strip(p)
    return ocr_token

def clear_cap(caption, ocr_tokens):
    caption = caption.strip()
    if caption.isupper():
        caption = caption.capitalize()

    for p in common_punc:
        caption = caption.replace(p, ' ')
    for s in special_punc_strip:
        caption = caption.strip(s)
    tokens = caption.split(' ')
    for mark in ['', ' ']:
        while mark in tokens:
            tokens.remove(mark)
    
    new_tokens = []
    all_keep_words = keep_words + ocr_tokens
    for it, t in enumerate(tokens):
        if it == 0:
            t = t.lower()
        punc_flag = False
        for p in special_punc:
            if p in t:
                punc_flag = True
                if t.lower() in all_keep_words:
                    new_tokens.append(t)
                elif ''.join(t.split(p)).lower() in ocr_tokens:
                    new_tokens.append(''.join(t.split(p)))
                elif 'n\'t' in t:
                    new_tokens.append(t.split('n\'t')[0])
                    new_tokens.append('not')
                elif '\'s' in t:
                    new_tokens.append(t.split('\'s')[0])
                    new_tokens.append('\'s')
                elif '\'ve' in t:
                    new_tokens.append(t.split('\'ve')[0])
                    new_tokens.append('\'ve')
                elif '\'re' in t:
                    new_tokens.append(t.split('\'re')[0])
                    new_tokens.append('\'re')
                elif '\'ll' in t:
                    new_tokens.append(t.split('\'ll')[0])
                    new_tokens.append('\'ll')
                elif '\'d' in t:
                    new_tokens.append(t.split('\'d')[0])
                    new_tokens.append('\'d')
                elif '%' in t:
                    new_tokens.append(t.split('%')[0])
                    new_tokens.append('%')
                else:
                    split_words = t.split(p)
                    istitle = split_words[0].istitle()
                    if istitle:
                        split_words = [sw.capitalize() for sw in split_words]
                    new_tokens.extend(split_words)
                    break
        if punc_flag == False:
            new_tokens.append(t)
    
    for mark in ['', ' ']:
        while mark in new_tokens:
            new_tokens.remove(mark)
    
    return new_tokens
                

        


