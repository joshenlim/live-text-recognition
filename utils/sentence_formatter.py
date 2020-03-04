# y threshold determines if the words are in the same line
# Am thinking that the threshold value probably has to be adaptive
# Cause will have to consider different image sizes
# But 30 seems to be the most generic and stable so far
y_threshold = 30

# x threshold determines if the words are in the same group
x_threshold = 60

def find_y_intercept(p1, p2):
    # Derive a y = mx + c, based on 2 points
    # p1: Bottom left point of bounding box
    # p2: Bottom right point of bounding box
    # First convert y to negative due to OpenCV2's xy direction

    p1 = [p1[0], -p1[1]]
    p2 = [p2[0], -p2[1]]
    m = (p2[1] - p1[1]) / (p2[0] - p1[0])

    return p1[1] - (m * p1[0])

def format_word(word):
    # Will need y-intercept and x-value of bottom left corner of
    # word to begin grouping words
    return {
      'y_intercept': find_y_intercept(word[1]['vertices'][0], word[1]['vertices'][3]),
      'x_value_left': word[1]['vertices'][0][0],
      'x_value_right': word[1]['vertices'][3][0],
      'word': word[1]['pred_text']
    }

def sentence_formatter_v1(words, debug=False):
    word_groups = {}

    # First sort the words based on x-value
    sorted_words = sorted(words, key=lambda x: x['x_value_left'])
    row_num = 0

    for word in sorted_words:
        if len(word_groups) == 0:
            if debug:
                print("first word is", word['word'])

            word_groups[f'row_{row_num}'] = [word]
            row_num += 1
                      
        else:
            for row in list(word_groups):
                if (abs(word_groups[row][len(word_groups[row]) - 1]['y_intercept'] - word['y_intercept']) < y_threshold):
                    
                    if debug:
                        print(word['word'], word['y_intercept'], "in the same row as", word_groups[row][len(word_groups[row]) - 1]['word'], word_groups[row][len(word_groups[row]) - 1]['y_intercept'])
                    
                    word_groups[row].append(word)
                    break

                else:
                    if int(row.split('_')[1]) == row_num - 1:
                        
                        if debug:
                            print(word['word'], "in a new row")

                        word_groups[f'row_{row_num}'] = [word]
                        row_num += 1

    # Then we sort the rows based on y-intercept
    word_groups = {k:v for k, v in sorted(word_groups.items(), key = lambda x: x[1][0]['y_intercept'], reverse=True)}

    # Now we have to detect which words are clusters and separate accordingly
    # Will need to think of a better data structure to have cluster context
    # Now is just treating it as a new row
    for key in list(word_groups):
        for idx, word in enumerate(word_groups[key]):
            if (idx != len(word_groups[key]) - 1 and
                word['x_value_right'] < word_groups[key][idx + 1]['x_value_left'] and
                abs(word['x_value_right'] - word_groups[key][idx + 1]['x_value_left']) > x_threshold):
                print('different cluster')
                word_groups[f'row_{row_num}'] = [word]
                row_num += 1
                word_groups[key].pop(idx)


    # Finally, simplify the data structure
    res = []
    for item in word_groups.items():
        res.append([x['word'] for x in item[1]])
    
    return res
    
def sentence_prettifier_v1(words):
    print('======================================')
    print('| Detected Texts by Groups           |')
    print('======================================')
    for idx, word in enumerate([' '.join(items) for items in words]):
        print(f'| {idx} | {word:30} |')
    print('======================================')

def format_sentence(words, debug=False):
    formatted_words = list(map(format_word, words.items()))
    res = sentence_formatter_v1(formatted_words, debug)

    if debug:
        print(formatted_words)
        print(res)

    sentence_prettifier_v1(res)