def best_scoring_sequence(length):
    # Transition matrix
    transitions = {
        'B': {'B': 0.1, 'K': 0.325, 'O': 0.25, '-': 0.325},
        'K': {'B': 0.4, 'K': 0.0,   'O': 0.4,  '-': 0.2},
        'O': {'B': 0.2, 'K': 0.2,   'O': 0.2,  '-': 0.4},
        '-': {'B': 1,   'K': 0.0,   'O': 0.0,  '-': 0.0}
    }

    alphabet = ['B', 'K', 'O']
    prob_table = {('B', 1): 1.0, ('K', 1): 0.0, ('O', 1): 0.0}
    path_tracker = {}

    # Dynamic programming over sequence length
    for pos in range(2, length + 1):
        for curr_char in alphabet:
            best_prob = 0.0
            best_prev = None
            for prev_char in alphabet:
                prev_prob = prob_table.get((prev_char, pos - 1), 0.0)
                trans_prob = transitions[prev_char][curr_char]
                current = prev_prob * trans_prob
                if current > best_prob:
                    best_prob = current
                    best_prev = prev_char
            prob_table[(curr_char, pos)] = best_prob
            path_tracker[(curr_char, pos)] = best_prev

    # Find final character before end state '-'
    final_char = None
    final_prob = 0.0
    for char in alphabet:
        prob = prob_table[(char, length)] * transitions[char]['-']
        if prob > final_prob:
            final_prob = prob
            final_char = char

    # Reconstruct the sequence
    sequence = [final_char]
    for step in range(length, 1, -1):
        sequence.append(path_tracker[(sequence[-1], step)])

    return final_prob, ''.join(reversed(sequence))

if __name__ == '__main__':
    print(best_scoring_sequence(5))