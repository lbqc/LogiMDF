import re


operator_precedence = {
    '¬': 4, 
    '∧': 3, 
    '∨': 2, 
    '→': 1,
    '#': -1
}

symbols = {'∧', '∨', '→', '¬', '(', ')'}

token2desc = {
    '∧': {'and'}, 
    '∨': {'or'}, 
    '→': {'imply'},
}


def clean_fol(fol, stopwords):
    """
    Clean the input text by handling specific patterns and removing stopwords.
    ∃x(is Trump(x) ∧ ∀y(is fired(y) ∧ is Trump(x))) → has an opposed attitude towards y(speaker, Trump) → Opposed 
    ( is Trump ∧ ( is fired ∧ is Trump ) ) → has an opposed attitude towards y → Opposed

    Args:
    fol (str): The text to be cleaned.
    stopwords (list): A list of stopwords to be removed from the text.

    Returns:
    str: The cleaned text.
    """

    # Replace quantifier expressions (e.g., ∀x or ∃y) with the expression followed by ' ¬' 
    processed_text = re.sub(r'(∀,?[a-zA-Z,]*|∃,?[a-zA-Z,]*)', '', fol)
    processed_text = re.sub(r'\((,?[a-zA-Z,]*)\)', '', processed_text)
    processed_text  = processed_text.replace('↔', '→').replace('>>', '→').replace('>','∧').replace('≡', '→')
    
    # Split the text into words, remove stopwords and extra spaces, and then rejoin the words →∧∨¬↔
    processed_text = re.sub(r'([∧∨¬→\(\)])', r' \1 ', processed_text)
    words = [word.strip() for word in processed_text.split()]
    if stopwords is not None:
        words = [word for word in words if word not in stopwords]
    processed_text = ' '.join(words)

    # Replace content within parentheses that includes only letters, numbers, spaces, and specified punctuation,
    # ignoring content that contains special characters.
    processed_text = re.sub(r'\((?![^\(\)]*?[∧∨¬→])([^\(\)])\)', r' \1 ', processed_text)
    
    # Adds a logical AND operator '∧' and a space before '(' if it is not preceded by any logical operators, another '(', or whitespace.
    processed_text = re.sub(r'(?<![∧∨¬→\(])(\s\()', r' ∧ (', processed_text)
    processed_text = re.sub(r'(\)\s)(?![∧∨¬→\)])', r') ∧ ',  processed_text)
    
    # Add 'stopwords' between operators and parentheses if separated by whitespace.
    processed_text = re.sub(r'([∧∨¬→\(])(\s[∧∨→\)])', r'\1 stopwords\2', processed_text)
    
    # Ensures that all parentheses in the text are properly matched, adding missing ones at the beginning or end as necessary.
    right_counter = 0
    left_counter = 0
    for char in processed_text:
        if char == '(':
            right_counter += 1
        elif char == ')':
            right_counter -= 1
            if right_counter < 0:
                left_counter += 1
                right_counter += 1
    left_parens = ['( '] * (left_counter) if left_counter > 0 else []
    right_parens = [' )'] * right_counter if right_counter > 0 else []
    processed_text = ''.join(left_parens) + processed_text + ''.join(right_parens)

    # Split the text into words, remove stopwords and extra spaces, and then rejoin the words
    words = [word.strip() for word in processed_text.split()]
    
    return ' '.join(words)


def convert_infix2suffix(fol: str, operator_precedence=operator_precedence, symbols=symbols):
    """
    Convert an infix first-order logic expression to a suffix expression.
    
    Args:
    - fol (str): The infix first-order logic expression.
    - operator_precedence (dict): Dictionary defining operator precedence.
    - symbols (set): Set of symbols that represent operators and parentheses.
    
    Returns:
    - list: A list representing the suffix expression.
    """
    
    output_queue = []
    operator_stack = ['#']
    word_buffer = []
    
    for token in fol.split(' '):
        if token in symbols:
            if word_buffer:
                output_queue.append(' '.join(word_buffer).strip())
                word_buffer.clear()
            
            if token == ')':
                while operator_stack[-1] != '(':
                    output_queue.append(operator_stack.pop())
                operator_stack.pop()
            elif token == '(':
                operator_stack.append(token)
            else:
                while operator_precedence.get(operator_stack[-1], -1) >= operator_precedence.get(token, -1):
                    output_queue.append(operator_stack.pop())
                
                operator_stack.append(token)
        else:
            word_buffer.append(token)

    if word_buffer:
        output_queue.append(' '.join(word_buffer).strip())

    while operator_stack[-1] != '#':
        output_queue.append(operator_stack.pop())

    return output_queue


def bulid_sgraph_from_fol(fol: str, stopwords=None, symbols=symbols):
    """
    Convert a First-Order Logic (FOL) expression to a graph representation.
    
    Args:
    - fol_expression (str): The input FOL expression.
    - stopwords (list): List of words to be excluded from the FOL expression.
    - symbols (set): Set of symbols that represent operators and parentheses.
    
    Returns:
    - set: Set of nodes representing unique tokens in the FOL expression.
    - set: Set of edges representing relationships between tokens.
    """
    fol = fol.replace('¬', 'not ')
    fol_cleaned = clean_fol(fol, stopwords)
    fol_suffix_list = convert_infix2suffix(fol_cleaned)
    operand_stack = []
    nodes = set()
    edges = set()
    
    for token in fol_suffix_list:
        if token in symbols:
            second_symbol = operand_stack.pop()
            if len(operand_stack) == 0:
                operand_stack.append(second_symbol)
                continue
            first_symbol = operand_stack.pop()
            second_symbol = first_symbol | second_symbol
            first_symbol = token2desc.get(token, {token})
            nodes |= first_symbol
            for ia in first_symbol:
                for ib in second_symbol:
                    edges.add((ia, ib, token))
                    if token != '→':
                        edges.add((ib, ia, token))
            operand_stack.append(first_symbol | second_symbol)
        else:
            operand_stack.append({token})
            nodes.add(token)
            
    return nodes, edges


def bulid_graph_from_fol(fol: str, stopwords=None, symbols=symbols):
    """
    Convert a First-Order Logic (FOL) expression to a graph representation.
    
    Args:
    - fol_expression (str): The input FOL expression.
    - stopwords (list): List of words to be excluded from the FOL expression.
    - symbols (set): Set of symbols that represent operators and parentheses.
    
    Returns:
    - set: Set of nodes representing unique tokens in the FOL expression.
    - set: Set of edges representing relationships between tokens.
    """
    fol = fol.replace('¬', 'not ')
    fol_cleaned = clean_fol(fol, stopwords)
    fol_suffix_list = convert_infix2suffix(fol_cleaned)
    operand_stack = []
    nodes = set()
    edges = set()
    
    for token in fol_suffix_list:
        if token in symbols:
            second_symbol = operand_stack.pop()
            if len(operand_stack) == 0:
                operand_stack.append(second_symbol)
                continue
            first_symbol = operand_stack.pop()
            for ia in first_symbol:
                for ib in second_symbol:
                    edges.add((ia, ib, token))
                    if token != '→':
                        edges.add((ib, ia, token))
            operand_stack.append(first_symbol | second_symbol)
        else:
            operand_stack.append({token})
            nodes.add(token)
            
    return nodes, edges




def bulid_hgraph_from_fol(fol: str, stopwords=None, symbols=symbols):
    """
    Convert a First-Order Logic (FOL) expression to a graph representation.
    
    Args:
    - fol_expression (str): The input FOL expression.
    - stopwords (list): List of words to be excluded from the FOL expression.
    - symbols (set): Set of symbols that represent operators and parentheses.
    
    Returns:
    - set: Set of nodes representing unique tokens in the FOL expression.
    - set: Set of edges representing relationships between tokens.
    """
    fol = fol.replace('¬', 'not ')
    fol_cleaned = clean_fol(fol, stopwords)
    fol_suffix_list = convert_infix2suffix(fol_cleaned)
    pre_relation = '#'
    fol_stack = []
    hyper_edge_nodes = []
    hyper_edge_types = []
    
    for token in fol_suffix_list:
        if token in symbols:
            second_symbol = fol_stack.pop()
            if len(fol_stack) == 0:
                fol_stack.append(second_symbol)
                continue
            first_symbol = fol_stack.pop()
            nodes = first_symbol | second_symbol
            fol_stack.append(nodes)
            if token == pre_relation:
                hyper_edge_nodes[-1] = nodes
            else:
                hyper_edge_nodes.append(nodes)
                hyper_edge_types.append(token)
            pre_relation = token
        else:
            fol_stack.append({token})
            
    return hyper_edge_nodes, hyper_edge_types


if __name__ == "__main__":
    fol = '∀x ((is Donald Trump(x) ∧ is in Back to the Future 2(x) ∧ is compared(x) ∧ is Biff(x) ∧ is in an alternate universe 1985(x)) → opposed) ∧ (∃x(is compared(x) ∧ is Trump(x) ∧ is Biff(x) ∧ is from Back to the Future movie(x)) → Neutral)'
    # node, edge = bulid_hgraph_from_fol(fol)
    nodes, edges = bulid_sgraph_from_fol(fol)
    print(nodes)
    print(edges)
    # node, edge = bulid_hgraph_from_fol(fol)
    # print(node)
    # print(edge)
    