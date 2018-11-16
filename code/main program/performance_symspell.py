import os
from symspellpy.symspellpy import SymSpell, Verbosity
from random import randint

def main():
    dictionary = 'Dictionary_symspell_50_clusters.txt'
    performance_sym = 'performance_sym.txt'
    fout = open(performance_sym,'w')
    word_list = []
    n = 100
    with open(dictionary, 'r') as f:
        for line in f:
            x = line.strip("\n").split(' ')
            word_list.append(x[0])
    initial_capacity = 600
    max_edit_distance_dictionary = 3
    prefix_length = 7
    sym_spell = SymSpell(initial_capacity, max_edit_distance_dictionary,prefix_length)
    term_index = 0
    count_index = 1  
    dictionary_path = os.path.join(os.path.dirname(__file__),dictionary)
    if not sym_spell.load_dictionary(dictionary_path, term_index, count_index):
        print("Dictionary file not found")
        return

    number_of_wrong_char = 2
    number_of_word = 2
    for i in range(n):
        c,w = gen_word(number_of_word,number_of_wrong_char,word_list)
        max_edit_distance_lookup = 2
        suggestion_verbosity = Verbosity.CLOSEST
        input_term = ""
        # print ("correct word : " + c)
        fout.write(c + "\t" + w + "\t")
        suggestions = sym_spell.lookup(w, suggestion_verbosity,max_edit_distance_lookup)
        for suggestion in suggestions:
            fout.write(suggestion.term + " ")
        fout.write("\n")


def gen_word (number_of_word,number_of_wrong_char,word_list):
    correct_word = ""
    wrong_word = ""
    # for i in word_list:
    #     print (i)
    for i in range(number_of_word):
        r = randint(1,len(word_list))-1
        correct_word += word_list[r]
        wrong_word += word_list[r]
    index_rand = []
    if number_of_wrong_char != 0:
        for i in range(number_of_wrong_char):
            # print (len(correct_word))
            index_rand.append(randint(1,len(correct_word))-1)
        for i in range(len(index_rand)):
            s = randint(0,1)
            n = randint(0,20)
            t = ""
            for j in range(len(wrong_word)):
                if j == index_rand[i]:
                    if s == 0:
                        t += chr(ord(wrong_word[j]) + n)
                        
                    else :
                        t += chr(ord(wrong_word[j]) - n)
                else :
                    t += chr(ord(wrong_word[j]))
            wrong_word = t
            # if s == 0:
            #     print (wrong_word[index_rand[i]],chr(ord(correct_word[index_rand[i]]) + n))
            #     wrong_word[index_rand[i]] = chr(ord(correct_word[index_rand[i]]) + n)
            # else :
            #     print (wrong_word[index_rand[i]],chr(ord(correct_word[index_rand[i]]) + n))
            #     wrong_word[index_rand[i]] = chr(ord(correct_word[index_rand[i]]) - n)
    return correct_word,wrong_word

main()