#!/usr/bin/env python3 
def main():
    fpath = "E:\Work\Thai-Handwriting-Recognition\code\dictionary.txt"
    with open(fpath, encoding="utf-8-sig") as file:
        all_words = file.readlines()
    all_words = [x.strip() for x in all_words] 
    # print (all_words)
    for word in all_words:
        for c in word:
            print (ord(c))

main ()