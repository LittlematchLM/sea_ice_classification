if __name__ == '__main__':
    with open('1.txt', 'r') as f:
        text = f.read()
    print(text)
    
    list = text.split('\n')
list[1::7]


