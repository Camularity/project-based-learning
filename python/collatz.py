def collatz(num_one):
    if num_one % 2 == 0:
        return num_one // 2
    elif num_one % 2 == 1:
        return num_one * 3 + 1
    else:
        return 0

running = True

try:
    print("Enter Number: ")
    input_num_int = int(input())
    result = collatz(input_num_int)
    while result != 1:
        print(result)
        result = collatz(result)
except:
    print("Must enter an integer!")



