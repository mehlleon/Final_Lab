import os
dir = os.path.dirname(__file__)
files_dir = dir + "/mnist_digits_bin/"
number_files = ["0.bin", "1.bin", "2.bin", "3.bin", "4.bin", "5.bin", "6.bin", "7.bin", "8.bin", "9.bin"]

numbers_dir = []
for i in range(len(number_files)):
    numbers_dir.append(files_dir + number_files[i])

current_number = 0
for i in range(10):
    current_dir = numbers_dir[current_number]
    file = open(current_dir, "rb")

    new_file = open(f"{files_dir}{current_number}_new.bin", "wb")
    b = file.read(1)
    odd = True
    while b:
        if odd:  
            new_file.write(b)
        b = file.read(1)
        odd = not odd

    file.close()
    new_file.close()
    current_number += 1

