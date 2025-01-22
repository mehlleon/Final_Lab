import os
dir = os.path.dirname(__file__)
files_dir = dir + "/example/weights/"

layers = 3
biases_dir = []
weights_dir = []
for i in range(layers):
    biases_dir.append(files_dir + f"Gemm{i}_biases.bin")
    weights_dir.append(files_dir + f"Gemm{i}_weights.bin")


def reduce_space(dirs, name:str, save_dir):
    current_number = 0
    for i in range(len(dirs)):
        current_dir = dirs[current_number]
        file = open(current_dir, "rb")

        new_file = open(f"{save_dir}Gemm{current_number}_{name}_new.bin", "wb")
        b = file.read(1)
        data_b = b
        even = False
        sign = b'\x00'
        while b:
            if even:
                if b == b'\x00':
                    sign = b'\x00'
                elif b == b'\xFF':
                    sign = b'\x01'
                else:
                    raise Exception("Invalid byte")
                new_b = bytes([data_b[0] >> 1 | int.from_bytes(sign, "big") << 7])
                new_file.write(new_b)
            else:
                data_b = b
            b = file.read(1)
            even = not even

        file.close()
        new_file.close()
        current_number += 1


reduce_space(weights_dir, "weights", files_dir)
reduce_space(biases_dir, "biases", files_dir)