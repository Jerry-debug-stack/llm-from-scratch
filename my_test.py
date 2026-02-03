from cs336_basics import tokenizer

if __name__ == "__main__":
    import numpy as np
    array = np.memmap("trained_values/openwebtext_success/encoded_txt.npy",mode="r")
    print(len(array))

