from PyBGEnv import qubic

def show_qubic(b):
    s = ""
    for i in range(4):
        for j in range(4):
            for k in range(4):
                idx = 16 * j + 4 * i + k
                if b[idx] == 1:
                    s += "O"
                elif b[idx + 64] == 1:
                    s += "X"
                else:
                    s += "-"
            s += " | "
        s += "\n"
    print(s)

def decode_qubic(b_hash: str):
    idx = 0
    b = qubic.init()
    for c in b_hash:
        code = ord(c)
        for _ in range(8):
            if code & 1 == 1:
                b[idx] = 1
            idx += 1