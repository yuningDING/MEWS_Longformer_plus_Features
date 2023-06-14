

file = input("log file: >>>")
folds = int(input("folds (10): >>>"))

with open(file, "r") as f:
    lines = f.readlines()
    out = [""] * len(lines)
    fold = -1
    for i, line in enumerate(lines):
        epoch = i % folds

        if epoch == 0:
            fold += 1

        out[(epoch * folds) + fold] = line.replace("QWK: ", "")

    for ol in out:
        print(ol, end="")
