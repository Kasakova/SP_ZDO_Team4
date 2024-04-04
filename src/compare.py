import argparse


def list2dict(lines):
    dict = {}
    for line in lines:
        spl = line.split(",")
        dict[spl[0]] = spl[1]
    return dict


def comp(r,p):
    real_dict = list2dict(r)
    pred_dict = list2dict(p)
    count =-1
    for key in pred_dict:
        if key in real_dict:
            if real_dict[key]==pred_dict[key]:
                print(key)
                count = count+1
    accuracy = count/(len(pred_dict)-1)
    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('anotace', nargs='?',type=str,default="anotace.csv",help="anotace")
    parser.add_argument('vystup', nargs='?',type=str,default="output.csv",help="vystup algoritmu")
    args = parser.parse_args()
    anot = args.anotace
    vyst = args.vystup
    with open(anot, encoding="UTF8") as f1:
        real= f1.readlines()
    with open(vyst, encoding="UTF8") as f2:
        pred = f2.readlines()
    print(comp(real, pred))

