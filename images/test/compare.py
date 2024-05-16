import argparse
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import skimage

def list2dict(lines):
    dict = {}
    for line in lines:
        spl = line.split(",")
        if spl[0] != "filename":
            dict[spl[0]] = spl[1]
    return dict


def comp(r,p):
    y_true = []
    y_pred = []
    real_dict = list2dict(r)
    pred_dict = list2dict(p)
    count =0
    for key in pred_dict:
        if key in real_dict:
            if real_dict[key] == pred_dict[key]:
                # print(key)
                count = count+1
            # else:
            #     img= skimage.io.imread("../images/visualization/" + key.split(".")[0]+".png")
            #     plt.imshow(img)
            #     plt.title("actual: "+str(real_dict[key])+"predicted: "+str(pred_dict[key]))
            #     plt.show()
            y_true.append(real_dict[key])
            y_pred.append(pred_dict[key])

    accuracy = count/(len(pred_dict))

    data = {'y_actual':y_true,
            'y_predicted': y_pred}
    df = pd.DataFrame(data)
    confusion_m = pd.crosstab(df['y_actual'], df['y_predicted'], rownames=['Actual'], colnames=['Predicted'])
    sn.heatmap(confusion_m, annot=True)
    plt.show()

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

