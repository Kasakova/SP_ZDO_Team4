import xmltodict
from run import write_output

if __name__ == "__main__":
    result = {}
    with open('annotations.xml') as fd:
        doc = xmltodict.parse(fd.read())

    for im in doc["annotations"]["image"]:
        count = 0
        try:
            if isinstance(im["polyline"], list):
                for item in im["polyline"]:
                    if item['@label']=='Stitch':
                        count = count+1
            else:
                if im["polyline"]['@label'] == 'Stitch':
                    count = 1
                else:
                    count = 0
            result[im['@name'].split("/")[1]] = count
        except KeyError:
            result[im['@name'].split("/")[1]] = -1

    write_output("anotace.csv", result)
