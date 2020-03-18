import matplotlib.pyplot as plt


def run():
    plt.xlabel("Hot to Cold")
    plt.ylabel("Percent")
    # plt.title(name_graph)
    with open('ASVSPLUNKCOF_Index_Data_Temperature.csv') as file:
        csv_reader = csv.reader(file, delimiter=',')
        for row in csv_reader:
            print(range(3), row)


if __name__ == '__main__':
    run()