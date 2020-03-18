import matplotlib.pyplot as plt
import csv
import numpy as np


def plot_gammas():
    plt.figure(figsize=(20, 10))
    with open('lunarlander_alpha_0.0001_gamma_0.8_epislon_0.1.csv') as file:
        csv_reader = csv.reader(file, delimiter=',')
        for row in csv_reader:
            row = list(map(float, row))
            average_row = average_reward(row, 20)
            length = (len(average_row) * 20) + 1
            plt.plot(range(1, length, 20), average_row, '-')
    with open('lunarlander_alpha_0.0001_gamma_0.9_epislon_0.1.csv') as file:
        csv_reader = csv.reader(file, delimiter=',')
        for row in csv_reader:
            row = list(map(float, row))
            average_row = average_reward(row, 20)
            length = (len(average_row) * 20) + 1
            plt.plot(range(1, length, 20), average_row, '-')
    with open('lunarlander_alpha_0.0001_gamma_0.99_epislon_0.1.csv') as file:
        csv_reader = csv.reader(file, delimiter=',')
        for row in csv_reader:
            row = list(map(float, row))
            average_row = average_reward(row, 20)
            length = (len(average_row) * 20) + 1
            plt.plot(range(1, length, 20), average_row, '-')
    with open('lunarlander_alpha_0.0001_gamma_0.995_epislon_0.1.csv') as file:
        csv_reader = csv.reader(file, delimiter=',')
        for row in csv_reader:
            row = list(map(float, row))
            average_row = average_reward(row, 20)
            length = (len(average_row) * 20) + 1
            plt.plot(range(1, length, 20), average_row, '-')
    with open('lunarlander_alpha_0.0001_gamma_0.999_epislon_0.1.csv') as file:
        csv_reader = csv.reader(file, delimiter=',')
        for row in csv_reader:
            row = list(map(float, row))
            average_row = average_reward(row, 20)
            length = (len(average_row) * 20) + 1
            plt.plot(range(1, length, 20), average_row, '-')
    with open('lunarlander_alpha_0.0001_gamma_0.9999_epislon_0.1.csv') as file:
        csv_reader = csv.reader(file, delimiter=',')
        for row in csv_reader:
            row = list(map(float, row))
            average_row = average_reward(row, 20)
            length = (len(average_row) * 20) + 1
            plt.plot(range(1, length, 20), average_row, '-')
    # plt.legend([r'$\gamma$ = 0.99'], loc='bottom left')
    plt.yticks(range(-300, 300, 50))
    plt.axhline(200, color='r')
    plt.margins(x=0)
    plt.ylabel("Reward")
    plt.xlabel("Episode")
    plt.title("Reward with Variable Gamma\nMoving Average Over 20 Episodes")

    # Put a legend to the right of the current axis
    plt.legend([r'$\gamma$ = 0.8', r'$\gamma$ = 0.9', r'$\gamma$ = 0.99', r'$\gamma$ = 0.995', r'$\gamma$ = 0.999', r'$\gamma$ = 0.9999'], loc='lower center')
    plt.savefig('variable_gamma.png', bbox_inches='tight')
    plt.clf()


def plot_gammas_zoom():
    plt.figure(figsize=(20, 10))
    with open('lunarlander_alpha_0.0001_gamma_0.99_epislon_0.1.csv') as file:
        csv_reader = csv.reader(file, delimiter=',')
        for row in csv_reader:
            row = list(map(float, row))
            average_row = average_reward(row, 20)
            length = (len(average_row) * 20) + 1
            plt.plot(range(1, length, 20), average_row, '-')
    with open('lunarlander_alpha_0.0001_gamma_0.995_epislon_0.1.csv') as file:
        csv_reader = csv.reader(file, delimiter=',')
        for row in csv_reader:
            row = list(map(float, row))
            average_row = average_reward(row, 20)
            length = (len(average_row) * 20) + 1
            plt.plot(range(1, length, 20), average_row, '-')
    with open('lunarlander_alpha_0.0001_gamma_0.999_epislon_0.1.csv') as file:
        csv_reader = csv.reader(file, delimiter=',')
        for row in csv_reader:
            row = list(map(float, row))
            average_row = average_reward(row, 20)
            length = (len(average_row) * 20) + 1
            plt.plot(range(1, length, 20), average_row, '-')
    with open('lunarlander_alpha_0.0001_gamma_0.9999_epislon_0.1.csv') as file:
        csv_reader = csv.reader(file, delimiter=',')
        for row in csv_reader:
            row = list(map(float, row))
            average_row = average_reward(row, 20)
            length = (len(average_row) * 20) + 1
            plt.plot(range(1, length, 20), average_row, '-')
    # plt.legend([r'$\gamma$ = 0.99'], loc='bottom left')
    plt.yticks(range(-300, 300, 50))
    plt.axhline(200, color='r')
    plt.margins(x=0)
    plt.ylabel("Reward")
    plt.xlabel("Episode")
    plt.title("Reward with Variable Gamma\nMoving Average Over 20 Episodes")

    # Put a legend to the right of the current axis
    plt.legend([r'$\gamma$ = 0.99', r'$\gamma$ = 0.995', r'$\gamma$ = 0.999', r'$\gamma$ = 0.9999'], loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('variable_gamma_without_0.9.png', bbox_inches='tight')
    plt.clf()


def plot_min_epsilon():
    plt.figure(figsize=(20, 10))
    with open('lunarlander_alpha_0.0001_gamma_0.999_epislon_0.0.csv') as file:
        csv_reader = csv.reader(file, delimiter=',')
        for row in csv_reader:
            row = list(map(float, row))
            average_row = average_reward(row, 20)
            length = (len(average_row) * 20) + 1
            plt.plot(range(1, length, 20), average_row, '-')
    with open('lunarlander_alpha_0.0001_gamma_0.999_epislon_0.05.csv') as file:
        csv_reader = csv.reader(file, delimiter=',')
        for row in csv_reader:
            row = list(map(float, row))
            average_row = average_reward(row, 20)
            length = (len(average_row) * 20) + 1
            plt.plot(range(1, length, 20), average_row, '-')
    with open('lunarlander_alpha_0.0001_gamma_0.999_epislon_0.1.csv') as file:
        csv_reader = csv.reader(file, delimiter=',')
        for row in csv_reader:
            row = list(map(float, row))
            average_row = average_reward(row, 20)
            length = (len(average_row) * 20) + 1
            plt.plot(range(1, length, 20), average_row, '-')
    with open('lunarlander_alpha_0.0001_gamma_0.999_epislon_0.2.csv') as file:
        csv_reader = csv.reader(file, delimiter=',')
        for row in csv_reader:
            row = list(map(float, row))
            average_row = average_reward(row, 20)
            length = (len(average_row) * 20) + 1
            plt.plot(range(1, length, 20), average_row, '-')
    # plt.legend([r'$\gamma$ = 0.99'], loc='bottom left')
    plt.yticks(range(-300, 300, 50))
    plt.axhline(200, color='r')
    plt.margins(x=0)
    plt.ylabel("Reward")
    plt.xlabel("Episode")
    plt.title("Reward with Variable Minimum Epsilon\nMoving Average Over 20 Episodes")

    # Put a legend to the right of the current axis
    plt.legend([r'$\epsilon$ = 0.0', r'$\epsilon$ = 0.05', r'$\epsilon$ = 0.1', r'$\epsilon$ = 0.2'], loc='lower center')
    plt.savefig('variable_min_epsilon.png', bbox_inches='tight')
    plt.clf()


def plot_alphas():
    plt.figure(figsize=(20, 10))
    with open('lunarlander_alpha_1e-05_gamma_0.999_epislon_0.1.csv') as file:
        csv_reader = csv.reader(file, delimiter=',')
        for row in csv_reader:
            row = list(map(float, row))
            average_row = average_reward(row, 20)
            length = (len(average_row) * 20) + 1
            plt.plot(range(1, length, 20), average_row, '-')
    with open('lunarlander_alpha_0.0001_gamma_0.999_epislon_0.1.csv') as file:
        csv_reader = csv.reader(file, delimiter=',')
        for row in csv_reader:
            row = list(map(float, row))
            average_row = average_reward(row, 20)
            length = (len(average_row) * 20) + 1
            plt.plot(range(1, length, 20), average_row, '-')
    with open('lunarlander_alpha_0.001_gamma_0.999_epislon_0.1.csv') as file:
        csv_reader = csv.reader(file, delimiter=',')
        for row in csv_reader:
            row = list(map(float, row))
            average_row = average_reward(row, 20)
            length = (len(average_row) * 20) + 1
            plt.plot(range(1, length, 20), average_row, '-')
    # plt.legend([r'$\gamma$ = 0.99'], loc='bottom left')
    plt.yticks(range(-300, 300, 50))
    plt.axhline(200, color='r')
    plt.margins(x=0)
    plt.ylabel("Reward")
    plt.xlabel("Episode")
    plt.title("Reward with Variable Alpha\nMoving Average Over 20 Episodes")
    # Shrink current axis by 20%
    # ax = plt.subplot(111)
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    plt.legend([r'$\alpha$ = 0.00001', r'$\alpha$ = 0.0001', r'$\alpha$ = 0.001'],
               loc='lower center')
    plt.savefig('variable_alpha.png', bbox_inches='tight')
    plt.clf()


def average_reward(list, step):
    new_list = []
    average = 0.0
    for i in range(1, len(list) + 1):
        average += list[i-1]
        if i % step == 0:
            new_list.append(average/step)
            average = 0
    return new_list


def plot_final_training():
    plt.figure(figsize=(20, 10))
    with open('lunarlander_alpha_0.0001_gamma_0.999_epislon_0.1.csv') as file:
        csv_reader = csv.reader(file, delimiter=',')
        for row in csv_reader:
            row = list(map(float, row))
            plt.plot(range(len(row)), list(row), '-')
            average_row = average_reward(row, 20)
            length = (len(average_row) * 20) + 1
            plt.plot(range(1, length, 20), average_row, '-')
    # plt.legend([r'$\gamma$ = 0.99'], loc='bottom left')
    plt.yticks(range(-500, 350, 50))
    plt.axhline(200, color='r')
    plt.margins(x=0)
    plt.ylabel("Reward")
    plt.xlabel("Episode")
    plt.title("Training the Agent")
    plt.savefig('final_training.png', bbox_inches='tight')
    plt.clf()


def plot_final_testing():
    with open('lunarlander_test_alpha_0.0001_gamma_0.999_epislon_0.1.csv') as file:
        csv_reader = csv.reader(file, delimiter=',')
        for row in csv_reader:
            row = list(map(float, row))
            plt.plot(range(len(row)), list(row), '-')
        # plt.legend([r'$\gamma$ = 0.99'], loc='bottom left')
        plt.yticks(range(0, 300, 50))
        plt.axhline(200, color='r')
        plt.ylabel("Reward")
        plt.xlabel("Episode")
        plt.title("Testing the Agent")
        # Shrink current axis by 20%
        # ax = plt.subplot(111)
        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        plt.margins(x=0)
        plt.savefig('final_testing.png', bbox_inches='tight')
        plt.clf()


def main():
    #plot_final_training()
    #plot_final_testing()
    #plot_gammas()
    #plot_alphas()
    plot_min_epsilon()


if __name__ == '__main__':
    main()
