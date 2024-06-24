import os
import re
import csv

def generate_training_strategy(pretrain, frozen):
    if pretrain == 'pretrained' and frozen == 'frozen':
        return 'Linear Probing'
    elif pretrain == 'pretrained' and frozen == 'unfrozen':
        return 'Full-Parameter Fine-Tuning'
    elif pretrain == 'scratch' and frozen == 'frozen':
        return 'Need to delete'
    elif pretrain == 'scratch' and frozen == 'unfrozen':
        return 'From Scratch'
    else:
        return 'N/A'

def find_accuracy_numbers(folder_path):
    # This function will search recursively for .txt files and extract numbers following the word "accuracy"
    # create a csv file to store the results
    counter = 0
    with open('experiment_results_vote.csv', 'w', newline='') as f:

        writer = csv.writer(f)
        writer.writerow(['Model Name', 'Granularity', 'Magnification', 'Modality', 'Training Strategy', 'Accuracy'])
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".txt"):
                    file_path = os.path.join(root, file)
                    print(file_path)
                    with open(file_path, 'r') as f:
                        all_test_acc = []
                        content = f.read()
                        # Regular expression to find numbers after the word "accuracy"
                        matches = re.findall(r'Test Acc @1:\s*(\d+\.?\d*)', content)
                        if matches:
                            for match in matches:
                                all_test_acc.append(float(match))
                            try:
                                Model_Name, Granularity, Magnification, Modality, Pretrain, Frozen, Vote, _ = file.split('_')
                                if Vote == 'novote':
                                    continue
                                # Write the results to the csv file
                                Training_Strategy = generate_training_strategy(Pretrain, Frozen)
                                writer.writerow([Model_Name, Granularity, Magnification, Modality, Training_Strategy, max(all_test_acc)])
                                counter += 1
                            except ValueError:
                                # This is for the case where the model name is composed of two words
                                try:
                                    Model_Name, Model_Name_two, Granularity, Magnification, Modality, Pretrain, Frozen, Vote, _ = file.split('_')
                                    if Vote == 'novote':
                                        continue
                                    # Write the results to the csv file
                                    Training_Strategy = generate_training_strategy(Pretrain, Frozen)
                                    writer.writerow([Model_Name+Model_Name_two, Granularity, Magnification, Modality, Training_Strategy, max(all_test_acc)])
                                    counter += 1
                                except ValueError:
                                    # This is for FVs
                                    Model_Name, Granularity, Magnification, Modality, Vote, _ = file.split('_')
                                    writer.writerow([Model_Name, Granularity, Magnification, Modality, 'N/A', max(all_test_acc)])
    print('Total number of files: ', counter)

def find_accuracy_numbers_novote(folder_path):
    # This function will search recursively for .txt files and extract numbers following the word "accuracy"
    # create a csv file to store the results
    counter = 0
    with open('experiment_results_novote.csv', 'w', newline='') as f:

        writer = csv.writer(f)
        writer.writerow(['Model Name', 'Granularity', 'Magnification', 'Modality', 'Training Strategy', 'Accuracy'])
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".txt"):
                    file_path = os.path.join(root, file)
                    print(file_path)
                    with open(file_path, 'r') as f:
                        all_test_acc = []
                        content = f.read()
                        # Regular expression to find numbers after the word "accuracy"
                        matches = re.findall(r'Test Acc @1:\s*(\d+\.?\d*)', content)
                        if matches:
                            for match in matches:
                                all_test_acc.append(float(match))
                            try:
                                Model_Name, Granularity, Magnification, Modality, Pretrain, Frozen, Vote, _ = file.split('_')
                                if Vote == 'vote':
                                    continue
                                # Write the results to the csv file
                                Training_Strategy = generate_training_strategy(Pretrain, Frozen)
                                writer.writerow([Model_Name, Granularity, Magnification, Modality, Training_Strategy, max(all_test_acc)])
                                counter += 1
                            except ValueError:
                                # This is for the case where the model name is composed of two words
                                try:
                                    Model_Name, Model_Name_two, Granularity, Magnification, Modality, Pretrain, Frozen, Vote, _ = file.split('_')
                                    if Vote == 'vote':
                                        continue
                                    # Write the results to the csv file
                                    Training_Strategy = generate_training_strategy(Pretrain, Frozen)
                                    writer.writerow([Model_Name+Model_Name_two, Granularity, Magnification, Modality, Training_Strategy, max(all_test_acc)])
                                    counter += 1
                                except ValueError:
                                    # This is for FVs
                                    Model_Name, Granularity, Magnification, Modality, Vote, _ = file.split('_')
                                    writer.writerow([Model_Name, Granularity, Magnification, Modality, 'N/A', max(all_test_acc)])
    print('Total number of files: ', counter)

if __name__ == '__main__':
    folder_path = './'
    find_accuracy_numbers_novote(folder_path)


