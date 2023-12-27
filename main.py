import fitz  # PyMuPDF
import os
import math
import numpy as np
import csv
import numpy as np


# Define the Softmax function
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


# Define the gradient descent function for Softmax regression
def gradient_descent_softmax(X, y_one_hot, weights, learning_rate, num_iterations):
    m = len(y_one_hot)

    for iteration in range(num_iterations):
        # Calculate the predicted probabilities using Softmax
        predictions = softmax(np.dot(X, weights))

        # Calculate the error
        error = predictions - y_one_hot

        # Update weights using gradient descent
        gradient = np.dot(X.T, error) / m
        weights -= learning_rate * gradient

    return weights


def multi_reader(file_path):
    reader = csv.DictReader(open(file_path))
    x = []
    y = []
    for row in reader:
        x.append([float(value) for key, value in row.items() if (key != 'Representative' and key != 'District')])
        y.append(float(row['District']))
    x = np.array(x)
    y = np.array(y)
    return x, y

def csv_reader(file_path):
    reader = csv.DictReader(open(file_path))
    bill_titles = [key for key in reader.fieldnames if (key != 'Representative' and key != 'Party')]
    x = []
    y = []
    for row in reader:
        x.append([1.0] + [float(value) for key, value in row.items() if (key != 'Representative' and key != 'Party')])
        y.append(float(row['Party']))
    x = np.array(x)
    y = np.array(y)
    return x, y, bill_titles

def personal_quiz_csv_reader(file_path):
    reader = csv.DictReader(open(file_path))
    bill_titles = [key for key in reader.fieldnames if (key != 'Representative' and key != 'Party')]
    x = []
    y = []
    for row in reader:
        x.append([1.0] + [float(value) for key, value in row.items() if (key == 'SHB 1240' or key == 'ESHB 1048')])
                                                                         # or key == '2SHB 1474' or key == 'SB 5768' or
                                                                         # key == '2SSB 5046' or key == 'ESHB 1155' or key == '2SHB 1559'
                                                                         # or key == "SB 5242" or key == "HB 1230")])
        y.append(float(row['Party']))
    x = np.array(x)
    y = np.array(y)
    return x, y, bill_titles


def test_model(theta, x, y):
    # where theta is our guess of thetas based on training, x is the testing data input, y is the solution
    model_guess = []
    correct_preds = 0
    for example in range(len(x)):
        denom = math.exp(np.dot(-theta, x[example]))
        guess = 1 / (1 + denom)
        rounded_guess = round(guess - 0.5) if guess == 0.5 else round(guess)
        model_guess.append(rounded_guess)
        if rounded_guess == y[example]:
            correct_preds += 1

    print(f"I predicted {correct_preds} examples correctly out of the total {len(x)} examples I tested on. "
          f"That means that my accuracy is {correct_preds / len(x)}.")
    return correct_preds / len(x)


def train_model(x, y, repetitions, rate):
    num_features = len(x[0])
    num_examples = len(y)
    theta = np.zeros(num_features)
    for iteration in range(repetitions):
        gradient = np.zeros(len(theta))
        for example in range(num_examples):
            for j in range(num_features):
                gradient[j] += x[example][j] * (y[example] - (1 / (1 + math.exp(np.dot(-theta, x[example])))))
        theta += rate * gradient
    print("I think the theta values should be: ", theta)
    return theta


def calculate_log_likelihood(theta, x, y):
    """ Where theta is the theta prediction by the model based on the training data
        where x is the array of training data inputs
        where y is the array of correct training data ouputs/solutions"""
    my_sum = 0
    for i in range(len(x)):
        my_sum += y[i] * math.log(1 / (1 + math.exp(np.dot(-theta, x[i])))) + ((1 - y[i]) * math.log(1 - (1 / (1 + math.exp(np.dot(-theta, x[i]))))))
    print(f"The log likelihood of the training data is {my_sum}.")



# This function cleans the pdf legislative voting records that are passed in into JSON that can be digitally manipulated.
def extract_table_from_pdf(pdf_path, output_csv_path):
    # Open the PDF file
    pdf_document = fitz.open(pdf_path)
    with open(output_csv_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write the title to the CSV
        title = pdf_document[0].get_text("text").splitlines()[4].strip()
        csv_writer.writerow([title])

        # Get column headers from the next 11 lines
        headers = ['Bill No.', 'Title', 'Veto/Chapter', 'Prime Sponsor', 'Motion', 'Date', 'Vote', 'Time', 'Y-N-A-E',
                   'Result', 'Day/Item']

        # Write column headers to the CSV
        csv_writer.writerow(headers)

        # Iterate through the remaining lines in chunks of 11
        for page_number in range(pdf_document.page_count):
            page = pdf_document[page_number]
            text = page.get_text("text").splitlines()
            if page_number == 0:
                lines = text[16:]  # Skip first 15 lines
            else:
                lines = text[10:]  # Skip first 11 lines
            # Write each chunk as a row in the CSV
            i = 0
            while i < len(lines) and lines[i] != 'V - Veto' and '2023 Roll Call Votes' not in lines[
                i] and "Page " not in lines[i] and "Page " not in lines[i + 1]:
                chunk = lines[i:i + 11]
                # Check if the 3rd line (and subsequent lines) in the chunk is a number
                valid_chunk = True
                try:
                    float(chunk[2])
                except ValueError:
                    valid_chunk = False

                if valid_chunk:
                    csv_writer.writerow(chunk)
                    i += 11
                else:
                    i += 10

    # Close the PDF document
    pdf_document.close()
    return output_csv_path


def summarize_CSVs(csv_directory, output_csv_path):
    # Initialize a dictionary to store the binary matrix
    binary_matrix = {}

    # Iterate through each CSV file in the directory
    for csv_file in os.listdir(csv_directory):
        if csv_file.endswith('.csv'):
            csv_path = os.path.join(csv_directory, csv_file)

            representative_name = ""

            # Initialize a dictionary to store vote values for each Bill No.
            bill_no_binary = {}

            with open(csv_path, 'r') as csv_file:
                csv_reader = csv.reader(csv_file)

                # Extract the representative name from the file name
                representative_name = next(csv_reader)[0]

                # Extract header from the first row
                header = next(csv_reader)
                # Find the indices of 'Bill No.' and 'Vote' columns
                bill_no_index = header.index('Bill No.')
                vote_index = header.index('Vote')

                # Iterate through each row in the CSV
                for row in csv_reader:
                    bill_no = row[bill_no_index]
                    vote = row[vote_index]

                    # Convert the 'Vote' column to binary (1 if 'YEA', 0 otherwise)
                    binary_value = 1 if vote == 'YEA' else 0

                    # Store the binary value for each Bill No.
                    bill_no_binary[bill_no] = binary_value

            # Add the binary values to the main dictionary
            binary_matrix[representative_name] = bill_no_binary

            # Add the binary values to the main dictionary
            binary_matrix[representative_name] = bill_no_binary

    # Get all unique bill names across all representatives
    all_bill_names = set(bill_name for votes in binary_matrix.values() for bill_name in votes)
    # Write to CSV
    with open(output_csv_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)

        # Write header
        header = ['Representative'] + list(all_bill_names) + ['Party']
        csv_writer.writerow(header)

        # Write data
        for rep_name, votes in binary_matrix.items():
            party = 0 if '(R-' in rep_name else (1 if '(D-' in rep_name or 'D -' in rep_name else None)
            row = [rep_name] + [votes.get(bill_name, 0) for bill_name in all_bill_names] + [party]
            csv_writer.writerow(row)

        # # Write header
        # header = ['Representative'] + list(all_bill_names) + ['District']
        # csv_writer.writerow(header)
        #
        # # Write data
        # for rep_name, votes in binary_matrix.items():
        #     region_A = [42, 39, 12, 7, 6, 4, 3, 40, 10, 38]
        #     region_B = [9, 16, 13, 15, 8, 14]
        #     region_C = [17, 49, 18, 20, 19, 35, 24]
        #     if any(str(element) in rep_name for element in region_A):
        #         district = 1
        #     elif any(str(element) in rep_name for element in region_B):
        #         district = 2
        #     elif any(str(element) in rep_name for element in region_C):
        #         district = 3
        #     else:
        #         district = 4
        #     row = [rep_name] + [votes.get(bill_name, 0) for bill_name in all_bill_names] + [district]
        #     csv_writer.writerow(row)


def main():
    # TRAINING DATA
    # scrape data into CSVs
    for i in range(1, 51):
        extract_table_from_pdf(f"documents\VotingRecord ({i}).pdf", f"csvs\Rep_{i}_output_table.csv")

    # summarize CSVs into data vector for probabilistic model
    summarize_CSVs('csvs', 'representative_votes.csv')

    # TESTING DATA
    # scrape data into CSVs
    for i in range(51, 99):
        extract_table_from_pdf(f"documents\VotingRecord ({i}).pdf", f"testing_csvs\Rep_{i}_output_table.csv")
    # summarize CSVs into data vector for probabilistic model
    summarize_CSVs('testing_csvs', 'answer_votes.csv')


    x, y, bill_titles = csv_reader('representative_votes.csv')
    theta = train_model(x, y, 100, 0.00625)
    sorted_thetas = sorted(range(len(theta)), key=lambda i: theta[i], reverse=True)
    print("max_theta value is ", max(sorted_thetas))
    print("bill_titles length is ", len(bill_titles))
    print(sorted_thetas)
    for val in sorted_thetas:
        if val != 0:  # 0 is bias term
            print("theta value is: " + str(theta[val]) + ", bill title is: " + str(bill_titles[val - 1]))
    x, y, bill_titles = csv_reader('answer_votes.csv')
    test_model(theta, x, y)

    x, y, bill_titles = personal_quiz_csv_reader('representative_votes.csv')
    theta = train_model(x, y, 100, 0.00625)
    # sorted_thetas = sorted(range(len(theta)), key=lambda i: theta[i], reverse=True)
    # print("max_theta value is ", max(sorted_thetas))
    # print("bill_titles length is ", len(bill_titles))
    # print(sorted_thetas)
    # for val in sorted_thetas:
    #     if val != 0:  # 0 is bias term
    #         print("theta value is: " + str(theta[val]) + ", bill title is: " + str(bill_titles[val - 1]))
    x, y, bill_titles = personal_quiz_csv_reader('answer_votes.csv')
    test_model(theta, x, y)


    #### FIND GEOGRAPHIC LOCATIONS
    # summarize_CSVs('csvs', 'district_votes.csv')
    # X, Y = multi_reader('district_votes.csv')
    # X = np.array(X)
    # Y = np.array(Y)
    # # Add a column of ones to X for the bias term
    # X = np.c_[np.ones(X.shape[0]), X]
    #
    # # Convert district numbers to a matrix of one-hot encoded vectors
    # num_classes = 4  # Update this to the number of classes you have
    # y_one_hot = np.eye(num_classes)[Y.astype(int) - 1]  # Adjusted to start indexing from 0
    #
    # # Initialize weights
    # weights = np.zeros((X.shape[1], num_classes))
    #
    # # Set hyperparameters
    # learning_rate = 0.00625
    # num_iterations = 900
    #
    # # Train the model
    # weights = gradient_descent_softmax(X, y_one_hot, weights, learning_rate, num_iterations)
    #
    # # Make predictions on a new set of representatives' voting behavior
    # summarize_CSVs('testing_csvs', 'district_votes.csv')
    # X, Y = multi_reader('district_votes.csv')
    # new_representatives = np.array(X)
    # Y = np.array(Y)
    # new_representatives_with_bias = np.c_[np.ones(new_representatives.shape[0]), new_representatives]
    # predictions = softmax(np.dot(new_representatives_with_bias, weights))
    #
    # # Print the predicted probabilities for each class
    # print("Predicted Probabilities:", predictions)
    #
    # # Assign the class with the highest probability as the predicted class
    # predicted_classes = np.argmax(predictions, axis=1) + 1  # Adjusted to start indexing from 1
    # counter = 0
    # for i in range(len(predicted_classes)):
    #     if predicted_classes[i] == Y[i]:
    #         counter += 1
    # print("Predicted Classes:", predicted_classes)
    # print("Actual Classes: ", Y)
    # print("Accuracy: ", counter/len(predicted_classes))

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
