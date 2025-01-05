import os
import tqdm
import pandas as pd
import ast
import time
import json


def parse_free_text(text):
    """Parse `free_text` as a list or wrap it in a list if it's plain text."""
    try:
        # Try to parse the value as a Python literal
        parsed_value = ast.literal_eval(text)
        # Ensure the parsed value is a list
        if isinstance(parsed_value, list):
            return parsed_value
        else:
            return [parsed_value]
    except (SyntaxError, ValueError):
        # If parsing fails, wrap the text in a list
        return [text.strip()]



# Helper Functions
def lower_values(expected, response):
    """Normalize values to lowercase for comparison."""
    for k in expected.keys():
        expected[k] = str(expected[k]).lower()
    for k in response.keys():
        response[k] = str(response[k]).lower()
    return expected, response


def correct_prediction(expected, response):
    """Prepare expected and response for comparison."""
    return lower_values(expected, response)


def clean_text(text):
    """Remove all non-alphanumeric characters and convert to lowercase."""
    return ''.join(char.lower() for char in text if char.isalnum())


def predict(self, free_text, row_id):
    """Predict rule instance using the self."""
    model_response = self.get_rule_instance(free_text,row_id,for_eval=True)
    if model_response["is_error"] == True:
        print("---------- model return error ---------: ", model_response)
        return False, False
    rule_instance = model_response["rule_instance"]
    response = rule_instance["params"]
    response["ruleInstanceName"] = rule_instance["ruleInstanceName"]
    response["severity"] = rule_instance["severity"]
    return response, model_response


# def normalize_empty_value(value):
#     """Normalize empty values to a common representation."""
#     if value in [None, "", "null", "None", "none", "empty"] or (isinstance(value, float) and pd.isna(value)):
#         return "EMPTY"
#     return value
#
def normalize_empty_value(value):
    """Normalize empty values to a common representation."""
    if value in [None, '', ' ', " ", "", "unknown", "null", "NULL", "None", "no", "none", "empty", "EMPTY", 0, '0'] + [
        "int", "Int", "String", "string"]:
        return "null"  # Choose a common representation for empty values
    return value


def normalize_values(dictionary):
    """Normalize all values in the dictionary."""
    return {k: normalize_empty_value(v) for k, v in dictionary.items()}


def score_param_type(expected, response, numerical=True):
    """Generic binary score for numerical or verbal parameters."""
    if numerical:
        # Filter numerical values
        bin_expected = {k: v for k, v in expected.items() if is_numerical(v)}
        bin_response = {k: v for k, v in response.items() if is_numerical(v)}
    else:
        # Filter verbal (non-numerical) values
        bin_expected = {k: v for k, v in expected.items() if not is_numerical(v) and k != "ruleInstanceName"}
        bin_response = {k: v for k, v in response.items() if not is_numerical(v) and k != "ruleInstanceName"}
    return int(bin_response == bin_expected)


def is_numerical(value):
    """Check if a value is numerical."""
    try:
        float(value)
        return True
    except ValueError:
        return False


def score_param_type_avg(expected, response, numerical=True):
    """Evaluate average score for verbal (non-numerical) parameters."""
    # Extract verbal (non-numerical) values from expected and response
    if numerical:
        # Filter numerical values
        verbal_keys = [k for k, v in expected.items() if is_numerical(v)]
    else:
        verbal_keys = [k for k, v in expected.items() if
                       not is_numerical(v) and k.lower() != "ruleInstanceName".lower()]

    score = 0

    for k in verbal_keys:
        normalized_expected = normalize_empty_value(expected[k])
        normalized_response = normalize_empty_value(response.get(k, None))
        if normalized_expected == normalized_response:
            score += 1

    return score / len(verbal_keys) if verbal_keys else 0


def score_rule_instance_name(expected, response):
    """Evaluate the correctness of 'ruleInstanceName'."""
    return int(expected["ruleInstanceName"] == response["ruleInstanceName"])


# Helper function to collect errors for each scoring type
def collect_error_data(param_name, row_id, expected, response, free_text, differences, correct_type_name,
                       predict_type_name, actual_type_name):
    """Helper function to create error data entry."""
    return {
        "param_name": param_name,
        "row_id": row_id,
        "differences": json.dumps(differences, indent=2),
        "response": json.dumps(response, indent=4, ensure_ascii=False),
        "expected": json.dumps(expected, indent=4, ensure_ascii=False),
        "free_text": free_text,
        "correct_type_name": correct_type_name,
        "predict_type_name": predict_type_name,
        "actual_type_name": actual_type_name,
    }


def score_binary(expected, response):
    for k, expected_v in expected.items():
        try:
            if normalize_empty_value(response[k]) != normalize_empty_value(expected_v) and k != "ruleInstanceName":
                return 0
        except:
            return 0
    return 1


def sort_response_by_expected(expected, response):
    """
    Sort the response dictionary based on the order of keys in the expected dictionary.
    Append any extra keys in response to the end.
    """
    sorted_response = {key: response[key] for key in expected if key in response}
    # Add extra keys that are not in expected
    extra_keys = {key: response[key] for key in response if key not in expected}
    sorted_response.update(extra_keys)
    return sorted_response


def find_differences(expected, response):
    """
    Find differences between the expected and response dictionaries.
    """
    differences = {
        "mismatched_keys": {},
        "missing_keys": [],
        "extra_keys": []
    }

    for key in expected:
        if key in response and expected[key] != response[key]:
            differences["mismatched_keys"][key] = {"expected": expected[key], "response": response[key]}
    differences["missing_keys"] = [key for key in expected if key not in response]
    differences["extra_keys"] = [key for key in response if key not in expected]

    return differences




# Evaluation Function
def evaluate_func(
        self,
        data_file_path,
        output_directory,
        start_point=0,
        end_point=2,  # None - all the data
        sleep_time_each_10=30,
        batch_size=250
):
    df_eval = pd.read_csv(data_file_path)
    # Parse `excepted_response` and `free_text`
    try:
        import ast

        def safe_literal_eval(val):
            try:
                return ast.literal_eval(val)
            except (ValueError, SyntaxError):
                return {}  # Return empty dict for malformed entries

        df_eval["expected_response"] = (
            df_eval["expected_response"]
            .str.replace("'", '"')  # Fix quotes
            .replace({"None": "{}", "N/A": "{}"})  # Replace invalid entries
            .apply(safe_literal_eval))  # Evaluate safely
    except:
        df_eval["expected_response"] = df_eval["expected_response"].apply(
        dict)  # Convert strings to dictionaries

    df_eval["free_text"] = df_eval["free_text"].apply(parse_free_text)  # Handle plain strings and lists

    # Create eval_data_generation from the DataFrame
    eval_data_generation = [
        (row["id"], row["rule_types_names"], row["expected_response"], row["free_text"])
        for _, row in df_eval.iterrows()
    ]

    rows = []
    error_df_param_numerical_binary_score = []
    error_df_param_verbal_binary_score = []
    error_df_rule_name_score = []

    # Ensure output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for i, (row_id, type_name, expected, free_text_list) in tqdm.tqdm(enumerate(eval_data_generation[start_point:end_point]),
                                                                      total=len(eval_data_generation[start_point:end_point])):
        # print(f"Processing row {i + 1}/{len(eval_data_generation)}")

        if not i % 10:
            time.sleep(sleep_time_each_10)

        for free_text in free_text_list:
            # print(f'Processing free_text: {free_text}')
            if not free_text.strip():
                print("Skipping empty free_text")
                continue

            try:
                response, rig_response = predict(self, free_text, row_id)
                if not rig_response:
                    print("Error: rig_response is None")

                else:
                    # print(i, rig_response)
                    pass

                # Normalize and sort response
                expected, response = correct_prediction(expected, response)
                expected = normalize_values(expected)
                response = normalize_values(response)
                response = sort_response_by_expected(expected, response)

                # Extract examples from rig_response
                examples = rig_response.get("examples", {})
                validation_score = rig_response.get("validation_score", -1)

                # Calculate scores
                binary_score_no_rule_instance = score_binary(expected, response)
                param_numerical_binary_score = score_param_type(expected, response, numerical=True)
                param_verbal_binary_score = score_param_type(expected, response, numerical=False)
                param_numerical_avg_score = score_param_type_avg(expected, response, numerical=True)
                param_verbal_avg_score = score_param_type_avg(expected, response, numerical=False)
                rule_name_score = score_rule_instance_name(expected, response)
                binary_score = 1 if rule_name_score and binary_score_no_rule_instance else 0
                correct_type_name = int(clean_text(rig_response["type_name"]) == clean_text(type_name))

                # Handle differences
                differences = "None"
                if binary_score == 0:
                    differences = find_differences(expected, response)
                if param_numerical_binary_score == 0:
                    error_df_param_numerical_binary_score.append(
                        collect_error_data('param_numerical_binary_score', row_id, expected, response, free_text,
                                           differences, correct_type_name, rig_response["type_name"], type_name))
                if param_verbal_binary_score == 0:
                    error_df_param_verbal_binary_score.append(
                        collect_error_data('param_verbal_binary_score', row_id, expected, response, free_text,
                                           differences, correct_type_name, rig_response["type_name"], type_name))
                if rule_name_score == 0:
                    error_df_rule_name_score.append(
                        collect_error_data('score_rule_instance_name', row_id, expected, response, free_text,
                                           differences, correct_type_name, rig_response["type_name"], type_name))

                # Append row
                new_row = {
                    "id": row_id,
                    "binary_score": binary_score,
                    "validation_score": validation_score,
                    "binary_score_no_rule_instance": binary_score_no_rule_instance,
                    "param_numerical_binary_score": param_numerical_binary_score,
                    "param_verbal_binary_score": param_verbal_binary_score,
                    "param_numerical_avg_score": param_numerical_avg_score,
                    "param_verbal_avg_score": param_verbal_avg_score,
                    "score_rule_instance_name": rule_name_score,
                    "differences": json.dumps(differences, indent=2),
                    "response": json.dumps(response, indent=4, ensure_ascii=False),
                    "expected": json.dumps(expected, indent=4, ensure_ascii=False),
                    "free_text": free_text,
                    "examples": examples,
                    "correct_type_name": correct_type_name,
                    "predict_type_name": rig_response["type_name"],
                    "actual_type_name": type_name,
                }
                rows.append(new_row)
            except Exception as e:
                print(f"Error processing row {i + 1}, free_text: {free_text}, Error: {e}")
                rows.append({
                    "id": row_id,
                    "binary_score": 0,
                    "binary_score_no_rule_instance": 0,
                    "param_numerical_binary_score": 0,
                    "param_verbal_binary_score": 0,
                    "param_numerical_avg_score": 0,
                    "param_verbal_avg_score": 0,
                    "score_rule_instance_name": 0,
                    "response": "error",
                    "expected": json.dumps(expected, indent=4, ensure_ascii=False),
                    "free_text": free_text,
                    "correct_type_name": 0,
                })

        # Write results and calculate accuracy after batch
        if (i + 1) % batch_size == 0 or (i + 1) == len(eval_data_generation):
            print(f"Writing results after processing {i + 1} rows...")
            write_results(rows, error_df_param_numerical_binary_score, error_df_param_verbal_binary_score,
                          error_df_rule_name_score, output_directory)
            calculate_and_save_accuracy(rows, output_directory)
    write_results(rows, error_df_param_numerical_binary_score, error_df_param_verbal_binary_score,
                  error_df_rule_name_score,output_directory)
    calculate_and_save_accuracy(rows)
    return rows


# Writing results to CSV files
def write_results(rows, numerical_errors, verbal_errors, rule_name_errors, output_directory="output"):
    df_results = pd.DataFrame(rows)
    df_error_param_numerical_binary_score = pd.DataFrame(numerical_errors)
    df_error_param_verbal_binary_score = pd.DataFrame(verbal_errors)
    df_error_rule_name_score = pd.DataFrame(rule_name_errors)

    file_path = generate_unique_filename(output_directory, "test_results")
    df_results.to_csv(file_path, index=False)

    file_path = generate_unique_filename(output_directory, "error_param_numerical_binary_score")
    df_error_param_numerical_binary_score.to_csv(file_path, index=False)

    file_path = generate_unique_filename(output_directory, "error_param_verbal_binary_score")
    df_error_param_verbal_binary_score.to_csv(file_path, index=False)

    file_path = generate_unique_filename(output_directory, "error_rule_name_score")
    df_error_rule_name_score.to_csv(file_path, index=False)


# Calculating accuracy and saving it to a text file
def calculate_and_save_accuracy(rows, output_directory="output"):
    df_results = pd.DataFrame(rows)
    accuracy_results = calculate_accuracy(df_results[df_results["correct_type_name"] == 1])
    accuracy_results_2 = calculate_accuracy(df_results)

    file_path = generate_unique_filename(output_directory, "accuracy_results", 'txt')
    with open(file_path, "w") as file:
        file.write("without classification mistakes:\n Average Accuracy Metrics:\n")
        for metric, value in accuracy_results.items():
            file.write(f"{metric}: {value:.2%}\n")

        file.write("with all the data:\n Average Accuracy Metrics:\n")
        for metric, value in accuracy_results_2.items():
            file.write(f"{metric}: {value:.2%}\n")


def calculate_accuracy(df):
    """
    Calculate the average accuracy for each scoring parameter.
    """
    accuracy_metrics = {
        "binary_score": df["binary_score"].mean(),
        "binary_score_no_instance_name": df["binary_score_no_rule_instance"].mean(),
        "param_numerical_binary_score": df["param_numerical_binary_score"].mean(),
        "param_numerical_avg_score": df["param_numerical_avg_score"].mean(),
        "param_verbal_binary_score": df["param_verbal_binary_score"].mean(),
        "param_verbal_avg_score": df["param_verbal_avg_score"].mean(),
        "score_rule_instance_name": df["score_rule_instance_name"].mean(),
        "classification score": df["correct_type_name"].mean()
    }

    # Print the results
    print("\nAverage Accuracy Metrics:")
    for metric, value in accuracy_metrics.items():
        print(f"{metric}: {value:.2%}")

    return accuracy_metrics


def generate_unique_filename(directory, base_name, extension="csv"):
    """
    Generate a unique filename by appending a sequential number.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)  # Create the directory if it doesn't exist
    # Start with no number
    i = 1
    while True:
        filename = f"{base_name}_{i}.{extension}"
        full_path = os.path.join(directory, filename)
        if not os.path.exists(full_path):
            return full_path  # Return the first available filename
        i += 1


def embed_queries_for_eval(data_file_path):
    def parse_free_text(text):
        """Parse ⁠ free_text ⁠ as a list or wrap it in a list if it's plain text."""
        try:
            parsed_value = ast.literal_eval(text)
            if isinstance(parsed_value, list):
                return parsed_value
            else:
                return [parsed_value]
        except (SyntaxError, ValueError):
            return [text.strip()]

    df = pd.read_csv(data_file_path)
    # free_text_list = df["free_text","expected_response"].tolist()
    df["expected_response"] = df["expected_response"].apply(ast.literal_eval)  # Convert strings to dictionaries
    df["free_text"] = df["free_text"].apply(parse_free_text)  # Handle plain strings and lists

    free_text_list = [
        (row["id"], row["free_text"], row["expected_response"], row["rule_types_names"])
        for _, row in df.iterrows()
    ]
    for row_id, free_text, expected, type_name in free_text_list:
        responses = {"row_id": row_id, "free_text": free_text, "model_response": expected, "type_name": type_name}
        log_question_and_answer(responses)
        # print(row_id, type_name)