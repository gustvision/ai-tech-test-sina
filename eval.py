
import argparse

def main():
    """Gets the trained model and test data, produces a csv file similar to the one for training"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='WhaleModelEvaluation',
                    description='Given an input, and a pytorch model, this program provides outputs of the model for the test set.',
                    )
    parser.add_argument('-m', '--model_path', type=str, default="",
                        help='The path to the saved model (jit version only)')
    parser.add_argument('-j', '--json_path', type=str, default="",
                        help='The path to the json file containing information about the data')

    args = parser.parse_args()
    main(args.model_path, args.input_audio)



